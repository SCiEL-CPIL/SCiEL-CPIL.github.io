"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using Halton Initialization + Nelder-Mead Refinement.
    
    Strategy:
    1. Global Exploration: Generate a large set of candidate points using a Halton sequence.
       This provides a low-discrepancy, space-filling distribution in 12D, which is superior
       to random sampling for covering the Pareto front of WFG4.
    2. Evaluation & Filtering: Evaluate a subset of these points to build an initial non-dominated archive.
    3. Local Exploitation: Use the Nelder-Mead simplex method to refine the most promising
       points in the archive. Since WFG4 is deterministic, Nelder-Mead can effectively
       navigate the landscape to find local optima near the true Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        used_evals = 0
        archive_X = []
        archive_F = []
        
        def is_dominated(f1, f2):
            # f1 dominates f2 if f1 <= f2 and f1 < f2
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if new point is dominated by any in archive
            for af in archive_F:
                if is_dominated(af, f):
                    return False
            # Remove any points in archive dominated by new point
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not is_dominated(f, af):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F
            return True

        # 1. Global Exploration: Halton Sequence
        # We want to distribute evaluations broadly.
        # Let's allocate 50% of the budget (100 evals) to global sampling.
        # The Halton sequence generates points in [0,1]^d.
        
        def halton(index, base):
            f = 1
            r = 0
            while index > 0:
                f = f / base
                r = r + f * (index % base)
                index = index // base
            return r

        def halton_sequence(count, dim):
            points = np.zeros((count, dim))
            for i in range(1, count + 1):
                for j in range(dim):
                    points[i-1, j] = halton(i, j + 2) # Bases start from 2
            return points

        n_global = self.budget // 2  # 100 points
        # Generate slightly more to ensure we have enough after clipping? No, Halton is 1:1.
        # Map [0,1]^d to [lo, hi]^d
        halton_points_unit = halton_sequence(n_global, self.dim)
        X_global = lo + range_ * halton_points_unit
        
        # Evaluate global points
        for i in range(n_global):
            if time.time() > deadline:
                break
            x = X_global[i]
            f = problem(x)
            used_evals += 1
            add_to_archive(x, f)
            
        # 2. Local Exploitation: Nelder-Mead
        # We have ~100 evals left.
        # We will pick the best points from the archive and run Nelder-Mead on them.
        # Since it's multi-objective, "best" is ambiguous. 
        # Strategy: Pick points that are extreme in each objective or just spread out.
        # A simple heuristic: Pick the points in the archive that are non-dominated 
        # and have the smallest distance to the ideal point (min of each objective) 
        # or just pick a diverse subset.
        
        # To keep it simple and robust, let's pick the first K points from the archive 
        # that were added (which are likely spread due to Halton order) 
        # and refine them.
        
        # However, Nelder-Mead minimizes a scalar function. 
        # For MO, we can minimize a weighted sum or Tchebycheff function.
        # Given the budget, running full NM (which takes ~dim*evals) is expensive.
        # dim=12. NM usually needs ~10-20 * dim evals per run.
        # 100 evals / 12 dims / 15 factor ~ 0.5 runs? That's too few.
        
        # Alternative: Use a simpler local search or fewer, faster refinements.
        # Let's try a "Cooperative" approach:
        # Select top 5-10 points from the archive based on hypervolume contribution or just random.
        # For each, run a simplified coordinate descent or a few steps of NM.
        
        # Let's try a specific Nelder-Mead implementation that is efficient.
        # We will define a scalar objective as the sum of the normalized objectives 
        # or the Tchebycheff metric to the reference point.
        # Reference point for scoring is [3.0, 5.0, 7.0].
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # We want to minimize the distance to the ideal, but weighted to prefer points 
        # that contribute to hypervolume.
        # A common trick: Minimize the Tchebycheff function to the reference point 
        # for points that are "promising".
        
        # Let's select the 5 most promising points from the archive.
        # "Promising" = non-dominated and close to the knee of the front.
        # We can approximate by minimizing the sum of normalized objectives.
        
        if len(archive_X) == 0:
             return np.array([])
        
        # Normalize objectives to [0,1] range based on current archive min/max? 
        # Or just use raw values? Raw values might be dominated by larger scales.
        # WFG4 objectives are roughly in similar scales.
        
        # Select candidates for refinement
        # Sort archive by sum of objectives (simple scalarization)
        scores = np.sum(archive_F, axis=1)
        sorted_indices = np.argsort(scores)
        
        # Take the top 5 "best" (lowest sum) points
        n_refine = min(5, len(archive_X))
        candidates = [archive_X[i] for i in sorted_indices[:n_refine]]
        
        # Run Nelder-Mead on these candidates
        # We have ~100 evals left.
        # Each NM run on 12D with maxiter=10 might take ~12*2*10 = 240 evals? 
        # Standard NM simplex size is dim+1 = 13 points.
        # Initialization: 13 evals.
        # Iterations: ~10.
        # Total per run: ~13 + 10 = 23 evals.
        # 5 runs * 23 = 115 evals. This is slightly over our remaining 100.
        # Let's reduce to 3 runs or reduce iterations.
        
        n_refine = 3
        candidates = [archive_X[i] for i in sorted_indices[:n_refine]]
        
        for x_center in candidates:
            if time.time() > deadline:
                break
            if used_evals >= self.budget:
                break
            
            # Define scalar objective for NM
            # We use a weighted sum of the objectives. 
            # To encourage hypervolume, we can use weights that reflect the 
            # desirability of each objective. 
            # Simple: Equal weights.
            
            def scalar_obj(x):
                f = problem(x)
                return np.sum(f)
            
            # Nelder-Mead Implementation
            # x0 is the center
            # We need to create an initial simplex.
            # x0 + delta * e_i
            delta = 0.05 * np.max(range_) # Small perturbation
            
            simplex_X = [x_center.copy()]
            simplex_F = [scalar_obj(x_center)]
            used_evals += 1
            
            for i in range(self.dim):
                x_i = x_center.copy()
                # Perturb in dimension i
                if x_center[i] + delta <= hi[i]:
                    x_i[i] += delta
                elif x_center[i] - delta >= lo[i]:
                    x_i[i] -= delta
                else:
                    x_i[i] = (lo[i] + hi[i]) / 2.0
                f_i = scalar_obj(x_i)
                simplex_X.append(x_i)
                simplex_F.append(f_i)
                used_evals += 1
                if used_evals >= self.budget:
                    break
            
            # Run NM iterations
            max_iter = 5 # Keep it small to save budget
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if used_evals >= self.budget:
                    break
                
                # Sort simplex by objective value
                order = np.argsort(simplex_F)
                simplex_X = [simplex_X[i] for i in order]
                simplex_F = [simplex_F[i] for i in order]
                
                # Centroid of all but the worst
                centroid = np.mean(simplex_X[:-1], axis=0)
                worst_x = simplex_X[-1]
                worst_f = simplex_F[-1]
                
                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - worst_x)
                x_r = np.clip(x_r, lo, hi)
                f_r = scalar_obj(x_r)
                used_evals += 1
                
                if f_r < simplex_F[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - worst_x)
                    x_e = np.clip(x_e, lo, hi)
                    f_e = scalar_obj(x_e)
                    used_evals += 1
                    
                    if f_e < f_r:
                        simplex_X[-1] = x_e
                        simplex_F[-1] = f_e
                    else:
                        simplex_X[-1] = x_r
                        simplex_F[-1] = f_r
                elif f_r < simplex_F[-2]:
                    # Accept reflection
                    simplex_X[-1] = x_r
                    simplex_F[-1] = f_r
                else:
                    # Contraction
                    rho = 0.5
                    x_c = centroid + rho * (worst_x - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = scalar_obj(x_c)
                    used_evals += 1
                    
                    if f_c < worst_f:
                        simplex_X[-1] = x_c
                        simplex_F[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex_X)):
                            simplex_X[i] = simplex_X[0] + 0.5 * (simplex_X[i] - simplex_X[0])
                            simplex_F[i] = scalar_obj(simplex_X[i])
                            used_evals += 1
                            if used_evals >= self.budget:
                                break
                        
                        if used_evals >= self.budget:
                            break
                        
                # Add the best point found in this NM run to the archive
                # We check the best point of the current simplex
                best_idx = np.argmin(simplex_F)
                best_x = simplex_X[best_idx]
                best_f_raw = problem(best_x) # Re-evaluate to get vector
                used_evals += 1
                
                add_to_archive(best_x, best_f_raw)

        # Final cleanup: ensure all points in archive are non-dominated
        final_X = []
        final_F = []
        
        for x, f in zip(archive_X, archive_F):
            dominated = False
            for af in final_F:
                if is_dominated(af, f):
                    dominated = True
                    break
            if not dominated:
                new_final_X = []
                new_final_F = []
                for fx, ff in zip(final_X, final_F):
                    if not is_dominated(f, ff):
                        new_final_X.append(fx)
                        new_final_F.append(ff)
                final_X = new_final_X
                final_F = new_final_F
                final_X.append(x)
                final_F.append(f)
        
        return np.array(final_X) if final_X else np.array([])
# EVOLVE-BLOCK-END
