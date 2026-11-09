"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using Multi-Start Local Search (Nelder-Mead) with global archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # Safety margin before 330s kill
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Global archive of non-dominated points
        archive_X = []
        archive_F = []
        
        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            if not archive_F:
                return False
            # A point dominates f if it is better in at least one objective and no worse in all
            # We check if there exists a point in archive that dominates f
            # archive_F: shape (N, n_obj)
            # f: shape (n_obj,)
            # Dominates if: all(archive_F <= f) and any(archive_F < f)
            better_or_equal = np.all(archive_F <= f, axis=1)
            strictly_better = np.any(archive_F < f, axis=1)
            return np.any(better_or_equal & strictly_better)

        def update_archive(x, f):
            """Add point to archive if not dominated, and remove dominated points from archive."""
            nonlocal archive_X, archive_F
            if not archive_F:
                archive_X.append(x)
                archive_F.append(f)
                return True
            
            # Check if new point is dominated by existing
            if is_dominated(f, np.array(archive_F)):
                return False
            
            # Remove points in archive dominated by new point
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                # If af dominates f, we already returned False. 
                # Check if f dominates af
                # f dominates af if: all(f <= af) and any(f < af)
                better_or_equal = np.all(f <= af)
                strictly_better = np.any(f < af)
                if not (better_or_equal and strictly_better):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F
            return True

        # Initial diverse points: Corners and Center
        init_points = []
        # Corners
        for i in range(2**dim):
            point = []
            for j in range(dim):
                if (i >> j) & 1:
                    point.append(hi[j])
                else:
                    point.append(lo[j])
            init_points.append(np.array(point))
        
        # Center
        init_points.append((lo + hi) / 2.0)
        
        # Add a few random points
        rng = np.random.default_rng(0)
        for _ in range(5):
            init_points.append(lo + (hi - lo) * rng.random(dim))
        
        # Shuffle init points
        rng.shuffle(init_points)
        
        used = 0
        
        # Multi-Start Loop
        for init_x in init_points:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Evaluate initial point
            f0 = problem(init_x)
            used += 1
            update_archive(init_x, f0)
            
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Nelder-Mead Simplex
            # Initialize simplex
            simplex = [init_x.copy()]
            simplex_F = [f0.copy()]
            
            # Create initial simplex vertices
            for i in range(dim):
                v = init_x.copy()
                # Step size: 5% of range
                step = 0.05 * (hi[i] - lo[i])
                if hi[i] - lo[i] < 1e-6:
                    step = 0.05
                v[i] += step
                v = np.clip(v, lo, hi)
                fv = problem(v)
                used += 1
                simplex.append(v)
                simplex_F.append(fv)
                update_archive(v, fv)
                
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
            
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Nelder-Mead iterations
            # We run a fixed number of iterations or until budget/time runs out
            # To be efficient, we limit iterations per restart
            max_iter = 20
            
            for _ in range(max_iter):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Sort simplex by sum of objectives (scalarization for NM)
                # For multi-objective, we can use a weighted sum or just sum.
                # Sum is a common heuristic for NM in MOLS.
                sums = np.sum(simplex_F, axis=1)
                sort_idx = np.argsort(sums)
                simplex = [simplex[i] for i in sort_idx]
                simplex_F = [simplex_F[i] for i in sort_idx]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, lo, hi)
                f_ref = problem(x_ref)
                used += 1
                update_archive(x_ref, f_ref)
                
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Check if reflection is better than worst
                if np.sum(f_ref) < np.sum(simplex_F[-1]):
                    # Check if it's better than best
                    if np.sum(f_ref) < np.sum(simplex_F[0]):
                        # Expansion
                        gamma = 2.0
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, lo, hi)
                        f_exp = problem(x_exp)
                        used += 1
                        update_archive(x_exp, f_exp)
                        
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        if np.sum(f_exp) < np.sum(f_ref):
                            simplex[-1] = x_exp
                            simplex_F[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            simplex_F[-1] = f_ref
                    else:
                        # Accept reflection
                        simplex[-1] = x_ref
                        simplex_F[-1] = f_ref
                else:
                    # Contraction
                    # Outside contraction if reflection is worse than worst
                    # Inside contraction if reflection is better than worst but not best
                    # Here reflection is worse than worst, so outside contraction?
                    # Standard NM: if f_ref > f_worst, do contraction.
                    # If f_ref < f_worst, accept.
                    # Since f_ref >= f_worst here:
                    rho = 0.5
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    f_con = problem(x_con)
                    used += 1
                    update_archive(x_con, f_con)
                    
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    if np.sum(f_con) < np.sum(simplex_F[-1]):
                        simplex[-1] = x_con
                        simplex_F[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            simplex_F[i] = problem(simplex[i])
                            used += 1
                            update_archive(simplex[i], simplex_F[i])
                            
                            if used >= budget:
                                break
                            if time.time() > deadline:
                                break
                        
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break

        return archive_X
# EVOLVE-BLOCK-END
