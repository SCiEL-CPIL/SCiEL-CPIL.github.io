"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Strategy:
        # 1. Use Latin Hypercube Sampling (LHS) for the first 50% of the budget to ensure wide coverage.
        # 2. Identify the non-dominated set from these points.
        # 3. Use the remaining 50% of the budget to perform local Nelder-Mead refinements 
        #    on the most promising non-dominated points (those with best hypervolume contribution or just top K).
        
        n_total = self.budget
        n_lhs = n_total // 2  # 100 evaluations for LHS
        n_refine = n_total - n_lhs # 100 evaluations for refinement
        
        # --- Phase 1: LHS Sampling ---
        # Generate LHS points
        # Standard LHS: divide each dimension into n_lhs equal intervals, and randomly permute the intervals for each dimension.
        # This ensures exactly one point per interval in each dimension.
        
        # We need to be careful with the random seed. The evaluator sets np.random.seed(seed).
        # We should use np.random calls in a deterministic order relative to the seed.
        
        # Generate permutations for each dimension
        perms = [np.random.permutation(n_lhs) for _ in range(dim)]
        
        X_lhs = np.zeros((n_lhs, dim))
        for i in range(dim):
            # Map permutation indices to the interval centers or random points within intervals
            # Using centers is a common simple LHS variant, but random within interval is better for search.
            # Let's use random points within the assigned interval.
            # Interval width: (hi[i] - lo[i]) / n_lhs
            # Interval index: perms[i][j]
            # Point: lo[i] + (perms[i][j] + np.random.rand()) * interval_width
            
            interval_width = (hi[i] - lo[i]) / n_lhs
            # np.random.rand(n_lhs) generates n_lhs random numbers in [0, 1)
            # These are the offsets within the interval
            offsets = np.random.rand(n_lhs)
            X_lhs[:, i] = lo[i] + (perms[i] + offsets) * interval_width
        
        # Evaluate LHS points
        F_lhs = np.array([problem(x) for x in X_lhs])
        
        # Track all evaluated points and their objective values
        # We record every objective vector returned. The evaluator records them.
        # We just need to manage our internal state for selection.
        
        # --- Phase 2: Local Refinement ---
        # Identify non-dominated points from the LHS set.
        # For simplicity and speed, let's pick the top K non-dominated points to refine.
        # K = 10. Each refinement uses (n_refine // K) evaluations.
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            current_front = np.where(n_dom == 0)[0]
            fronts = []
            
            while len(current_front) > 0:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = np.array(next_front)
            
            return fronts

        fronts = fast_non_dominated_sort(F_lhs)
        
        # Get the first front (non-dominated)
        if len(fronts) == 0:
            # Fallback: pick random points if no front found (shouldn't happen)
            refine_indices = np.random.choice(n_lhs, size=min(10, n_lhs), replace=False)
        else:
            # We want to refine points that are likely to improve hypervolume.
            # A simple heuristic: pick the first 10 points from the first front.
            # To make it more robust, we could pick points that are well-spread, but let's stick to simple for now.
            # Actually, let's pick the points in the first front that have the smallest sum of objectives?
            # Or just the first 10.
            front_indices = fronts[0]
            # Shuffle the front indices to add some randomness, or just take the first 10?
            # Taking the first 10 might be biased by order. Let's shuffle them using the RNG.
            np.random.shuffle(front_indices)
            n_to_refine = min(10, len(front_indices))
            refine_indices = front_indices[:n_to_refine]
        
        # Allocate remaining budget to these refinements
        if n_to_refine > 0:
            evals_per_refine = n_refine // n_to_refine
            if evals_per_refine < 5:
                evals_per_refine = 5
                # Adjust n_to_refine if we overshoot budget?
                # If evals_per_refine is too small, reduce n_to_refine
                while n_to_refine * evals_per_refine > n_refine and n_to_refine > 1:
                    n_to_refine -= 1
            else:
                # We might have extra budget. Let's just use the calculated evals_per_refine.
                pass
            
            # Perform Nelder-Mead on each selected point
            # We need a scalar objective function for Nelder-Mead.
            # For multi-objective, we can use a Tchebycheff scalarization or simply minimize the sum of objectives.
            # Given the reference point [3, 5, 7], minimizing the distance to the origin in objective space is a good proxy.
            # Let's use the sum of objectives as the scalar objective.
            
            def scalar_obj(x):
                return np.sum(problem(x))
            
            # Nelder-Mead implementation
            def nelder_mead(func, x0, bounds, max_iter=100):
                n = len(x0)
                # Initial simplex
                simplex = np.zeros((n+1, n))
                simplex[0] = x0
                for i in range(1, n+1):
                    simplex[i] = x0.copy()
                    # Perturb dimension i-1
                    step = (bounds[i-1][1] - bounds[i-1][0]) * 0.05
                    simplex[i, i-1] += step
                    simplex[i, i-1] = np.clip(simplex[i, i-1], bounds[i-1][0], bounds[i-1][1])
                
                # Evaluate initial simplex
                f_simplex = np.array([func(x) for x in simplex])
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                
                for _ in range(max_iter):
                    # Sort simplex by function value
                    order = np.argsort(f_simplex)
                    simplex = simplex[order]
                    f_simplex = f_simplex[order]
                    
                    # Centroid of all but worst point
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
                    f_r = func(x_r)
                    
                    if f_r < f_simplex[0]:
                        # Expansion
                        x_e = centroid + gamma * (centroid - simplex[-1])
                        x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
                        f_e = func(x_e)
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_simplex[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_simplex[-1] = f_r
                    elif f_r < f_simplex[-2]:
                        # Accept reflection
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                    else:
                        # Contraction
                        x_c = centroid + rho * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
                        f_c = func(x_c)
                        if f_c < f_simplex[-1]:
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, n+1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                                f_simplex[i] = func(simplex[i])
                    
                    # Check convergence
                    if np.std(f_simplex) < 1e-10:
                        break
                
                return simplex[0]

            for idx in refine_indices:
                if time.time() > deadline:
                    break
                
                x0 = X_lhs[idx]
                # Run Nelder-Mead with limited iterations
                # evals_per_refine is the budget for this refinement.
                # NM uses n+1 evals per iteration.
                max_iter = evals_per_refine // (dim + 1)
                if max_iter < 1:
                    max_iter = 1
                
                x_opt = nelder_mead(scalar_obj, x0, bounds, max_iter=max_iter)
                # The final point x_opt has been evaluated inside nelder_mead.
                # We don't need to evaluate it again, but it's already in the "recorded" list by the evaluator?
                # Wait, the evaluator records every call to problem().
                # nelder_mead calls problem() many times. So all these are recorded.
                # We just need to ensure we don't exceed the total budget.
                # The logic above allocates evals_per_refine to each.
                # We should probably stop early if we hit the deadline.

        return None
# EVOLVE-BLOCK-END
