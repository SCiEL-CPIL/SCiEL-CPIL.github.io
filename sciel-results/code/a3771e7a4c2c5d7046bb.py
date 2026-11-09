"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Multi-Start Nelder-Mead Local Search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Generate initial points using Latin Hypercube Sampling for good spread
        # We use a fraction of the budget for initial seeds.
        # Nelder-Mead is expensive per iteration (dim+1 evals per step).
        # Let's estimate cost:
        # If we have B budget, and NM takes ~K steps to converge (K ~ 10-50 depending on dim),
        # and each step costs dim+1 evals, total cost per start ~ K*(dim+1).
        # With dim=5, cost ~ 6*K.
        # If K=20, cost ~ 120.
        # So we can afford ~1-2 starts? That's too few.
        
        # Alternative: Use a cheaper local search or limit NM steps.
        # Let's try a hybrid: 
        # Use a small number of NM runs, but with a strict step limit.
        # Or, use a different local search like Coordinate Descent which is cheaper?
        # NM is robust. Let's stick to NM but be careful with budget.
        
        # Strategy:
        # 1. Generate N_STARTS points via LHS.
        # 2. For each start, run Nelder-Mead with a max step limit.
        # 3. Monitor budget.
        
        # Let's determine N_STARTS dynamically.
        # Cost per NM step = dim + 1 = 6 evals.
        # Let's allow max 10 steps per NM run to keep it fast and exploratory.
        # Cost per run ~ 60 evals.
        # With budget 200, we can do ~3 runs.
        # 3 runs might not cover the space well enough for hypervolume.
        
        # Better Strategy: 
        # Use a "Simplified" NM or a Gradient-free local search that is cheaper.
        # Or, use the budget to do many cheap local refinements.
        
        # Let's try: 
        # 1. LHS for 50% of budget (100 points).
        # 2. For the remaining 100 points, perform local search around the best non-dominated points found so far?
        # This is complex to implement correctly within the black-box constraint.
        
        # Let's go with a simpler, more effective multi-start:
        # 1. Generate 10 initial points (LHS).
        # 2. For each point, run a limited Nelder-Mead optimization.
        # 3. If budget runs out, stop.
        
        # Actually, Nelder-Mead on 5D can be tricky. 
        # Let's use a robust "Powell's Method" or just "Nelder-Mead" with small initial simplex.
        
        # Let's refine the plan:
        # 1. Create 5 initial points (LHS).
        # 2. For each, run Nelder-Mead.
        # 3. NM Step count limit: 5 steps.
        # 4. Cost: 5 starts * 5 steps * 6 evals/step = 150 evals.
        # 5. Remaining 50 evals: Random sampling to fill gaps?
        
        # This seems reasonable.
        
        n_starts = 5
        max_nm_steps = 5
        
        # Generate initial points
        rng = np.random
        X_starts = np.zeros((n_starts, dim))
        for d in range(dim):
            indices = np.arange(n_starts)
            offsets = rng.random(n_starts)
            u = (indices + offsets) / n_starts
            perm = rng.permutation(n_starts)
            X_starts[:, d] = u[perm]
            X_starts[:, d] = lo[d] + X_starts[:, d] * (hi[d] - lo[d])
        
        # Helper to check if point is within bounds (problem clips, but we should stay close)
        def in_bounds(x):
            return np.all(x >= lo) and np.all(x <= hi)
            
        # Nelder-Mead implementation for multi-objective?
        # Standard NM minimizes a scalar. For MO, we can minimize the sum of objectives 
        # or a hypervolume contribution metric. 
        # Minimizing the sum is a common heuristic that drives towards the "knee" or center of the front.
        # However, it might collapse diversity.
        # To maintain diversity, we can run NM from different starts and hope they find different local optima.
        
        # Let's define a scalar objective for NM: Sum of objectives.
        # Note: RE34 objectives might have different scales. 
        # The reference point is [1705.0, 12.0, 0.3]. 
        # This suggests Obj 1 is ~1000s, Obj 2 ~10s, Obj 3 ~1.
        # Summing them directly will be dominated by Obj 1.
        # We should normalize or weight them.
        # Since we don't know the exact scale of the *optimal* values, just the reference,
        # we can use the reference point to normalize: (f_i - ref_i) / ref_i ?
        # Or just use the raw sum if scales are similar? 
        # Ref: 1705, 12, 0.3. 
        # If we minimize sum, we minimize f1 + f2 + f3.
        # A point (1000, 10, 0.2) has sum 1010.2.
        # A point (1700, 1, 0.1) has sum 1701.1.
        # The first is better in HV terms potentially, but worse in sum?
        # Actually, HV depends on the volume.
        # A weighted sum is a valid scalarization for finding supported Pareto points.
        # To get unsupported points, we need more.
        # But with only 5 starts, we are limited.
        
        # Let's try minimizing the *Hypervolume Contribution* of the new point with respect to the current archive?
        # That requires maintaining an archive and calculating HV incrementally.
        # That is computationally expensive inside the loop.
        
        # Simpler: Minimize the sum of normalized objectives.
        # Normalization: (f_i - min_i) / (max_i - min_i) ? We don't know min/max.
        # Let's just use the sum of objectives, assuming the problem is reasonably scaled or the "knee" is found.
        # Given the reference point, the objectives are likely to be minimized.
        
        def scalar_objective(x):
            f = np.array(problem(x))
            # Clip to reference point? No, problem clips x, but returns f.
            # If f is worse than ref, it doesn't count for HV.
            # For NM, we just need a gradient direction.
            # Let's use the sum of f_i.
            return np.sum(f)
        
        # We will run NM on the scalar sum.
        
        # Initialize NM simplex
        def nelder_mead(x0, max_steps, budget_counter):
            # x0: initial point
            # Returns best point found
            
            # Create initial simplex
            # x0 is center
            # Perturb each dimension
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(dim):
                simplex[i+1] = x0.copy()
                # Perturbation size: 10% of range
                step_size = 0.1 * (hi[i] - lo[i])
                simplex[i+1][i] += step_size
            
            # Evaluate initial simplex
            f_values = np.zeros(dim + 1)
            for i in range(dim + 1):
                if budget_counter['count'] >= budget:
                    break
                f_values[i] = scalar_objective(simplex[i])
                budget_counter['count'] += 1
            
            # If we ran out of budget during init, return best found
            if budget_counter['count'] >= budget:
                idx_min = np.argmin(f_values)
                return simplex[idx_min], f_values[idx_min]
                
            for step in range(max_steps):
                if budget_counter['count'] >= budget:
                    break
                
                # Sort simplex by objective value
                order = np.argsort(f_values)
                simplex = simplex[order]
                f_values = f_values[order]
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                
                # Clip to bounds
                x_ref = np.clip(x_ref, lo, hi)
                
                f_ref = scalar_objective(x_ref)
                budget_counter['count'] += 1
                
                if f_ref < f_values[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, lo, hi)
                    f_exp = scalar_objective(x_exp)
                    budget_counter['count'] += 1
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                elif f_ref < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                else:
                    # Contraction
                    beta = 0.5
                    x_con = centroid + beta * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    f_con = scalar_objective(x_con)
                    budget_counter['count'] += 1
                    
                    if f_con < f_values[-1]:
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if budget_counter['count'] >= budget:
                                break
                            f_values[i] = scalar_objective(simplex[i])
                            budget_counter['count'] += 1
                            
                        if budget_counter['count'] >= budget:
                            break
                        
            # Return best point
            idx_min = np.argmin(f_values)
            return simplex[idx_min], f_values[idx_min]
            
        budget_counter = {'count': 0}
        
        # Run multi-start
        for i in range(n_starts):
            if budget_counter['count'] >= budget:
                break
            x_best, f_best = nelder_mead(X_starts[i], max_nm_steps, budget_counter)
            # The best point from NM is already evaluated in the process (as part of simplex)
            # But we might want to ensure the *final* best is explicitly recorded?
            # The problem() calls inside NM are recorded by the evaluator.
            # So we don't need to call it again.
            
        # If we have budget left, do some random sampling to ensure coverage
        # This helps with hypervolume if NM collapses to one region.
        while budget_counter['count'] < budget:
            if time.time() > deadline:
                break
            x_rand = np.random.uniform(lo, hi)
            problem(x_rand)
            budget_counter['count'] += 1

        return None

# EVOLVE-BLOCK-END
