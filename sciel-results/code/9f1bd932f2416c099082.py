"""Direct construction approach for multi-objective optimization on ZDT4."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for ZDT4.
    Strategy: 
    1. ZDT4's f1 and f2 depend only on x_2..x_n. x_1 is irrelevant.
    2. The ideal point is f1=0, f2=0.
    3. f2 is minimized when f1 is close to 0.
    4. We use a single-objective search to minimize f1^2 using the variables x_2..x_n.
    5. We use multiple random restarts with local refinement (Nelder-Mead style) to handle multimodality.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # x_1 is irrelevant for ZDT4 objectives, fix it to 0.5
        # We only search x_2..x_dim
        dim_search = self.dim - 1
        if dim_search <= 0:
            # Fallback if dim is 1
            x = np.random.uniform(lo, hi, self.dim)
            return problem(x)

        lo_search = lo[1:]
        hi_search = hi[1:]
        
        # Objective function for single-objective search
        # We want to minimize f1^2 to drive f1 to 0
        def objective(x_search):
            # Check deadline
            if time.time() > deadline:
                return float('inf')
            
            x = np.empty(self.dim)
            x[0] = 0.5
            x[1:] = x_search
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                # f[0] is f1, f[1] is f2
                # We minimize f1^2
                return f[0] ** 2
            except:
                return float('inf')

        best_x_search = None
        best_obj = float('inf')
        
        # Use a portion of the budget for random restarts + local search
        # Nelder-Mead typically needs ~10 * dim evaluations for convergence
        # dim_search is 9. So ~90 evaluations per run.
        # We have 200 evaluations. We can do 2 full runs or 1 run + random sampling.
        
        # Strategy: 
        # 1. Do 2 Nelder-Mead runs with different initializations.
        # 2. Use remaining budget for random sampling to see if we can beat it.
        
        n_restarts = 2
        evals_per_restart = self.budget // (n_restarts + 1) # Reserve some for random
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            x_init = np.random.uniform(lo_search, hi_search, dim_search)
            
            # Simple Nelder-Mead implementation
            # Initialize simplex
            simplex = [x_init.copy()]
            for i in range(dim_search):
                perturbation = x_init.copy()
                perturbation[i] += 0.1 * (hi_search[i] - lo_search[i])
                simplex.append(perturbation)
            
            simplex = np.array(simplex)
            f_values = np.array([objective(x) for x in simplex])
            
            # Nelder-Mead loop
            max_iter = 50
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Sort simplex by objective value
                idx = np.argsort(f_values)
                simplex = simplex[idx]
                f_values = f_values[idx]
                
                # Check convergence
                if np.max(f_values) - np.min(f_values) < 1e-6:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + (centroid - simplex[-1])
                f_ref = objective(x_ref)
                
                if f_ref < f_values[0]:
                    # Expansion
                    x_exp = centroid + 2 * (centroid - simplex[-1])
                    f_exp = objective(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                elif f_ref < f_values[-2]:
                    # Reflection is good
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + 0.5 * (simplex[-1] - centroid)
                    f_con = objective(x_con)
                    if f_con < f_values[-1]:
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim_search + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_values[i] = objective(simplex[i])
            
            # Update best
            best_idx = np.argmin(f_values)
            if f_values[best_idx] < best_obj:
                best_obj = f_values[best_idx]
                best_x_search = simplex[best_idx].copy()

        # Use remaining budget for random sampling
        # This helps if local search gets stuck
        while time.time() < deadline - 5:
            x_rand = np.random.uniform(lo_search, hi_search, dim_search)
            f_rand = objective(x_rand)
            if f_rand < best_obj:
                best_obj = f_rand
                best_x_search = x_rand.copy()

        # Construct final result
        if best_x_search is None:
            best_x_search = np.random.uniform(lo_search, hi_search, dim_search)
        
        x_final = np.empty(self.dim)
        x_final[0] = 0.5
        x_final[1:] = best_x_search
        x_final = np.clip(x_final, lo, hi)
        
        return problem(x_final)
# EVOLVE-BLOCK-END
