# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        # We will use a two-phase strategy:
        # 1. Latin Hypercube Sampling (LHS) for diverse initial population
        # 2. Local Nelder-Mead refinement on non-dominated points
        
        # Phase 1: LHS Initialization
        # Number of initial points: enough to cover the space, but leave budget for refinement
        # Let's use ~50% of budget for initialization, rest for refinement
        # But refinement is expensive (Nelder-Mead takes many evals).
        # Let's start with a modest LHS size, say 20-30 points.
        
        lhs_size = min(30, budget // 4)
        
        # Generate LHS points
        # In LHS, we divide each dimension into lhs_size equal intervals and pick one point per interval
        lhs_points = []
        for i in range(lhs_size):
            x = np.zeros(dim)
            for d in range(dim):
                # Random permutation of intervals
                # This ensures that for each dimension, we have one point in each bin
                # But standard LHS uses a single permutation for all dims? No, usually random permutation per dim.
                # To be efficient, let's just use random points but ensure some structure?
                # Actually, simple random is often good enough for small dims.
                # Let's stick to random for simplicity and robustness, but maybe stratified?
                # Let's use stratified sampling (random within each bin)
                # For simplicity and speed, let's just use random for now, 
                # but the directive asks for LHS.
                # True LHS:
                # 1. Create bins [i/n, (i+1)/n] for i in 0..n-1
                # 2. For each dim, assign a random bin to each point? No, that's not LHS.
                # LHS: 
                # - For each dim, create a permutation of 0..n-1.
                # - Point j in dim d falls in bin perm_d[j].
                # - Within the bin, pick a random offset.
                
                pass
        
        # Let's implement a simple LHS
        lhs_perms = [np.random.permutation(lhs_size) for _ in range(dim)]
        X_lhs = np.zeros((lhs_size, dim))
        for j in range(lhs_size):
            for d in range(dim):
                bin_idx = lhs_perms[d][j]
                # Random point in [bin_idx/n, (bin_idx+1)/n]
                u = (bin_idx + np.random.random()) / lhs_size
                X_lhs[j, d] = u
        
        X_init = lo + (hi - lo) * X_lhs
        X_init = np.clip(X_init, lo, hi)
        
        # Evaluate initial population
        X_list = []
        F_list = []
        used = 0
        
        for x in X_init:
            if used >= budget:
                break
            f = np.array(problem(x), dtype=float)
            X_list.append(x)
            F_list.append(f)
            used += 1
            if time.time() > deadline:
                break
        
        if not X_list:
            return np.array([])
        
        X = np.array(X_list)
        F = np.array(F_list)
        
        # Helper function to find non-dominated points
        def get_non_dominated(F):
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                le = np.all(F <= F[i], axis=1)
                lt = np.any(F < F[i], axis=1)
                le[i] = False
                if np.any(le & lt):
                    dominated[i] = True
            return np.where(~dominated)[0]
        
        # Phase 2: Local Refinement
        # Use Nelder-Mead on non-dominated points to refine them.
        # We need to minimize a scalar objective. Since we have multiple objectives,
        # we can use a weighted sum or just minimize each objective separately?
        # Better: use a hypervolume-aware selection or just refine the "best" points.
        # Let's refine the non-dominated points by minimizing a random weighted sum of objectives.
        
        # How many points to refine?
        # Nelder-Mead is expensive. Let's limit the number of refinements.
        # Each Nelder-Mead run takes ~dim * 10 evaluations.
        # With dim=12, that's ~120 evals per run.
        # Budget remaining: budget - used.
        # Max runs: (budget - used) // (dim * 10)
        
        remaining_budget = budget - used
        nm_dim = dim
        nm_cost_per_run = nm_dim * 10  # Rough estimate
        max_nm_runs = max(1, remaining_budget // nm_cost_per_run)
        
        # Select points to refine: non-dominated ones
        nd_idx = get_non_dominated(F)
        
        # If we have more non-dominated points than max_nm_runs, select a subset
        if len(nd_idx) > max_nm_runs:
            # Select randomly or by hypervolume contribution?
            # Let's just select randomly for now
            refine_idx = np.random.choice(nd_idx, size=max_nm_runs, replace=False)
        else:
            refine_idx = nd_idx
        
        # Perform Nelder-Mead refinement
        # We will use scipy.optimize.minimize with method='Nelder-Mead'
        # Objective: weighted sum of objectives
        # Weights: random positive weights
        
        from scipy.optimize import minimize
        
        for idx in refine_idx:
            if used >= budget or time.time() > deadline:
                break
            
            # Random weights for scalarization
            weights = np.random.random(n_obj)
            weights /= np.sum(weights)
            
            def objective(x):
                if used >= budget:
                    return np.inf
                # Clip x to bounds
                x_clipped = np.clip(x, lo, hi)
                f = np.array(problem(x_clipped), dtype=float)
                used += 1
                return np.dot(f, weights)
            
            # Initial guess: the point itself
            x0 = X[idx].copy()
            
            # Run Nelder-Mead
            try:
                result = minimize(objective, x0, method='Nelder-Mead', 
                                  options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-4})
                if result.success and used < budget:
                    x_new = np.clip(result.x, lo, hi)
                    f_new = np.array(problem(x_new), dtype=float)
                    used += 1
                    # Update population
                    X[idx] = x_new
                    F[idx] = f_new
            except Exception:
                pass
        
        return X
# EVOLVE-BLOCK-END
