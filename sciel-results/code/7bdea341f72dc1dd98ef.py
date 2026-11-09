import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                raise TimeoutError()
            # Ensure x is a proper numpy array
            x = np.asarray(x, dtype=float)
            # Clip to box [-5, 5]
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if not np.isfinite(fx):
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Sampling
        # Use a deterministic low-discrepancy-like or just random sampling to find a good basin.
        # Rosenbrock has a narrow valley. Random sampling in [-4, 4] is reasonable.
        # Let's use 20-30 points for global search.
        rng = np.random.default_rng(42)
        n_global = 30
        
        # Include center and some structured points
        candidates = [np.zeros(dim)]
        # Add corners of the box? No, too far.
        # Add points near 1? The unconstrained minimum is at 1.
        # The shifted minimum is in [-4, 4].
        # Let's sample uniformly in [-4, 4].
        for _ in range(n_global - 1):
            candidates.append(rng.uniform(-4.0, 4.0, dim))

        # Evaluate global candidates
        global_results = []
        for p in candidates:
            if calls >= budget or time.time() > deadline:
                break
            f = safe_func(p)
            global_results.append((f, p))

        if not global_results:
            return best_x, best_f

        # Sort by function value
        global_results.sort(key=lambda x: x[0])

        # Phase 2: Local Refinement using L-BFGS-B
        # Try the top few global candidates.
        # L-BFGS-B is efficient for Rosenbrock.
        # We have budget - calls remaining.
        # Each L-BFGS-B run might take ~50-100 calls.
        # We can afford 3-5 runs.
        
        top_k = min(5, len(global_results))
        
        for i in range(top_k):
            if calls >= budget - 10 or time.time() > deadline:
                break
            
            f_init, x_init = global_results[i]
            
            # If we've already found a very good value, maybe stop?
            # But let's try to refine the best ones.
            
            # Use L-BFGS-B
            # maxiter should be limited to not consume all budget.
            # remaining_budget = budget - calls
            # Let's allow up to 100 iterations per run, but it will stop early if converged.
            # ftol and gtol should be tight to ensure convergence.
            
            try:
                res = minimize(
                    safe_func,
                    x_init,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-12,
                        'gtol': 1e-6,
                        'maxls': 20
                    }
                )
                # If successful, we might have found the global min.
                # If res.success is True, we can break, as further runs are unlikely to improve significantly
                # unless the first one got stuck in a local min (Rosenbrock has only one global min, but local mins exist? 
                # Actually Rosenbrock is multimodal in higher dims? In 5D it has local minima.
                # So trying multiple starts is good.)
                if res.success:
                    # Check if we improved best_f significantly
                    # If we are close to 0, break?
                    # For now, just continue to next top candidate if budget allows, 
                    # but if success, it's likely a local min. 
                    # Let's try the next one to see if we get a better local min.
                    pass
            except Exception:
                pass

        # Phase 3: Final Polish with Nelder-Mead if budget remains?
        # L-BFGS-B usually does a good job. NM might help if L-BFGS-B stopped early.
        # But L-BFGS-B is gradient-based (approximate) and NM is derivative-free.
        # Given the budget is 500, and L-BFGS-B is efficient, we might have plenty of budget left.
        # Let's add a final NM pass on the best_x found so far.
        
        remaining = budget - calls
        if remaining > 20:
            # Initialize NM simplex around best_x
            x0 = best_x
            sigma = 0.1 # Smaller step size for polishing
            
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                if xi[i] + sigma > 5:
                    xi[i] -= 2 * sigma
                elif xi[i] - sigma < -5:
                    xi[i] += 2 * sigma
                else:
                    xi[i] += sigma
                simplex.append(xi)
            
            simplex_f = []
            for i in range(dim + 1):
                if calls >= budget - 5 or time.time() > deadline:
                    break
                simplex_f.append(safe_func(simplex[i]))
            
            if len(simplex_f) == dim + 1:
                simplex_data = list(zip(simplex_f, simplex))
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_shrink = 0.5
                
                # Run NM for a few iterations or until budget runs out
                while calls < budget - 5 and time.time() < deadline:
                    simplex_data.sort(key=lambda item: item[0])
                    f_worst = simplex_data[-1][0]
                    x_worst = simplex_data[-1][1]
                    f_best = simplex_data[0][0]
                    x_best = simplex_data[0][1]
                    
                    if f_best < best_f:
                        best_f = f_best
                        best_x = x_best.copy()
                    
                    f_spread = f_worst - f_best
                    if f_spread < 1e-8:
                        break
                        
                    centroid = np.mean([item[1] for item in simplex_data[:-1]], axis=0)
                    
                    x_r = centroid + alpha * (centroid - x_worst)
                    x_r = np.clip(x_r, -5, 5)
                    f_r = safe_func(x_r)
                    
                    if f_r < f_best:
                        x_e = centroid + gamma * (centroid - x_worst)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = safe_func(x_e)
                        if f_e < f_r:
                            simplex_data[-1] = (f_e, x_e)
                        else:
                            simplex_data[-1] = (f_r, x_r)
                    elif f_r < simplex_data[-2][0]:
                        simplex_data[-1] = (f_r, x_r)
                    else:
                        if f_r < f_worst:
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            x_c = centroid - rho * (centroid - x_worst)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_func(x_c)
                        if f_c <= f_r:
                            simplex_data[-1] = (f_c, x_c)
                        else:
                            for i in range(1, len(simplex_data)):
                                x_s = x_best + sigma_shrink * (simplex_data[i][1] - x_best)
                                x_s = np.clip(x_s, -5, 5)
                                f_s = safe_func(x_s)
                                simplex_data[i] = (f_s, x_s)

        return best_x, best_f
