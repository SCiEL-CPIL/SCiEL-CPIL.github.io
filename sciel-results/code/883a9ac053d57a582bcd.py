import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        total_budget = self.budget
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0
        
        # We need to track the best point found by CMA-ES and potentially by the polish
        # CMA-ES will do the bulk of the work.
        
        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget or time.time() > deadline:
                return best_f
            calls_made += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # CMA-ES Parameters
        # Population size: lambda = 4 + floor(3 * log(n)) is standard, but for multimodal, larger is better.
        # Let's use lambda = 12 for n=5. 
        # Initial sigma: 2.0 (covers a good portion of the [-5,5] box from origin)
        
        # We will implement a simple CMA-ES loop or use scipy's if available and robust.
        # SciPy's CMA-ES is robust. However, we need to strictly control the budget.
        # SciPy's minimize with method='CMA-ES' has a 'maxiter' option but it's in generations, not evals.
        # It's safer to implement a tight loop or carefully wrap it.
        
        # Let's use a manual CMA-ES implementation to have full control over the budget and time, 
        # or use scipy but with a tight wrapper.
        # Given the constraint of "Python standard library, NumPy and SciPy", and the need for efficiency,
        # I will implement a lightweight CMA-ES. It's ~50 lines and very effective.
        
        # CMA-ES State
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n)
        D = 1.0
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Hyperparameters
        cc = (4 + n / (4 + n)) / ((n + 4) / 2 + 4 + n / (4 + n))
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        cmu = 2 / ((n + 2) ** 2 + mu_eff)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Population size
        lambda_ = 12
        # Number of parents
        mu = lambda_ // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w ** 2)
        
        # Recalculate hyperparameters with correct mu_eff
        cc = (4 + n / (4 + n)) / ((n + 4) / 2 + 4 + n / (4 + n))
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        cmu = 2 / ((n + 2) ** 2 + mu_eff)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        def sample_normal():
            return np.random.normal(0, 1, n)
            
        def update_B_D():
            nonlocal B, D, C
            # Eigendecomposition of C
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive
            eigvals = np.maximum(eigvals, 1e-12)
            B = eigvecs
            D = np.sqrt(eigvals)
            
        update_B_D()
        
        # Main Loop
        # Each generation uses lambda_ evaluations.
        # We can afford roughly total_budget / lambda_ generations.
        
        while calls_made + lambda_ <= total_budget and time.time() < deadline:
            # Sample offspring
            ys = []
            fs = []
            
            for i in range(lambda_):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                z = sample_normal()
                y = x_mean + sigma * (B @ (D * z))
                f_y = evaluate(y)
                ys.append(y)
                fs.append(f_y)
                
            if len(ys) < mu:
                break
                
            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = np.array(ys)[:mu]
            fs_sorted = np.array(fs)[:mu]
            
            # Update mean
            x_mean_new = np.sum(w[:, np.newaxis] * ys_sorted, axis=0)
            
            # Update covariance matrix
            yc = (ys_sorted - x_mean) / sigma
            C_new = C + cmu / sigma**2 * (np.sum(w[:, np.newaxis] * np.outer(yc.T, yc.T), axis=0) - (1 - cs * (2 - cs)) * np.outer(pc, pc))
            
            # Update evolution paths
            pc = (1 - cc) * pc + cc * (1 / np.sqrt(cc * (2 - cc))) * (x_mean_new - x_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs)) * B.T @ ((x_mean_new - x_mean) / sigma)
            
            # Damping of sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update state
            x_mean = x_mean_new
            C = C_new
            update_B_D()
            
            # Check for stagnation or time
            if sigma < 1e-8:
                break
                
        # Phase 2: Polish with L-BFGS-B
        # L-BFGS-B is good for final refinement in a box.
        if calls_made < total_budget and time.time() < deadline and best_x is not None:
            remaining_budget = total_budget - calls_made
            if remaining_budget > 10:
                try:
                    res = minimize(lambda x: evaluate(x), best_x, method='L-BFGS-B',
                                   bounds=[(-5, 5)] * n,
                                   options={'maxiter': remaining_budget // 2, 'ftol': 1e-12, 'gtol': 1e-8})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass
        
        # Phase 3: Final random checks if time permits (unlikely to help much)
        # Just return best found
        return best_x, best_f
