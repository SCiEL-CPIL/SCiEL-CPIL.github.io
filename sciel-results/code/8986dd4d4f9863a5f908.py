import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # Global state
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            if time.time() > deadline:
                raise TimeoutError
            if eval_count >= budget:
                raise BudgetExceeded
            f_val = func(x)
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Sampling
        # Use a mix of uniform and LHS-like sampling for better coverage
        rng = np.random.default_rng(42)
        n_samples = 20
        for i in range(n_samples):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            safe_func(x)
        
        if best_x is None:
            return np.zeros(n), 0.0

        # Phase 2: CMA-ES with adaptive restarts
        # We will run CMA-ES. If it stagnates (sigma becomes too small or no improvement for a while),
        # we can try a restart from the best point with a larger sigma or a new random point.
        # Given the budget is only 500, we should be careful with restarts.
        
        # Let's implement a robust CMA-ES loop.
        # Parameters for dim=5
        lam = 4 * n  # 20
        mu = lam // 2  # 10
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        chi_n = np.sqrt(n) * (1 - 1/(4*n) - 1/(20*n**2))
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        ds = 1 + 2 * max(0, (mu_eff - 2) / (n + mu_eff)) + cs
        cmu = 2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff)
        
        def run_cmaes(x_start, sigma_start):
            nonlocal best_x, best_f, eval_count
            
            x_mean = x_start.copy()
            sigma = sigma_start
            C = np.eye(n)
            Cs = np.zeros(n)
            ps = np.zeros(n)
            
            B, D = self._cov_to_b_d(C, n)
            
            # Track stagnation
            best_f_local = best_f
            stagnation_counter = 0
            max_stagnation = 30  # Generations without improvement before considering restart
            
            while eval_count < budget:
                if time.time() > deadline:
                    break
                
                # Generate lambda samples
                ys = []
                xs = []
                fs = []
                
                for k in range(lam):
                    if eval_count >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = B @ (D * z)
                    x = np.clip(x_mean + sigma * y, -5, 5)
                    ys.append(y)
                    xs.append(x)
                    fs.append(safe_func(x))
                
                if eval_count >= budget:
                    break
                
                if len(fs) < mu:
                    break
                
                # Sort by fitness
                idx = np.argsort(fs)
                w_sorted = w[:mu]
                
                # Calculate new mean
                new_mean = np.average([xs[i] for i in idx[:mu]], axis=0, weights=w_sorted)
                
                # Update ps
                # Need C^(-1/2)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = (eigvecs * (1/np.sqrt(eigvals))) @ eigvecs.T
                
                ps = (1 - cs) * ps + cs * np.sqrt(mu_eff) * (C_inv_sqrt @ (new_mean - x_mean) / sigma)
                
                # Check hsig
                norm_ps = np.linalg.norm(ps)
                hsig = 1.0 if (norm_ps / chi_n - (1 - 0.5 * cs)) / (1 + cs) > 1.4 else 0.0
                
                # Update sigma
                sigma = sigma * np.exp((1.0 / ds) * (norm_ps / chi_n - 1))
                
                # Update Cs
                Cs = (1 - cs) * Cs + hsig * cs * ps
                
                # Update C
                y_sel = np.array([ys[idx[i]] for i in range(mu)])
                y_bar = np.average(y_sel, axis=0, weights=w_sorted)
                
                decay = 1 - cs * cs * (1 - ds) - cmu
                term_cs = cs * cs * (1 - ds) * np.outer(Cs, Cs)
                term_cmu = cmu * np.average([np.outer(y, y) for y in (y_sel - y_bar)], axis=0, weights=w_sorted)
                
                C = decay * C + term_cs + term_cmu
                C = (C + C.T) / 2
                
                # Update mean
                x_mean = new_mean
                
                # Recompute B, D
                B, D = self._cov_to_b_d(C, n)
                
                # Check for improvement
                if best_f < best_f_local - 1e-10:
                    best_f_local = best_f
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                # If stagnated and sigma is very small, break to allow restart logic outside
                if stagnation_counter > max_stagnation and sigma < 1e-4:
                    break

        # Initial CMA-ES run
        run_cmaes(best_x, 2.0)
        
        # If we still have budget and haven't found a very good solution, try a restart
        # Check if we are close to the boundary or if improvement has stalled
        # A simple heuristic: if best_f is still high (e.g., > 50) and we have > 50 calls left, restart
        if eval_count < budget - 50 and best_f > 50:
            # Try restarting from a new random point near the boundary or center
            # Or just a new random point
            x_new = rng.uniform(-4, 4, n)
            run_cmaes(x_new, 1.0)
            
        return best_x, best_f

    def _cov_to_b_d(self, C, n):
        # C = B D^2 B^T
        # Use Cholesky: C = L L^T
        # So B = L, D = 1
        # Or Eigen: C = V D V^T, B=V, D=sqrt(eigvals)
        # Cholesky is faster.
        try:
            L = np.linalg.cholesky(C)
            return L, np.ones(n)
        except np.linalg.LinAlgError:
            # Fallback to eigen if not PD
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            return eigvecs, np.sqrt(eigvals)
