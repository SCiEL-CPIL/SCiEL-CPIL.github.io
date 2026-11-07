"""Rewritten optimizer for BBOB f17 (Schaffers F7) with multistart Nelder-Mead + CMA-ES polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multistart Nelder-Mead with optional CMA-ES polish pass."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s wall-clock

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # We will use a fraction of budget for multistart Nelder-Mead,
        # and the rest for a final CMA-ES polish on the best point.
        # Based on findings: multistart NM improved score significantly.
        # Let's allocate ~60% to multistart NM and ~40% to CMA-ES polish.

        # Track best point and value
        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            if calls_used >= budget:
                return best_f
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Multistart Nelder-Mead phase
        # Number of restarts: use budget / ~30 calls per restart
        # Each NM iteration uses dim+1 function evaluations per iteration,
        # and typically converges in ~100-200 iterations for dim=5.
        # Let's be more careful: allocate calls.
        
        nm_budget = int(budget * 0.65)  # 65% for multistart NM
        cma_budget = budget - nm_budget  # 35% for CMA-ES polish

        # Number of restarts: each restart uses up to ~100 calls
        n_restarts = max(1, nm_budget // 120)
        
        for i in range(n_restarts):
            if calls_used >= nm_budget or time.time() > deadline:
                break
            
            # Random initial point
            x0 = rng.uniform(-5.0, 5.0, dim)
            f0 = safe_func(x0)
            
            # Nelder-Mead
            try:
                res = minimize(
                    safe_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # CMA-ES polish phase on the best point
        if cma_budget > 20 and best_x is not None and time.time() < deadline:
            try:
                # Use a small CMA-ES with the best point as initial mean
                # and a small sigma
                sigma0 = 0.5
                max_iter = cma_budget // 2  # each CMA-ES iteration uses dim+1 evals
                
                # Implement a simple CMA-ES
                def cma_es(func, x0, sigma0, max_iter, dim, budget_left):
                    x = x0.copy()
                    sigma = sigma0
                    C = np.eye(dim)
                    C_inv = np.eye(dim)
                    Cs = np.eye(dim)
                    
                    # Parameters
                    mu = max(1, dim // 2)
                    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    w /= np.sum(w)
                    mu_eff = 1.0 / np.sum(w**2)
                    
                    c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
                    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
                    c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
                    c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
                    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
                    
                    chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
                    
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    lambda_ = max(4, int((4 + int(np.log(dim))) * mu))
                    
                    iter_count = 0
                    best_x_local = x0.copy()
                    best_f_local = func(x0)
                    calls_local = 1
                    
                    while calls_local < budget_left and iter_count < max_iter:
                        # Generate lambda_ points
                        ys = []
                        xs = []
                        fs = []
                        
                        for j in range(lambda_):
                            if calls_local >= budget_left:
                                break
                            # Sample from N(0, C)
                            z = rng.standard_normal(dim)
                            # Cholesky of C
                            try:
                                L = np.linalg.cholesky(C)
                            except np.linalg.LinAlgError:
                                L = np.eye(dim)
                            y = x + sigma * (L @ z)
                            y = np.clip(y, -5.0, 5.0)
                            fy = func(y)
                            calls_local += 1
                            ys.append(z)
                            xs.append(y)
                            fs.append(fy)
                        
                        if len(xs) == 0:
                            break
                        
                        # Sort by fitness
                        order = np.argsort(fs)
                        xs_sorted = [xs[i] for i in order]
                        fs_sorted = [fs[i] for i in order]
                        ys_sorted = [ys[i] for i in order]
                        
                        # Update best
                        if fs_sorted[0] < best_f_local:
                            best_f_local = fs_sorted[0]
                            best_x_local = xs_sorted[0].copy()
                        
                        # Update mean
                        x_new = np.sum(np.array([w[i] * xs_sorted[i] for i in range(mu)]), axis=0)
                        
                        # Update covariance
                        y_w = np.array([(xs_sorted[i] - x) / sigma for i in range(mu)])
                        delta_mean = (x_new - x) / sigma
                        C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(delta_mean, delta_mean) + (pc @ pc.T - c_c * (2 - c_c) * C)) + c_mu * np.sum([w[i] * np.outer(y_w[i], y_w[i]) for i in range(mu)], axis=0)
                        
                        # Ensure symmetry and positive semi-definite
                        C = (C + C.T) / 2
                        try:
                            eigvals, eigvecs = np.linalg.eigh(C)
                            eigvals = np.maximum(eigvals, 1e-10)
                            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                        except:
                            C = (C + C.T) / 2
                        
                        # Update path
                        pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * delta_mean
                        ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (np.linalg.norm(C @ x) / np.linalg.norm(x + 1e-10) - 1) * delta_mean
                        
                        # Update sigma
                        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                        
                        x = x_new
                        iter_count += 1
                    
                    return best_x_local, best_f_local, calls_local

                cma_x, cma_f, cma_calls = cma_es(safe_func, best_x, sigma0, max_iter, dim, cma_budget)
                if cma_f < best_f:
                    best_f = cma_f
                    best_x = cma_x
            except Exception:
                pass

        # Ensure we return the best found
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
