import numpy as np
import time

class Optimizer:
    """
    CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        calls = 0
        best_x = None
        best_f = np.inf

        def eval_pt(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol quasi-random global exploration
        from scipy.stats import qmc
        n_init = min(10, budget)
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        init_points = sampler.random(n=n_init)
        init_points = 2.0 * init_points - 1.0
        init_points = 5.0 * init_points
        
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            eval_pt(init_points[i])
            
        if best_x is None:
            best_x = np.zeros(dim)
            eval_pt(best_x)

        # Phase 2: CMA-ES with corrected covariance update
        lam = 4 + int(3 * np.log(dim))
        if lam < 8: lam = 8
        if lam > 15: lam = 15
        
        sigma = 2.0
        
        mean = best_x.copy()
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        c_sigma = (mu_eff/2 + 1) / (dim + 2 + mu_eff/2)
        c_c = (4 + mu_eff/dim) / (dim + 4 + 2*mu_eff/dim)
        c_1 = 2 / ((dim + 1.3)**2 + lam)
        c_mu = min(1 - c_1, 2 * ((mu_eff/2 - 2 + 1/mu_eff) / ((dim + 2) + mu_eff/2)))
        
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        max_iter = (budget - calls) // lam + 2
        
        for it in range(max_iter):
            if calls >= budget or time.time() > deadline:
                break
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1/np.sqrt(eigvals)) @ eigvecs.T
            
            z = np.random.randn(lam, dim)
            x_samples = mean[None, :] + sigma * (z @ C_sqrt.T)
            
            fvals = []
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                f = eval_pt(x_samples[i])
                fvals.append(f)
            
            if len(fvals) < lam:
                break
                
            order = np.argsort(fvals)
            x_samples = x_samples[order]
            fvals = np.array(fvals)[order]
            
            mean_new = np.sum(weights[:, None] * x_samples[:mu], axis=0)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (mean_new - mean) / sigma)
            
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2*(it+1))) < (1.4 + 2/(dim+1)) * chi_n else 0
            
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (mean_new - mean))
            
            h_one = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(2*(it+1))) < (1.2 + 2/(dim+1)) * chi_n else 0
            
            y = (mean_new - mean) / sigma
            
            # Corrected covariance update: rank-1 + rank-mu
            C_new = (1 - c_1 - c_mu) * C + c_1 * np.outer(y, y)
            for i in range(mu):
                y_i = (x_samples[i] - mean) / sigma
                C_new += c_mu * weights[i] * np.outer(y_i, y_i)
            
            C = (C_new + C_new.T) / 2
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            mean = mean_new
            
            if sigma < 1e-10:
                break
                
        # Phase 3: Final polish with Nelder-Mead if budget remains
        if calls < budget and best_x is not None and time.time() < deadline:
            from scipy.optimize import minimize
            # Use Nelder-Mead for local refinement
            x0 = best_x.copy()
            options = {'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
            try:
                res = minimize(func, x0, method='Nelder-Mead', options=options)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass
        
        return best_x, best_f
