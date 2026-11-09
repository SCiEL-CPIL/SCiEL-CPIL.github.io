import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280

        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = max(1, int(lambda_ / 2))
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        calls = 0
        best_x = x.copy()
        best_f = func(x)
        calls += 1
        
        rng = np.random.default_rng(42)
        
        while calls + lambda_ <= budget:
            if time.time() > deadline:
                break
                
            # Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
                sigma *= 0.5
                
            z = rng.standard_normal((lambda_, dim))
            y = x[None, :] + sigma * (z @ L.T)
            y = np.clip(y, -5, 5)
            
            fy = np.array([func(yp) for yp in y])
            calls += lambda_
            
            idx = np.argsort(fy)
            y_sel = y[idx[:mu]]
            fy_sel = fy[idx[:mu]]
            
            if fy_sel[0] < best_f:
                best_f = fy_sel[0]
                best_x = y_sel[0].copy()
                
            y_mean = np.sum(w[:, None] * y_sel, axis=0)
            
            # Update paths
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * np.linalg.solve(L, y_mean - x)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.linalg.solve(L, L @ np.linalg.solve(L, y_mean - x)) / sigma
            # Note: np.linalg.solve(L, L @ solve(L, v)) is equivalent to C @ v? No.
            # We need the standardized step.
            # Let v = y_mean - x.
            # Standardized v_std = L^-1 v.
            # ps update: ps = (1-cs)ps + sqrt(cs(2-cs)mu_eff) * L^-1 v / sigma
            # So we need L^-1 v.
            
            # Let's recompute ps correctly.
            # v = y_mean - x
            # v_std = np.linalg.solve(L, v)
            # ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v_std / sigma
            
            # Re-doing the update logic inside the loop for clarity
            
            v = y_mean - x
            v_std = np.linalg.solve(L, v)
            
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * v_std
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v_std / sigma
            
            # Sigma update
            chi_N = np.sqrt(dim) * (1 - 1.0/(4*dim) - 1.0/(25*dim*dim))
            sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / chi_N - 1))
            
            # Covariance update
            # Rank 1
            rank1 = np.outer(v_std, v_std)
            # Rank mu
            y_sel_std = np.linalg.solve(L, y_sel - x[None, :]) / sigma
            rankmu = np.sum(w[:, None, None] * np.einsum('i,ij,ik->ijk', w, y_sel_std, y_sel_std), axis=0)
            
            C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * rank1 + c_mu * rankmu
            C = (C + C.T) / 2.0
            C += 1e-9 * np.eye(dim)
            
            x = y_mean
            
            if sigma < 1e-8:
                break

        # Polish with a few local steps if budget remains
        # Simple coordinate descent or Nelder-Mead polish
        # For Sphere, a simple local search is effective.
        # Let's do a few steps of (1+1)-ES with small sigma for polish.
        sigma_polish = max(sigma, 1e-4)
        while calls < budget:
            if time.time() > deadline:
                break
            y = np.clip(best_x + sigma_polish * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < best_f:
                best_x = y
                best_f = fy
                sigma_polish *= 1.2
            else:
                sigma_polish *= 0.8
                if sigma_polish < 1e-6:
                    break

        return best_x, best_f
