import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        lam = 10
        mu = lam // 2

        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)

        # Initial parameters
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)

        # Adaptation parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 - mueff / (dim + 4)) / (dim + 4 + mueff)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))

        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        best_x = None
        best_f = np.inf
        calls_used = 0

        chi_N = np.sqrt(dim) * (1 - 1/(4*dim) - 1/(25*dim**2))
        c_hsig = (1 + 2 / max(1, mueff - 1) + c_sigma) / (1 + c_sigma)

        def update_cholesky(C_new):
            try:
                L_new = np.linalg.cholesky(C_new)
                return L_new
            except np.linalg.LinAlgError:
                C_reg = C_new + 1e-4 * np.eye(dim)
                try:
                    L_new = np.linalg.cholesky(C_reg)
                    return L_new
                except:
                    return np.eye(dim)

        # Evaluation loop
        while calls_used + lam <= budget:
            if time.time() > deadline:
                break

            L = update_cholesky(C)

            # Generate population
            z = np.random.randn(lam, dim)
            y = z @ L.T
            x_pop = x_mean + sigma * y

            # Clip to box [-5, 5]
            x_pop = np.clip(x_pop, -5, 5)

            # Evaluate
            f_pop = np.array([func(x) for x in x_pop])
            calls_used += lam

            # Update best
            idx_best = np.argmin(f_pop)
            if f_pop[idx_best] < best_f:
                best_f = f_pop[idx_best]
                best_x = x_pop[idx_best].copy()

            # Sort
            idx_sort = np.argsort(f_pop)
            x_sorted = x_pop[idx_sort]

            # Selection
            x_w = x_sorted[:mu]
            y_w = y[idx_sort][:mu]

            # Update mean
            x_mean_old = x_mean
            x_mean = np.sum(weights[:, np.newaxis] * x_w, axis=0)

            # Update p_sigma
            d_mean = x_mean - x_mean_old

            try:
                L_inv_T = np.linalg.inv(L).T
            except:
                L_inv_T = np.eye(dim)

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (L_inv_T @ (d_mean / sigma))

            # h_sigma
            gen = calls_used / lam
            if gen > 0:
                norm_ps = np.linalg.norm(p_sigma)
                denom = np.sqrt(1 - (1 - c_sigma) ** (2 / gen))
                h_sigma = 1.0 if (norm_ps / denom) < (1.4 + 2 / (dim + 1)) * c_hsig else 0.0
            else:
                h_sigma = 1.0

            # Update sigma
            sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (L_inv_T @ (d_mean / sigma))

            # Update C
            C = (1 - c_1 - c_mu) * C \
                + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) \
                + c_mu * np.sum(weights[:, np.newaxis]**2 * (y_w - np.mean(y_w, axis=0))[:, np.newaxis] * (y_w - np.mean(y_w, axis=0)), axis=0)

            # Ensure symmetry
            C = (C + C.T) / 2

        # Final high-precision polish pass using Nelder-Mead
        if best_x is not None and calls_used < budget and time.time() < deadline:
            def local_func(x):
                nonlocal calls_used
                if calls_used >= budget or time.time() > deadline:
                    return np.inf
                val = func(x)
                calls_used += 1
                return val

            remaining_budget = budget - calls_used
            try:
                res = minimize(local_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': remaining_budget, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

            # Additional polish: coordinate-wise fine search
            if calls_used < budget and time.time() < deadline:
                polish_sigma = 0.005
                polish_deadline = deadline - 5.0
                for _ in range(20):
                    if calls_used >= budget or time.time() > polish_deadline:
                        break
                    improved = False
                    for i in range(dim):
                        if calls_used >= budget or time.time() > polish_deadline:
                            break
                        step = np.zeros(dim)
                        step[i] = polish_sigma
                        cand1 = np.clip(best_x + step, -5, 5)
                        f1 = func(cand1)
                        calls_used += 1
                        if f1 < best_f:
                            best_f = f1
                            best_x = cand1
                            improved = True
                        step[i] = -step[i]
                        cand2 = np.clip(best_x + step, -5, 5)
                        f2 = func(cand2)
                        calls_used += 1
                        if f2 < best_f:
                            best_f = f2
                            best_x = cand2
                            improved = True
                    if not improved:
                        polish_sigma *= 0.5
                        if polish_sigma < 1e-6:
                            break

        return best_x, best_f
