import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Phase 1: Random Search
        n_random = 20
        for i in range(n_random):
            if time.time() > deadline - 5 or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        # CMA-ES Parameters
        lam = 5
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        w = np.concatenate([w, np.zeros(lam - mu)])
        mueff = w.sum() ** 2 / (w ** 2).sum()
        
        c_sigma = (mueff + 2) / ((dim + 1) ** 1.5 + mueff)
        c_c = (4 + dim / c_sigma) / ((dim + 2) ** 1.5 + mueff)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x_mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        evals = 0
        no_improve_count = 0

        while calls_made < budget:
            if time.time() > deadline - 2:
                break
            
            remaining = budget - calls_made
            if remaining < lam:
                # Switch to 1+1 ES
                while calls_made < budget and time.time() < deadline - 2:
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (C @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        x_mean = y.copy()
                    else:
                        sigma *= 0.8
                break

            # Generate offspring
            Y = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C @ z)
                y = np.clip(y, -5, 5)
                Y.append(y)

            # Evaluate
            F = []
            for y in Y:
                if time.time() > deadline - 2:
                    break
                fy = func(y)
                F.append(fy)
                calls_made += 1

            if len(F) < lam:
                F.extend([best_f] * (lam - len(F)))

            idx = np.argsort(F)
            Y = np.array(Y)[idx]
            F = np.array(F)[idx]

            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            evals += lam

            # Update mean
            x_new = np.sum(w[:mu, None] * Y[:mu], axis=0)

            # Update paths
            diff = (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * diff
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (C @ diff)

            # Update Covariance
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * evals))) / chi_N < 1.4 + 2 / (dim + 1) else 0.0
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)

            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff_i = (Y[i] - x_mean) / sigma
                rank_mu += w[i] * np.outer(diff_i, diff_i)

            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_h * C) + c_mu * rank_mu

            # Update Sigma
            sigma *= np.exp(c_sigma / chi_N * (np.linalg.norm(p_sigma) / chi_N - 1))

            # Update Mean
            x_mean = x_new

            # Numerical Stability
            if evals % 50 == 0:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, 1e10)
                    C = (eigvecs * eigvals) @ eigvecs.T
                except:
                    pass

            # Convergence / Restart
            if sigma < 1e-10:
                break
            
            # Restart if no improvement for 15 generations
            if no_improve_count > 15:
                # Restart with a new random point
                if calls_made + 1 < budget:
                    x = rng.uniform(-5, 5, dim)
                    fx = func(x)
                    calls_made += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    
                    x_mean = best_x.copy()
                    sigma = 1.0
                    C = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    evals = 0
                    no_improve_count = 0

        return best_x, best_f
