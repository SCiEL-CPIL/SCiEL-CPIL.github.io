import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = 12
        mu = lam // 2
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        
        mueff = 1.0 / np.dot(w, w)
        
        cs = (mueff + 2) / (d + mueff + 5)
        dsigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        cc = (4 - mueff) / (d + 4 - mueff)
        c1 = 2 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        
        x_mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        ps = np.zeros(d)
        pc = np.zeros(d)
        
        chi_lam = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        best_x = x_mean.copy()
        best_f = np.inf
        calls_made = 0
        
        for gen in range(100):
            if time.time() > deadline:
                break
            
            z = np.random.randn(lam, d)
            y = z @ (D[:, None] * B.T)
            x_pop = x_mean[None, :] + sigma * y
            x_pop = np.clip(x_pop, -5, 5)
            
            f_pop = np.zeros(lam)
            for i in range(lam):
                try:
                    f_pop[i] = func(x_pop[i])
                except:
                    f_pop[i] = np.inf
            calls_made += lam
            
            if calls_made >= budget:
                break
                
            order = np.argsort(f_pop)
            x_sorted = x_pop[order]
            f_sorted = f_pop[order]
            
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
                
            x_mean_old = x_mean.copy()
            x_mean = np.dot(w, x_sorted[:mu])
            
            y_star = (x_mean - x_mean_old) / sigma
            ps = (1 - cs) * ps + np.sqrt(mueff) * cs * (1 - cs) * y_star
            
            norm_ps = np.linalg.norm(ps)
            sigma = sigma * np.exp(cs / dsigma * (norm_ps / chi_lam - 1))
            
            hs = 1.0
            if norm_ps / chi_lam < (1 + 2 / (d + 1)) * (1 - cs):
                hs = 1.0
            pc = (1 - cc) * pc + hs * np.sqrt(mueff) * cc * (1 - cc) * y_star
            
            dx_m = y_star
            dx_mu = (x_sorted[:mu] - x_mean_old) / sigma
            
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(dx_m, dx_m) + (1 - hs) * cc * (1 - cc - cc) * np.outer(pc, pc))
            C += cmu * np.sum([w[i] * np.outer(dx_mu[i], dx_mu[i]) for i in range(mu)], axis=0)
            
            try:
                L = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(L))
                B = L / np.outer(D, np.ones(d))
            except np.linalg.LinAlgError:
                break

            if sigma < 1e-12:
                break

        return best_x, best_f
