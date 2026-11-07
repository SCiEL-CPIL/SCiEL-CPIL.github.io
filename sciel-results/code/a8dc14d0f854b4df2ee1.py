import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        N = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(N))
        if lam < 6: lam = 6
        
        w = np.log((lam + 1) / 2) - np.log(np.arange(1, lam + 1))
        w = w / w.sum()
        
        mu = lam // 2
        w = w[:mu]
        w = w / w.sum()
        
        mueff = 1.0 / np.sum(w**2)
        
        cs = (mueff + 1) / (N + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (N + 1)) - cs
        
        c1 = (2 / ((N + 1.3)**2 + mueff)) * (2 - (N + 1.3) / (N + 1.3 + 2))
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((N + 2)**2 + mueff)))
        
        rng = np.random.default_rng(42)
        m = np.zeros(N)
        sigma = 2.0
        
        C = np.eye(N)
        pc = np.zeros(N)
        ps = np.zeros(N)
        
        B = np.linalg.cholesky(C)
        
        calls = 0
        best_x = m.copy()
        best_f = np.inf
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            z = rng.standard_normal((lam, N))
            y = z @ B.T
            x = m[None, :] + sigma * y
            x = np.clip(x, -5, 5)
            
            f = np.array([func(xi) for xi in x])
            calls += lam
            
            idx = np.argsort(f)
            x_sorted = x[idx]
            f_sorted = f[idx]
            y_sorted = y[idx]
            
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
                
            m_new = m + sigma * np.sum(w[:, None] * y_sorted[:mu], axis=0)
            step = (m_new - m) / sigma
            
            # Update step size
            # ps update
            # C^(-1/2) step
            v = np.linalg.solve(B.T, step)
            w_step = np.linalg.solve(B, v)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * w_step
            norm_ps = np.linalg.norm(ps)
            
            k = calls // lam
            c1_term = 1 - (1 - cs)**(2 * (k + 1) / lam)
            c2_term = (1 - cs)**(2 * (k + 1) / lam) + cs * (2 - cs)
            hsig = 1.0 if norm_ps / np.sqrt(1 - c1_term) < (1.4 + 2.0 / (N + 1.0)) * np.sqrt(c2_term) else 0.0
            
            sigma = sigma * np.exp(cs / ds * (norm_ps / np.sqrt(2 * N) - 1.0))
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Update covariance
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * step
            
            C_mu = y_sorted[:mu].T @ (w[:, None] * y_sorted[:mu])
            
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) + (1 - hsig) * c1 * (2 - c1) * C)
            C += cmu * C_mu
            
            C = (C + C.T) / 2.0
            
            for _ in range(10):
                try:
                    B = np.linalg.cholesky(C)
                    break
                except np.linalg.LinAlgError:
                    C += 1e-4 * np.eye(N)
            else:
                C = np.eye(N)
                B = np.eye(N)
            
            m = m_new
            
        return best_x, best_f
