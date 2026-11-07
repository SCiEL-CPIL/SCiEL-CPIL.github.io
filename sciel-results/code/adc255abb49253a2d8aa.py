import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        N = self.dim
        
        # CMA-ES Parameters
        lambda_pop = 4 + int(3 * np.log(N))
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (N + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff / N - 1) - 1) + c_sigma
        c_c = (4 + mu_eff / N) / (N + 4 + 2 * mu_eff / N)
        c_1 = 2 / ((N + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (np.sqrt(mu_eff / N + 1) - 1) / ((N + 2)**2 + mu_eff))
        
        # State
        m = np.zeros(N)
        sigma = 2.0
        C = np.eye(N)
        pc = np.zeros(N)
        ps = np.zeros(N)
        chi_N = np.sqrt(N) * (1 - 1/(4*N) + 1/(21*N**2))
        
        calls = 0
        best_f = np.inf
        best_x = m.copy()
        
        # Initial evaluation
        f0 = func(m)
        calls += 1
        best_f = f0
        
        gen = 0
        while calls + lambda_pop <= self.budget and time.time() < deadline:
            m_old = m.copy()
            
            ys = []
            xs = []
            fs = []
            
            # Ensure C is positive definite
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small regularization
                C = C + 1e-6 * np.eye(N)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Reset C
                    C = np.eye(N)
                    L = np.eye(N)
                    
            for i in range(lambda_pop):
                z = np.random.standard_normal(N)
                y = L @ z
                x = m + sigma * y
                x = np.clip(x, -5, 5)
                f = func(x)
                calls += 1
                ys.append(y)
                xs.append(x)
                fs.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            xs = [xs[i] for i in order]
            
            # Update mean
            m = np.sum([weights[i] * xs[i] for i in range(mu)], axis=0)
            
            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (m - m_old) / sigma
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_N - 1))
            
            # Update covariance
            # h_sigma calculation
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))
            if denom > 0:
                h_sigma = 1.0 if np.linalg.norm(ps) / denom < (1.4 + 2 / (N + 1)) * chi_N else 0.0
            else:
                h_sigma = 1.0
                
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - m_old) / sigma
            
            rank1 = np.outer(pc, pc)
            rankmu = np.sum([weights[i] * np.outer(ys[i], ys[i]) for i in range(mu)], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank1 + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * rankmu
            C = (C + C.T) / 2
            
            gen += 1
            
        return best_x, best_f
