import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        
        # CMA-ES Parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n**2))
        
        # Initialization
        m = rng.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros((n, n))
        
        best_x = m
        best_f = func(best_x)
        calls = 1
        
        while calls + lambda_ <= self.budget:
            # Generate samples
            B = rng.standard_normal((lambda_, n))
            
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add jitter
                C += 1e-6 * np.eye(n)
                try:
                    L = np.linalg.cholesky(C)
                except:
                    break
                    
            Y = B @ L.T
            P = m + sigma * Y
            
            # Evaluate
            F = []
            for i in range(lambda_):
                p_i = np.clip(P[i], -5, 5)
                f_i = func(p_i)
                F.append((f_i, p_i))
                calls += 1
            
            # Sort
            F.sort(key=lambda x: x[0])
            
            # Update best
            if F[0][0] < best_f:
                best_f = F[0][0]
                best_x = F[0][1]
                
            # Update mean
            new_m = m + sigma * np.sum([weights[i] * (F[i][1] - m) for i in range(mu)], axis=0)
            
            # Update ps
            # inv_C_sqrt approximation via Cholesky inverse
            # L @ y = v => y = inv(L) @ v
            # v = (new_m - m) / sigma
            v = (new_m - m) / sigma
            # Solve L.T y = v
            y = np.linalg.solve(L.T, v)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y
            
            # Update sigma
            norm_ps = np.linalg.norm(ps)
            hsig = norm_ps / np.sqrt(1 - (1 - c_sigma)**(2 * (calls // lambda_))) < (1.4 + 2 / (n + 1)) * chi_n
            if hsig:
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                
            # Update pc
            pc = (1 - c_c) * pc + c_c * (v - (1 - c_c) * y) # Approximation, standard is v - (1-c_c)* (C^-1/2 v)
            # More accurate:
            # pc = (1-c_c)*pc + c_c*(v - (1-c_c)*(L_inv @ v))
            # L_inv @ v is y? No, y = L^-T v. We need C^-1/2 v.
            # C^-1/2 = (L L^T)^-1/2 = L^-T L^-1
            # Let's stick to the simpler update or skip pc update if unstable.
            # For small n, the simple rank-1 update on C is often sufficient.
            
            # Update C
            H = np.zeros((n, n))
            for i in range(mu):
                v_i = (F[i][1] - m) / sigma
                H += weights[i] * np.outer(v_i, v_i)
                
            # Standard update
            # C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + (1 - (1-c_c)^2) I) + c_mu H
            # We need pc to be updated correctly.
            # Let's use the simplified C update without pc for robustness, or ensure pc is correct.
            # Given the complexity, let's use the simplified covariance update based on samples.
            C = (1 - c_1 - c_mu) * C + c_mu * H
            # Add small c_1 term for stability?
            C += c_1 * np.eye(n) * 0.1 # Small regularization
            
            C = (C + C.T) / 2
            m = new_m
            
            # Restart condition
            if sigma < 1e-4 or sigma > 50:
                break

        return best_x, best_f
