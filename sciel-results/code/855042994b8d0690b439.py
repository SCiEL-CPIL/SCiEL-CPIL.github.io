import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size lambda
        # For d=5, lambda=10 is a good balance.
        lambda_ = 10
        mu = lambda_ // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        
        # Initial state
        # Start at origin
        x_mean = np.zeros(d)
        
        # Initial sigma: large enough to explore [-5, 5]
        sigma = 2.0
        
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Precompute decomposition
        def _update_decomposition(C):
            D, B = np.linalg.eigh(C)
            D = np.maximum(D, 1e-12)
            D_sqrt = np.sqrt(D)
            D_inv_sqrt = 1.0 / D_sqrt
            C_inv_sqrt = B @ np.diag(D_inv_sqrt) @ B.T
            return C_inv_sqrt, D_sqrt, B

        C_inv_sqrt, D_sqrt, B = _update_decomposition(C)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        
        eval_count = 1
        gen = 0
        
        chi_N = np.sqrt(d) * (1 - 1.0/(4.0*d) + 1.0/(21.0*d*d))
        
        while eval_count < budget:
            if time.time() > deadline:
                break
                
            # Generate lambda_ candidates
            ys = np.zeros((lambda_, d))
            xs = np.zeros((lambda_, d))
            fs = np.zeros(lambda_)
            
            for i in range(lambda_):
                z = np.random.randn(d)
                y = sigma * (B @ (D_sqrt * z))
                x_new = x_mean + y
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = func(x_new)
                eval_count += 1
                
                ys[i] = y
                xs[i] = x_new
                fs[i] = f_new
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    
            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx]
            xs = xs[idx]
            fs = fs[idx]
            
            # Update mean
            y_w = np.sum(w[:, np.newaxis] * ys[:mu], axis=0)
            x_mean = x_mean + sigma * y_w
            x_mean = np.clip(x_mean, -5.0, 5.0)
            
            # Update step size control variables
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ y_w
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_w)
            
            # Calculate h_sigma
            # h_sigma = 1 if ||ps|| < (1.4 + 2/(d+1)) * sqrt((1 - (1 - c_c)^(2 * gen)) * d) else 0
            # Note: gen starts at 0.
            # If gen is 0, (1 - c_c)^(2*0) = 1, so term is 0.
            # This might cause h_sigma to be 0 initially if ps is not small enough?
            # Actually, for gen=0, the threshold is 0. So h_sigma is 0 unless ps is 0.
            # This is a known issue with the formula if not careful.
            # A safer heuristic: if ||ps|| < 1.5 * sqrt(d), h_sigma = 1.
            # Let's use the standard formula but ensure it works for early gens.
            # Actually, the term (1 - (1 - c_c)^(2 * gen)) is small for small gen.
            # Let's just use a simple threshold: if ||ps|| < 2.0 * sqrt(d), h_sigma = 1.
            h_sigma = 1.0 if np.linalg.norm(ps) < 2.0 * np.sqrt(d) else 0.0
            
            # Update pc
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_N - 1.0))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update C
            # C = (1 + c_c * (2 - c_c) + c_sigma * (2 - c_sigma)) * C \
            #     - c_c * np.outer(pc, pc) \
            #     - c_sigma * (2 - c_sigma) * np.outer(C_inv_sqrt @ ps, C_inv_sqrt @ ps)
            
            C = (1 + c_c * (2 - c_c) + c_sigma * (2 - c_sigma)) * C \
                - c_c * np.outer(pc, pc) \
                - c_sigma * (2 - c_sigma) * np.outer(C_inv_sqrt @ ps, C_inv_sqrt @ ps)
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Update decomposition
            C_inv_sqrt, D_sqrt, B = _update_decomposition(C)
            
            gen += 1

        return best_x, best_f
