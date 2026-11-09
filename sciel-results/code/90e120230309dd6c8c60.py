import numpy as np

class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    CMA-ES adapts the covariance matrix, effectively estimating the rotation
    and scaling of the function's contours, which is ideal for Katsuura-like
    functions with hidden rotations.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0

    def __call__(self, func):
        self.call_count = 0
        d = self.dim
        budget = self.budget
        
        # Fixed seed for deterministic behavior
        rng = np.random.RandomState(42)
        
        # Define a wrapper to count evaluations and enforce budget
        def wrapped_func(x):
            if self.call_count >= budget:
                return 1e10
            self.call_count += 1
            x = np.asarray(x, dtype=float)
            return float(func(x))

        # CMA-ES Parameters
        # Population size
        lam = 12
        # Weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1) - 1)) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_cov = (2 + 2 * min(weights[0], 0.5 / mu_eff)) / (d + 1 + 3 * mu_eff)
        
        # State initialization
        m = np.zeros(d)
        C = np.eye(d)
        sigma = 1.0
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Cholesky factor
        B = np.eye(d)
        
        best_x = m.copy()
        best_f = wrapped_func(best_x)
        
        # Main loop
        while self.call_count < budget:
            # Generate population
            xs = []
            ys = []
            fs = []
            
            for i in range(lam):
                if self.call_count >= budget:
                    break
                z = rng.randn(d)
                y = B @ z
                x = m + sigma * y
                # Clip to bounds [-5, 5]
                x = np.clip(x, -5, 5)
                xs.append(x)
                f = wrapped_func(x)
                fs.append(f)
                ys.append(y)
                
            if len(xs) < lam:
                break
                
            # Sort by fitness
            indices = np.argsort(fs)
            xs = [xs[i] for i in indices]
            fs = [fs[i] for i in indices]
            ys = [ys[i] for i in indices]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
                
            # Update mean
            m_new = np.sum([weights[i] * xs[i] for i in range(mu)], axis=0)
            
            # Update path accumulators
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (m_new - m) / sigma
            
            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            # Update covariance matrix
            y_w = np.array(ys[:mu])
            rank_mu = np.outer(y_w.T, y_w * weights[:, np.newaxis])
            C = (1 - c_cov) * C + c_cov * (rank_mu + (1 + (1 - c_c) * c_c) * np.outer(pc, pc))
            
            # Symmetrize and ensure positive definiteness
            C = 0.5 * (C + C.T)
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(d)
                B = np.linalg.cholesky(C)
                
            m = m_new
            
            # Convergence check
            if sigma < 1e-12 or np.linalg.norm(m_new - m) < 1e-10:
                break
                
        return best_x, best_f
