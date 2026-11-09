import numpy as np

class Optimizer:
    """CMA-ES optimizer on the box [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        
        self.global_best_x = None
        self.global_best_f = float('inf')
        self.calls_count = 0

        def track_func(x):
            self.calls_count += 1
            f_val = func(x)
            if f_val < self.global_best_f:
                self.global_best_f = f_val
                self.global_best_x = x.copy()
            return f_val

        # CMA-ES Parameters
        lam = 10
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        mueff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff)))
        
        mean = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        gen = 0
        
        # Initial evaluation
        f0 = track_func(mean)
        if self.calls_count >= total_budget:
            return self.global_best_x, self.global_best_f

        while self.calls_count < total_budget:
            if total_budget - self.calls_count < lam:
                break
                
            gen += 1
            
            # Sample population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-10 * np.eye(n)
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(n)
                    
            ys = np.random.randn(lam, n)
            xs = mean + sigma * (ys @ L.T)
            
            # Evaluate
            fs = []
            for x in xs:
                if self.calls_count >= total_budget:
                    break
                fs.append(track_func(x))
            
            if len(fs) < lam:
                break
                
            idx = np.argsort(fs)
            xs = xs[idx]
            fs = np.array(fs[idx])
            
            # Update mean
            y = np.dot(weights, xs[:mu])
            mean = mean + y
            
            # Update step size
            norm_y = np.linalg.norm(y)
            expected_norm = sigma * np.sqrt(n)
            sigma = sigma * np.exp(c_sigma * (norm_y / expected_norm - 1))
            sigma = np.clip(sigma, 1e-12, 1e3)
            
            # Update pc
            pc = (1 - c_c) * pc + c_c * (y / sigma)
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (y / sigma)
            
            # Delta sigma
            delta_sigma = 1 if (np.dot(ps, ps) / (1 - (1 - c_sigma) ** (2 * gen)) < (1 + 3 / n) * (n + 1)) else 0
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - (1-c_c)^2) * delta_sigma) + c_mu * sum(weights * (y_i * y_i.T))
            
            # Rank one update
            term1 = pc * pc.T + (1 - (1 - c_c) ** 2) * delta_sigma
            
            # Rank mu update
            term2 = np.zeros((n, n))
            for i in range(mu):
                yi = (xs[i] - (mean - y)) / sigma # y_i = (x_i - old_mean) / sigma
                term2 += weights[i] * np.outer(yi, yi)
                
            C = (1 - c_1 - c_mu) * C + c_1 * term1 + c_mu * term2
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Check convergence?
            # If sigma is very small, we might be stuck.
            # But budget is the main constraint.

        return self.global_best_x, self.global_best_f
