"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the Rastrigin rotated function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(n))
        n = self.dim
        lam = 4 + int(3 * np.log(n))
        lam = max(lam, 2 * n)  # Ensure reasonable population size
        
        # Initial point: center of the box
        x0 = np.zeros(n)
        
        # Initial step size: roughly 1/10 of the box width
        sigma = 1.0
        
        # CMA-ES parameters
        mu = lam // 2  # Number of selected parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / mueff) / ((n + 2)**2 + mueff))
        
        # Covariance matrix
        C = np.eye(n)
        
        # Evolution path for covariance
        p_c = np.zeros(n)
        
        # Evolution path for sigma
        p_sigma = np.zeros(n)
        
        # Cholesky factor of C
        try:
            B = np.linalg.cholesky(C)
            D = np.sqrt(np.diag(C))
        except np.linalg.LinAlgError:
            B = np.eye(n)
            D = np.ones(n)
        
        # Initial evaluation
        x0_clipped = np.clip(x0, -5, 5)
        f0 = func(x0_clipped)
        calls_made += 1
        best_x = x0_clipped.copy()
        best_f = f0
        
        # CMA-ES main loop
        while calls_made < self.budget and time.time() < deadline:
            # Generate lambda candidate points
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if time.time() > deadline:
                    break
                    
                # Sample from multivariate normal
                z = rng.standard_normal(n)
                y = B @ z / D  # Transform to unit sphere
                x = x0 + sigma * y
                x = np.clip(x, -5, 5)
                
                # Evaluate
                fx = func(x)
                calls_made += 1
                
                ys.append(y)
                xs.append(x)
                fs.append(fx)
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    
                if calls_made >= self.budget:
                    break
            
            if len(ys) < 2:
                break
                
            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)
            
            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            xs = xs[order]
            fs = fs[order]
            
            # Update mean
            x0 = np.sum(weights[:, None] * xs[:mu], axis=0)
            
            # Update evolution path for sigma
            y_w = (x0 - xs[:mu].mean(axis=0)) / sigma if sigma > 1e-10 else np.zeros(n)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_w
            
            # Update sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update evolution path for covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * y_w
            
            # Update covariance matrix
            delta_diag = (np.linalg.norm(p_sigma) / chi_n > 1 + 4 / (n + 1)) and 1 or 0
            C = (1 + c_1 * (delta_diag - 1) - c_mu) * C
            C += c_1 * (p_c[:, None] * p_c[None, :] + (delta_diag - 1) * C)
            C += c_mu * np.sum(weights[:mu, None, None] * np.outer(ys[:mu], ys[:mu]), axis=0)
            
            # Ensure positive semi-definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = (C + C.T) / 2  # Symmetrize
            except np.linalg.LinAlgError:
                C = np.eye(n)
            
            # Update Cholesky factor
            try:
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(C))
            except np.linalg.LinAlgError:
                B = np.eye(n)
                D = np.ones(n)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
