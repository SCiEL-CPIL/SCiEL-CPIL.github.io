import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = None
        start_time = time.time()
        deadline = start_time + 280.0

        def wrapper(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            return func(x)

        rng = np.random.default_rng(42)

        # Phase 1: Initial sampling to find a good starting point for CMA-ES
        # Use Latin Hypercube Sampling for better coverage with few points
        n_init = 15
        # LHS: divide range into n intervals, pick one random point in each
        intervals = np.linspace(-5, 5, n_init + 1)
        init_points = np.zeros((n_init, dim))
        for j in range(dim):
            # For each dimension, permute the intervals and pick random offsets
            perm = rng.permutation(n_init)
            offsets = rng.uniform(0, 1, n_init)
            init_points[:, j] = intervals[perm] + offsets * (intervals[1] - intervals[0])
        
        init_points = np.clip(init_points, -5, 5)

        for x in init_points:
            if calls_made >= total_budget or time.time() > deadline:
                break
            try:
                fx = wrapper(x)
            except TimeoutError:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = wrapper(best_x)
            except TimeoutError:
                return np.zeros(dim), float('inf')

        # Phase 2: CMA-ES Optimization
        # CMA-ES parameters
        lambda_ = 4 * dim + 2  # Population size
        # Weights
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial step size
        sigma = 1.0
        # Mean
        m = best_x.copy()
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C (for efficient updates)
        # Using eigendecomposition for C^{-1/2}
        def invsqrt(C):
            D, V = np.linalg.eigh(C)
            D = np.where(D > 1e-12, D, 1e-12)
            return V @ np.diag(1.0 / np.sqrt(D)) @ V.T

        C_inv_sqrt = invsqrt(C)
        # Path lengths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Condition number monitoring (optional, for restarts)
        cond_C = 1.0

        while calls_made < total_budget and time.time() < deadline:
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(lambda_):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                # Sample y ~ N(0, I)
                y = rng.normal(0, 1, dim)
                # x = m + sigma * C^{1/2} * y
                # We maintain C_inv_sqrt, so C^{1/2} = (C_inv_sqrt)^{-1}
                # But it's easier to just compute x = m + sigma * (C_inv_sqrt.T @ y) if we update C_inv_sqrt properly?
                # Standard CMA-ES update:
                # x = m + sigma * (C^{-1/2} @ y) ? No, x = m + sigma * B * d * y where C = B B^T.
                # Let's use the simpler formulation: x = m + sigma * (C_inv_sqrt.T @ y) is incorrect dimensionally if not careful.
                # Let's stick to: x = m + sigma * (np.linalg.cholesky(C) @ y)
                # Cholesky is stable for SPD matrices.
                
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                x = m + sigma * (L @ y)
                x = np.clip(x, -5, 5)
                
                try:
                    fx = wrapper(x)
                except TimeoutError:
                    break
                
                ys.append(y)
                xs.append(x)
                fs.append(fx)
            
            if len(fs) == 0:
                break
                
            # Sort by fitness
            order = np.argsort(fs)
            ys = np.array([ys[i] for i in order])
            xs = np.array([xs[i] for i in order])
            fs = np.array([fs[i] for i in order])
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
            
            # Update mean
            y_w = np.sum(w[:mu] * ys[:mu], axis=0)
            
            # Update path lengths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_w)
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made // lambda_ + 1))) < 
                       (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Update covariance
            C = (1 + c_1 * (h_sigma * 2 - 1) - c_1 - c_mu) * C + \
                c_1 * np.outer(p_c, p_c) + \
                c_mu * np.sum(w[:mu] * np.outer(ys[:mu] - y_w, ys[:mu] - y_w), axis=0)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                C_inv_sqrt = invsqrt(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
            
            # Update step size
            sigma *= np.exp(c_sigma / c_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1)
            # Actually, standard update is:
            sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
            
            # Update mean
            m = m + sigma * (C_inv_sqrt @ y_w)
            m = np.clip(m, -5, 5)
            
            # Restart condition: if sigma is too small or too large
            if sigma < 1e-6 or sigma > 1e6:
                break

        return best_x, best_f
