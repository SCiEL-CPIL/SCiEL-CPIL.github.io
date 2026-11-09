import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        n = self.dim
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(n)
        
        def eval_point(x):
            nonlocal calls, best_f, best_x
            if calls >= self.budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(n))
        # Weights
        w = np.log((lam + 1) / 2) - np.log(np.arange(1, lam + 1))
        w = w / np.sum(w)
        # Effective population size
        mu = lam // 2
        mueff = 1.0 / np.sum(w[:mu] ** 2)
        
        # Step size
        sigma = 1.0
        # Mean
        m = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        # Square root of covariance matrix
        C_sqrt = np.eye(n)
        C_sqrt_inv = np.eye(n)
        
        # Evolution path
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_mu = min(1 - c_c, 2 * (mueff - 2 + 1 / mueff) / (n + 2))
        
        # Initial step size estimation
        # We start with a random point to get a rough idea of the scale
        # But standard CMA-ES starts with sigma=1.
        
        # Generate initial population
        # We need to make sure we don't exceed budget in the first generation
        if calls + lam > self.budget:
            # If we can't even do one generation, just do random search
            for _ in range(self.budget - calls):
                x = np.random.uniform(-5, 5, n)
                eval_point(x)
            return best_x, best_f

        # Initial evaluation of the mean? No, we just sample.
        # But we should evaluate the first point to have a baseline?
        # CMA-ES doesn't strictly require it, but it helps.
        
        # Let's do a small random search first to find a decent starting point for the mean
        # This is a common heuristic: "restart" or "initialize" near a good point.
        # Given the budget is 500 and n=5, we can afford ~10-20 random points.
        
        n_init = min(10, self.budget - lam) # Leave room for at least one generation
        if n_init > 0:
            for _ in range(n_init):
                x = np.random.uniform(-5, 5, n)
                f = eval_point(x)
                if f is None:
                    break
            
            if calls >= self.budget:
                return best_x, best_f
            
            # Set the mean to the best point found
            m = best_x.copy()
            # Set initial sigma to something reasonable, maybe 1.0 or based on spread
            # Let's stick with 1.0 for simplicity, or maybe 0.5 if we found a good point quickly.
            sigma = 1.0
        else:
            m = np.zeros(n)
            sigma = 1.0

        # Main CMA-ES loop
        while calls + lam <= self.budget and time.time() < deadline:
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                # Sample y from standard normal
                y = np.random.randn(n)
                # x = m + sigma * C_sqrt @ y
                x = m + sigma * (C_sqrt @ y)
                # Clip to box
                x_clipped = np.clip(x, -5, 5)
                
                f = eval_point(x_clipped)
                if f is None:
                    break
                
                ys.append(y)
                xs.append(x_clipped)
                fs.append(f)
            
            if len(ys) < lam:
                break
                
            # Sort by fitness
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Update mean
            y_w = np.sum([w[i] * ys[i] for i in range(mu)], axis=0)
            m_new = m + sigma * (C_sqrt @ y_w)
            m_new = np.clip(m_new, -5, 5)
            
            # Update step size
            # ||C^{1/2} y_w|| / ||C^{1/2} e_1|| approximation
            # Since C_sqrt is maintained, we can compute this
            # But for simplicity, we use the norm of y_w scaled by sigma?
            # Standard CMA-ES uses ||C^{1/2} y_w|| / ||C^{1/2} e_1||
            # If C is diagonal, this is easy. Here we use the full matrix.
            # Let's approximate with norm(y_w) * sigma / sigma? No.
            # The formula is: sigma = sigma * exp(c_sigma/d_sigma * (||C^{1/2} y_w|| / ||C^{1/2} e_1|| - 1))
            # We need C^{1/2} e_1.
            # C_sqrt @ e_1 is the first column of C_sqrt.
            e1 = np.zeros(n)
            e1[0] = 1.0
            c_sqrt_e1 = C_sqrt @ e1
            norm_c_sqrt_e1 = np.linalg.norm(c_sqrt_e1)
            if norm_c_sqrt_e1 < 1e-10:
                norm_c_sqrt_e1 = 1e-10
            c_sqrt_yw = C_sqrt @ y_w
            norm_c_sqrt_yw = np.linalg.norm(c_sqrt_yw)
            
            sigma *= np.exp(c_sigma / d_sigma * (norm_c_sqrt_yw / norm_c_sqrt_e1 - 1.0))
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Update evolution path
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / np.linalg.norm(y_w + 1e-10) * y_w
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) / np.linalg.norm(y_w + 1e-10) * y_w
            
            # Check for hessian update (delta function)
            delta_h = (1 - (1 - c_c) ** (2 * (calls / lam)))
            # This is a heuristic, let's just use a simple check
            # if ||p_c|| / sqrt(1 - (1-c_c)^{2*gen}) < (1.4 + 2/(n+1)) * h_n(0)
            # For simplicity, we'll just update C if the path is long enough
            
            # Update covariance matrix
            # C = (1 - c_mu) * C + c_mu * sum w_i (x_i - m)(x_i - m)^T + (1 - c_mu) * (1 - c_c^2) * p_c p_c^T
            # Actually, the standard update is:
            # C = (1 - c_mu) * C + c_mu * (sum_{i=1..mu} w_i (y_i - y_w)(y_i - y_w)^T + (1 - c_c^2) * p_c p_c^T)
            # Where y_i are the sampled points in the y-space? No, x-space.
            # Let's use the x-space update.
            # x_i are the clipped points.
            # y_w is in y-space. We need to transform it back?
            # The update formula is:
            # C <- (1 - c_mu) * C + c_mu * ( sum_{i=1..mu} w_i (x_i - m_new)(x_i - m_new)^T + (1 - c_c^2) * p_c p_c^T )
            # Note: x_i - m_new is in x-space.
            
            # Calculate the outer products
            # We need to be careful with the dimensions.
            # x_i is (n,), m_new is (n,)
            # (x_i - m_new) is (n,)
            # Outer product is (n, n)
            
            delta_C = np.zeros((n, n))
            for i in range(mu):
                diff = xs[i] - m_new
                delta_C += w[i] * np.outer(diff, diff)
            
            delta_C += (1 - c_c * (2 - c_c)) * np.outer(p_c, p_c)
            
            C = (1 - c_mu) * C + c_mu * delta_C
            
            # Update C_sqrt
            # We need to recompute the square root.
            # For n=5, this is cheap.
            try:
                C_eigvals, C_eigvecs = np.linalg.eigh(C)
                # Ensure positive definite
                C_eigvals = np.maximum(C_eigvals, 1e-10)
                C_sqrt = C_eigvecs @ np.diag(np.sqrt(C_eigvals)) @ C_eigvecs.T
                C_sqrt_inv = C_eigvecs @ np.diag(1.0 / np.sqrt(C_eigvals)) @ C_eigvecs.T
            except:
                # If it fails, reset C to identity
                C = np.eye(n)
                C_sqrt = np.eye(n)
                C_sqrt_inv = np.eye(n)
            
            m = m_new
            
            # Check for convergence
            if sigma < 1e-10:
                break
            if np.linalg.norm(C) < 1e-10:
                break

        return best_x, best_f
