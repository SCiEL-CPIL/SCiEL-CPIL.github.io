import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0
        
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            calls += 1
            try:
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except:
                return None

        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        # Weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Initialization
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, large enough to explore [-5,5]
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        # Evolution paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Damping parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        def update_sigma():
            nonlocal sigma, ps, cs
            # Update evolution path for sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / np.linalg.norm(C @ np.eye(dim) ** 0.5) * (x_mean / sigma - (1 - cc) * pc) 
            # Note: The standard formula uses C^{-1/2} (x_mean - x_mean_old) / sigma
            # But in the update loop, we have the old mean.
            # Let's stick to the standard update logic inside the loop.
            pass

        # We need to store old mean
        x_mean_old = x_mean.copy()
        
        while calls < budget and time.time() < deadline:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            xs = np.zeros((lam, dim))
            # Cholesky decomposition of C for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
            
            for i in range(lam):
                z = np.random.randn(dim)
                xs[i] = x_mean + sigma * (L @ z)
                # Clip to bounds
                xs[i] = np.clip(xs[i], -5, 5)
                
            # Evaluate
            fs = []
            valid_indices = []
            for i in range(lam):
                if calls + 1 > budget:
                    break
                fi = safe_call(xs[i])
                if fi is not None:
                    fs.append(fi)
                    valid_indices.append(i)
            
            if len(fs) == 0:
                break
                
            # Sort by fitness
            sorted_indices = np.argsort(fs)
            xs_sorted = xs[valid_indices][sorted_indices]
            fs_sorted = np.array(fs)[sorted_indices]
            
            # Update best
            # safe_call already updates best_x and best_f
            
            # Update mean
            x_mean_new = np.sum([weights[i] * xs_sorted[i] for i in range(mu)], axis=0)
            
            # Update covariance matrix
            # Standard CMA-ES update
            # y_k = (x_mean_new - x_mean) / sigma
            y_k = (x_mean_new - x_mean) / sigma
            
            # Update pc
            pc = (1 - cc) * pc + cc * y_k
            
            # Update ps
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ y_k
            # We need C_inv_sqrt. We can compute it from Cholesky or SVD.
            # For simplicity and robustness, we can approximate or compute SVD each step (expensive for large dim, but dim=5 is small).
            # SVD of C: C = U S V^T. C_inv_sqrt = V S^{-1/2} U^T.
            U, S, Vt = np.linalg.svd(C)
            C_inv_sqrt = Vt.T @ np.diag(1.0 / np.sqrt(S)) @ U.T
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ y_k)
            
            # Update C
            # Rank-mu update
            v_k = np.zeros((dim, mu))
            for i in range(mu):
                v_k[:, i] = (xs_sorted[i] - x_mean) / sigma
            
            # H = (1/lambda) * sum (x_i - x_mean)(x_i - x_mean)^T + ...
            # Standard formula:
            # C = (1 - c1 - cmu) * C + c1 * (pc pc^T + (1 - delta_sigma) * ps ps^T) + cmu * sum w_i (x_i - x_mean)(x_i - x_mean)^T / sigma^2
            
            # Note: delta_sigma is a small term, often omitted or set to 0 for simplicity in basic implementations.
            # We will omit it for simplicity, it's a minor correction.
            
            C_new = (1 - c1 - cmu) * C
            C_new += c1 * (np.outer(pc, pc) + (1 - (1 - cc)**2) * np.outer(ps, ps))
            
            for i in range(mu):
                C_new += (cmu * weights[i]) * np.outer(v_k[:, i], v_k[:, i])
                
            # Ensure symmetry
            C_new = (C_new + C_new.T) / 2.0
            
            # Update sigma
            # Delta_sigma is approx ||ps||^2 / dim - 1
            delta_sigma = (np.linalg.norm(ps)**2 / dim) - 1
            sigma = sigma * np.exp(cs / (dim + 3) * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
            
            # Update state
            x_mean = x_mean_new
            C = C_new
            
            # If sigma becomes too small, we might be converged
            if sigma < 1e-8:
                break

        return best_x, best_f
