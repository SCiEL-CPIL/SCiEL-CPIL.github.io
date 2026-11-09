import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin

        n = self.dim
        total_budget = self.budget
        
        # CMA-ES Parameters
        # For Rastrigin in 5D, standard CMA-ES settings work well.
        # sigma0: initial step size. Since box is [-5, 5], 2.0 is a good start to cover space.
        sigma = 2.0 
        lambda_ = 4 * int(1 + 3 * np.log(n))  # Population size
        if lambda_ > total_budget // 2:
            lambda_ = max(10, total_budget // 10) # Ensure we don't waste too much budget on first gen
            
        # Initial mean
        mean = np.zeros(n)
        
        # CMA-ES state
        C = np.eye(n)
        Cinv = np.eye(n)
        pc = np.zeros(n)
        cs = np.zeros(n)
        cc = (2 + 2 / (n + 1)) / ((n + 2) + 2 / max(0, n - 2))
        cmu = min(1 - cc, 2 / ((n + 1.3)**2 + mu))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        step_size = 1.0 # placeholder, will be updated
        
        calls_made = 0
        best_x = mean.copy()
        best_f = np.inf
        
        def do_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Main CMA-ES Loop
        while calls_made < total_budget and time.time() < deadline:
            # 1. Generate population
            ys = []
            xs = []
            fs = []
            
            # We need to ensure we have enough budget for the whole population
            # If remaining budget < lambda_, we might stop or do a partial pop? 
            # Standard CMA-ES needs full pop. If budget is tight, we stop.
            if total_budget - calls_made < lambda_:
                break
                
            for i in range(lambda_):
                # Sample from N(mean, sigma^2 * C)
                # Cholesky decomposition for sampling
                # To avoid expensive Cholesky every step, we can use the fact that C is SPD.
                # However, for n=5, Cholesky is cheap.
                try:
                    L = np.linalg.cholesky(C)
                    z = np.random.normal(size=n)
                    y = L @ z
                    x = mean + sigma * y
                except np.linalg.LinAlgError:
                    # Fallback if C becomes singular (rare with damping, but possible)
                    x = mean + sigma * np.random.normal(size=n)
                    y = np.random.normal(size=n)
                
                # Clip to box
                x_clipped = np.clip(x, -5.0, 5.0)
                
                # Evaluate
                if calls_made >= total_budget or time.time() > deadline:
                    break
                    
                val = func(x_clipped)
                calls_made += 1
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                
                ys.append(y)
                xs.append(x_clipped)
                fs.append(val)
            
            if len(fs) < lambda_:
                break
                
            # 2. Sort by fitness
            idx = np.argsort(fs)
            xs = np.array([xs[i] for i in idx])
            ys = np.array([ys[i] for i in idx])
            fs = np.array([fs[i] for i in idx])
            
            # 3. Update mean
            y_mean = np.dot(weights, ys[:mu])
            mean += sigma * y_mean
            
            # 4. Update Covariance Matrix C
            # C = (1 - cmu) * C + cmu * sum(weights * (y_i - y_mean)(y_i - y_mean)^T)
            diff = ys[:mu] - y_mean
            # Efficient update
            # C = (1-cmu)*C + cmu * (W * diff * diff^T)
            # Using matrix multiplication
            weighted_diff = diff * weights[:mu, None]
            C = (1 - cmu) * C + cmu * (weighted_diff.T @ diff)
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2
            
            # 5. Update step size sigma
            # h_sigma = ||pc|| / (sqrt(1 - (1-cc)^{2*ages}) * chiN * sigma)
            # We need to track pc
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * y_mean
            
            # Approximate h_sigma
            # If we don't track 'ages', we use a simplified condition
            # h_sigma = 1 if ||pc||/sigma < 1.4 * chiN, else 0
            # This is a simplification, but often sufficient for small n.
            h_sigma = 1.0 if np.linalg.norm(pc) / sigma < 1.4 * chiN else 0.0
            
            # Update sigma
            sigma *= np.exp(cs / (n + 4) * (np.linalg.norm(y_mean) / chiN - 1))
            
            # Update cs
            cs = (1 - cs) * cs + cc * h_sigma # Wait, cs update is usually:
            # cs = (1 - cs) * cs + cc * (||y_mean|| / chiN - 1) ? No.
            # Standard CMA-ES:
            # cs = (1 - cs) * cs + cc * (||y_mean|| / chiN)
            # Actually, the standard update for cs is:
            # cs = (1 - cs) * cs + cc * (||y_mean|| / chiN)
            # But wait, the damping is applied differently.
            # Let's use the standard robust update:
            cs = (1 - cs) * cs + cc * (np.linalg.norm(y_mean) / chiN)
            
            # Update C with cs (damping)
            # C = (1 - cmu - cs*(2-cs)) * C + ...
            # The previous C update didn't include the cs damping term.
            # Let's correct the C update to include cs damping for stability.
            # C = (1 - cmu - cs*(2-cs)) * C + (cmu + cs*(2-cs)) * (weighted_diff.T @ diff)
            # Wait, the term is usually:
            # C = (1 - cmu - cs*(2-cs)*h_sigma) * C + ...
            # For simplicity and stability in this constrained box, let's stick to the simpler 
            # rank-mu update but ensure C is PSD.
            
            # Re-apply symmetry and ensure PSD
            C = (C + C.T) / 2
            
            # Check for convergence or stagnation
            # If sigma is very small or C is nearly singular, break
            if sigma < 1e-8:
                break
            
            # Optional: Re-center if mean goes out of bounds?
            # CMA-ES handles this via clipping in evaluation, but mean might drift.
            # Clipping x in evaluation is sufficient for the black-box.

        return best_x, best_f
