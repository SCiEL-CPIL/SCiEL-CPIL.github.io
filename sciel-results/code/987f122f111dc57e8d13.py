import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        
        n = self.dim
        budget = self.budget
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol Scan to locate approximate region
        # Use scipy.stats.qmc for high-quality low-discrepancy sampling
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n, scramble=True, seed=42)
            # Number of points for the scan
            # 5D, budget 500. Let's use 32 points for a good spread.
            # 2^5 = 32 points is a good start.
            num_scan_points = 32
            if budget < num_scan_points:
                num_scan_points = budget
            
            # Generate points in [0, 1]^n
            sobol_points = sampler.random(n=num_scan_points)
            
            # Map to [-5, 5]^n
            for p in sobol_points:
                x = 10.0 * p - 5.0
                evaluate(x)
                
                if calls_made >= budget or time.time() > deadline:
                    break
        except Exception:
            # Fallback to random if scipy fails
            for _ in range(min(32, budget)):
                if calls_made >= budget or time.time() > deadline:
                    break
                x = np.random.uniform(-5, 5, n)
                evaluate(x)

        # Phase 2: (1+1)-ES with step-size adaptation
        # Start from the best point found in Phase 1
        
        if calls_made < budget and time.time() <= deadline:
            x = best_x.copy()
            fx = best_f
            
            # Initial step size: somewhat large to allow exploration, 
            # but constrained by the box.
            # The minimizer is in [-4, 4]^5.
            # A step size of 1.0 is reasonable.
            sigma = 1.0
            
            # Parameters for (1+1)-ES
            # c_sigma: adaptation rate
            c_sigma = 1.0 / (n + 3)
            # d_sigma: normalization factor
            d_sigma = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 1)) - 1) + c_sigma
            
            # Path accumulator for CSA
            p_sigma = np.zeros(n)
            
            # Expected norm of standard normal
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            # We will run until budget or deadline
            # Each iteration uses 1 evaluation
            while calls_made < budget and time.time() < deadline:
                # Generate candidate
                z = np.random.standard_normal(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                
                fy = evaluate(y)
                
                if time.time() > deadline:
                    break
                
                # Update path for CSA
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (y - x) / (sigma + 1e-30)
                norm_p_sigma = np.linalg.norm(p_sigma)
                
                # Success?
                success = fy < fx
                if success:
                    x = y
                    fx = fy
                    # best_x and best_f are updated in evaluate()
                # else: x and fx remain unchanged
                
                # Adapt sigma
                # h_sigma is typically 1 if norm is not too large, 0 otherwise
                # For simplicity, we use a basic adaptation rule
                # sigma *= exp(c_sigma/d_sigma * (norm_p_sigma/chi_n - 1))
                # This is the standard CSA rule
                
                # To avoid divergence, we can clamp sigma
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                sigma = np.clip(sigma, 1e-6, 5.0)

        return best_x, best_f
