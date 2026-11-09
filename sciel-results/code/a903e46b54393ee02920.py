import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        d = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x = None
        best_f = np.inf
        n_calls = 0

        def _call(x):
            nonlocal n_calls, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            n_calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial point: Center is often a good start for BBOB, 
        # but random is safer if shift is unknown. 
        # Let's start with a random point in [-2, 2] to be close to center but not exactly.
        rng = np.random.default_rng(42)
        
        # We will run a loop that manages restarts.
        # Each "run" is a (1+1)-ES with covariance adaptation.
        # We restart if stagnation is detected.
        
        while n_calls < budget and time.time() < deadline:
            # Initialize a new run
            # Start from the global best if available, else random
            if best_x is None:
                x = rng.uniform(-2, 2, d)
                fx = _call(x)
            else:
                x = best_x.copy()
                fx = best_f
            
            # Initialize CMA-ES parameters
            sigma = 2.0  # Initial step size
            C = np.eye(d)
            Cs = np.eye(d)  # Cholesky factor of C
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            
            # CMA-ES parameters for (1+1)
            # Standard CMA-ES uses population, but (1+1) is simpler.
            # We use a simplified covariance update.
            
            c_sigma = 0.5  # Learning rate for sigma
            c_c = 0.5  # Learning rate for covariance path
            c_1 = 0.5  # Learning rate for rank-1 update
            c_mu = 0.1  # Learning rate for rank-mu update (not used in strict 1+1, but we can approximate)
            
            # We will use a simple adaptive step size and covariance.
            # To ensure robustness, we track the last few function values.
            last_improvement_step = n_calls
            stagnation_limit = 30  # Restart if no improvement in 30 steps
            
            run_calls = 0
            max_run_calls = 100  # Limit calls per run to allow multiple restarts
            
            while n_calls < budget and time.time() < deadline and run_calls < max_run_calls:
                run_calls += 1
                
                # Sample noise
                z = rng.standard_normal(d)
                
                # Update evolution paths
                # p_sigma update
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (z / sigma)
                # Simple sigma update based on norm of p_sigma
                # Target norm is 1
                norm_p_sigma = np.linalg.norm(p_sigma)
                sigma *= np.exp(0.5 * (norm_p_sigma - 1.0))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Candidate point
                y = np.clip(x + sigma * (Cs @ z), -5, 5)
                fy = _call(y)
                
                # Update covariance and position
                if fy < fx:
                    # Success
                    # Update p_c
                    p_c = (1 - c_c) * p_c + c_c * (z / sigma)
                    # Update C (Rank-1 update)
                    # C = (1 + c_1) * C + c_1 * (z/sigma @ z/sigma.T) - c_1 * C
                    # This is a simplified view. Proper CMA-ES uses more terms.
                    # For (1+1), a common approximation is:
                    # C <- (1 - c_1) * C + c_1 * (v @ v.T) where v = z/sigma
                    v = z / sigma
                    C = (1 - c_1) * C + c_1 * np.outer(v, v)
                    
                    # Regularize C to ensure positive semi-definite
                    # and update Cholesky factor
                    # Add small diagonal for stability
                    C += 1e-8 * np.eye(d)
                    try:
                        Cs = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # Fallback to identity if Cholesky fails
                        C = np.eye(d)
                        Cs = np.eye(d)
                    
                    x, fx = y, fy
                    last_improvement_step = n_calls
                else:
                    # Failure
                    # Optional: Rank-mu update or just decay C
                    # To avoid getting stuck in a bad direction, we can slightly shrink C
                    C = 0.99 * C
                    try:
                        Cs = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(d)
                        Cs = np.eye(d)
                
                # Check for stagnation
                if n_calls - last_improvement_step > stagnation_limit:
                    # Restart
                    break

        return best_x, best_f
