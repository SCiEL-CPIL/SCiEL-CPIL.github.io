import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            dim = self.dim
            budget = self.budget
            start_time = time.time()
            deadline = start_time + 280.0 # Safety margin

            # CMA-ES Parameters
            pop_size = 4 + int(3 * np.log(dim))
            # Ensure pop_size is at least 2
            pop_size = max(2, pop_size)
            
            # Initial population
            x0 = np.random.uniform(-5, 5, dim)
            sigma = 1.0
            
            # CMA-ES State
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            cs = (pop_size + 2) / (dim + 4)
            ds = 1 + 2 * max(0, np.sqrt((np.sqrt(dim) - 1) / (np.sqrt(dim) + 1)) - 1) / cs
            cmu = 2 / ((dim + 1.3)**2 + pop_size)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(20*dim**2))
            
            # Best solution tracking
            best_x = x0.copy()
            best_f = func(best_x)
            calls = 1
            
            # Main Loop
            while calls < budget:
                # Check time
                if time.time() > deadline:
                    break
                    
                # Generate population
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim) # Fallback if singular
                
                # Generate normal random numbers
                z = np.random.randn(pop_size, dim)
                # Transform to standard normal with covariance C
                y = np.array([L @ zi for zi in z])
                # Add mean
                pop = np.array([x0 + sigma * yi for yi in y])
                
                # Evaluate population
                fvals = []
                for p in pop:
                    if calls >= budget:
                        break
                    if time.time() > deadline:
                        break
                    # Clip to bounds
                    p_clipped = np.clip(p, -5, 5)
                    f_val = func(p_clipped)
                    fvals.append(f_val)
                    calls += 1
                    if f_val < best_f:
                        best_f = f_val
                        best_x = p_clipped.copy()
                
                if len(fvals) < pop_size:
                    break # Out of budget or time
                
                # Sort by fitness
                idx = np.argsort(fvals)
                pop_sorted = pop[idx]
                fvals_sorted = np.array(fvals)[idx]
                
                # Update mean
                mu = pop_size // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= np.sum(weights)
                
                x_new = np.sum(pop_sorted[:mu] * weights[:, np.newaxis], axis=0)
                
                # Update step size (Sigma)
                # Calculate ps
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (L.T @ (x_new - x0) / sigma)
                
                # Update C
                # Calculate pc
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * (x_new - x0) / sigma
                
                # Rank-1 update
                C = (1 - cmu * (1 + cmu)) * C + cmu * np.outer(x_new - x0, x_new - x0) / (sigma**2)
                
                # Rank-mu update
                # This part is complex, often simplified or omitted in basic implementations.
                # For 5D and 500 evals, Rank-1 might be sufficient or we can use a simplified CMA-ES.
                
                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update mean
                x0 = x_new
                
                # Handle bounds: If x0 goes out of bounds, project it?
                # CMA-ES doesn't handle bounds natively well.
                # We clip the generated points, but the mean x0 might drift.
                # A common trick: if x0 is out of bounds, reflect or clip.
                # Clipping x0 can distort the distribution.
                # Better: keep x0 unconstrained but clip samples.
                # However, if x0 is far out of bounds, all samples might be clipped to the boundary, causing stagnation.
                # Strategy: If x0 is outside [-5, 5], move it towards the center?
                # Or simply clip x0 for the next iteration's center?
                # Let's clip x0 to be safe, though it's not theoretically pure.
                x0 = np.clip(x0, -5, 5)
                
                # Reset C if it becomes ill-conditioned?
                # For simplicity, we assume it stays stable.
                
            return best_x, best_f
