"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for Büche-Rastrigin f4 on [-5,5]^5 using Cross-Entropy Method.
    
    Strategy:
    1. Initial exploration to establish baseline
    2. CEM (Cross-Entropy Method) to iteratively concentrate search
       on the region containing the global minimum
    3. Final local refinement with adaptive (1+1)-ES
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 45s safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        best_x = np.zeros(dim)
        best_f = float('inf')
        budget_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Initial exploration (30 evaluations)
        n_explore = 30
        for _ in range(n_explore):
            if budget_used >= budget or not time_ok():
                break
            pt = np.clip(rng.uniform(-5, 5, dim), -5, 5)
            f_val = func(pt)
            budget_used += 1
            update_best(pt, f_val)

        # Phase 2: Cross-Entropy Method
        # Maintain a multivariate normal distribution, sample from it,
        # keep the top fraction, update mean and covariance
        
        if budget_used < budget and time_ok():
            # Initialize CEM parameters from best point
            mean = best_x.copy()
            # Start with a wide covariance that covers the search space
            cov = np.eye(dim) * 4.0  # std ~ 2.0 per dimension
            cov_scale = 4.0  # current std per dim

            n_cem = min(350, budget - budget_used)
            pop_size = 50
            elite_ratio = 0.25  # keep top 25%
            n_elite = max(5, int(pop_size * elite_ratio))
            mix_rate = 0.3  # blend old and new parameters

            for iteration in range(n_cem // pop_size + 1):
                if budget_used + pop_size > budget or not time_ok():
                    # Use remaining budget
                    pop_size = max(5, budget - budget_used)
                    if pop_size < 5:
                        break
                    n_elite = max(3, int(pop_size * elite_ratio))
                
                # Sample from current distribution
                samples = []
                values = []
                for _ in range(pop_size):
                    if budget_used >= budget or not time_ok():
                        break
                    try:
                        # Use Cholesky for proper multivariate normal sampling
                        L = np.linalg.cholesky(cov)
                    except np.linalg.LinAlgError:
                        # If cov is not positive definite, reset to diagonal
                        cov = np.eye(dim) * cov_scale
                        L = np.sqrt(cov_scale) * np.eye(dim)
                    
                    z = rng.standard_normal(dim)
                    sample = mean + L @ z
                    sample = np.clip(sample, -5, 5)
                    
                    f_val = func(sample)
                    budget_used += 1
                    samples.append(sample)
                    values.append(f_val)
                    update_best(sample, f_val)
                
                if len(samples) < n_elite:
                    break
                
                samples = np.array(samples[:len(values)])
                values = np.array(values[:len(values)])
                
                # Find elite indices
                elite_idx = np.argsort(values)[:n_elite]
                elite_samples = samples[elite_idx]
                
                # Update mean
                new_mean = np.mean(elite_samples, axis=0)
                mean = mix_rate * mean + (1 - mix_rate) * new_mean
                
                # Update covariance
                if n_elite > 1:
                    centered = elite_samples - new_mean
                    new_cov = (centered.T @ centered) / (n_elite - 1)
                    # Add small regularization for stability
                    new_cov += 1e-6 * np.eye(dim)
                    cov = mix_rate * cov + (1 - mix_rate) * new_cov
                    # Ensure positive definiteness
                    try:
                        np.linalg.cholesky(cov)
                    except np.linalg.LinAlgError:
                        cov = cov + 0.1 * np.eye(dim)
                
                # Gradually reduce covariance scale to focus search
                cov_scale *= 0.95
                # Scale covariance to maintain desired spread
                current_std = np.sqrt(np.mean(np.diag(cov)))
                if current_std > 0:
                    cov *= (cov_scale / current_std) ** 2

        # Phase 3: Final local refinement with adaptive (1+1)-ES
        remaining = budget - budget_used
        if remaining > 5 and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            min_sigma = 1e-5
            max_sigma = 1.0
            window_size = 5
            success_window = []
            
            for _ in range(remaining):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                budget_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_window.append(1)
                    update_best(x, fx)
                else:
                    success_window.append(0)
                
                # Adapt sigma
                if len(success_window) >= window_size:
                    success_rate = np.mean(success_window[-window_size:])
                    if success_rate > 0.3:
                        sigma *= 1.15
                    elif success_rate < 0.1:
                        sigma *= 0.85
                    if len(success_window) > window_size:
                        success_window.pop(0)
                    sigma = max(min_sigma, min(sigma, max_sigma))

        return best_x, best_f
# EVOLVE-BLOCK-END
