"""Combined optimizer: multi-start (1+1)-ES + CMA-ES refinement for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive (1+1)-ES with CMA-ES refinement for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 35s safety margin before 330s kill

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        n = dim

        best_x = None
        best_f = np.inf
        call_count = 0

        def check_time():
            return time.time() > deadline - 5

        # Phase 1: Multi-start (1+1)-ES exploration
        n_restarts = 20
        exploration_budget = int(budget * 0.35)
        calls_per_restart = max(1, exploration_budget // n_restarts)

        # Create diverse starting points
        start_points = []
        # Structured points along axes
        for i in range(n):
            for val in [-3.5, -1.5, 1.5, 3.5]:
                pt = np.zeros(dim)
                pt[i] = val
                start_points.append(pt.copy())
        # Center and some random points
        start_points.append(np.zeros(dim))
        for _ in range(10):
            start_points.append(rng.uniform(-4, 4, dim))

        # Shuffle and select
        idx = rng.permutation(len(start_points))
        start_points = [start_points[i] for i in idx[:n_restarts]]

        for sp in start_points:
            if call_count >= budget - 1 or check_time():
                break

            x = np.clip(sp.copy(), -5, 5)
            fx = func(x)
            call_count += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Run (1+1)-ES with adaptive sigma
            sigma = 0.8
            successes = 0
            attempts = 0

            for _ in range(calls_per_restart - 1):
                if call_count >= budget - 1 or check_time():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1
                attempts += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                # Adaptive sigma: 1/5 success rule
                if attempts >= 5:
                    if successes / attempts > 0.5:
                        sigma *= 1.25
                    elif successes / attempts < 0.2:
                        sigma *= 0.75
                    successes = 0
                    attempts = 0

                # Stagnation detection: try a larger jump
                if sigma < 0.05 and attempts > 15:
                    sigma = 0.5
                    attempts = 0

        # Phase 2: CMA-ES refinement on the best point
        if best_x is not None:
            refinement_budget = budget - call_count
            if refinement_budget > 10 and not check_time():
                # CMA-ES parameters
                x = best_x.copy()
                sigma = 0.5
                C = np.eye(dim)
                lambda_ = 4 + int(3 * np.log(dim))  # population size ~9 for dim=5
                c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
                c_c = (4 + dim / lambda_) / (dim + 4 + 2 * lambda_ / dim)
                c_1 = 2 / ((dim + 1.3) ** 2 + lambda_)
                
                mu = lambda_ // 2
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w = w / w.sum()
                mu_eff = 1 / np.sum(w ** 2)
                c_mu = min(1 - c_1, (mu_eff - 2) / ((dim + 2) * mu_eff - 2))
                
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                
                # Precompute Cholesky factor of C
                try:
                    C_cholesky = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_cholesky = np.eye(dim)
                
                max_iters = refinement_budget // lambda_ + 1
                for _ in range(max_iters):
                    if call_count >= budget or check_time():
                        break
                    
                    # Generate lambda_ samples
                    samples = []
                    for _ in range(lambda_):
                        z = rng.standard_normal(dim)
                        y = np.clip(x + sigma * (C_cholesky @ z), -5, 5)
                        samples.append((y, func(y)))
                        call_count += 1
                        if call_count >= budget or check_time():
                            break
                    
                    if len(samples) < lambda_:
                        break

                    # Sort by fitness
                    samples.sort(key=lambda s: s[1])
                    
                    # Update best
                    if samples[0][1] < best_f:
                        best_f = samples[0][1]
                        best_x = samples[0][0].copy()

                    # Update mean
                    new_mean = np.zeros(dim)
                    for i in range(mu):
                        new_mean += w[i] * samples[i][0]
                    
                    # Update covariance
                    y_w = np.zeros(dim)
                    for i in range(mu):
                        y_w += w[i] * (samples[i][0] - x)
                    
                    C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T + (1 - (1 - c_sigma) ** 2) * np.eye(dim)) + \
                        c_mu * (y_w @ y_w.T)
                        
                    # Ensure symmetry
                    C = (C + C.T) / 2
                    
                    # Update paths
                    p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) / sigma * (new_mean - x)
                    p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) / sigma * (new_mean - x)
                    
                    # Update sigma
                    sigma *= np.exp(c_sigma / 5 * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) - 1))
                    
                    # Regularize C
                    try:
                        C_cholesky = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = C + 1e-10 * np.eye(dim)
                        C_cholesky = np.linalg.cholesky(C)
                    
                    x = new_mean
                    
                    if sigma < 1e-5:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
