"""Portfolio-based optimizer for BBOB f3 (Rastrigin) with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-restart adaptive optimizer for Rastrigin-like functions.
    Uses a combination of:
      1. Random restarts with local refinement
      2. (1+1)-ES with success rule for local search
      3. Coarse-to-fine grid sampling around promising points
      4. Time-bounded to stay within 330s wall-clock limit
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill

        def check_deadline():
            return time.time() > deadline

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial random sampling to find good basins
        # Rastrigin has many local minima; we need good initial points
        n_initial = min(50, budget // 10)
        for _ in range(n_initial):
            if check_deadline() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            safe_eval(x)

        # Phase 2: Local refinement from best point and nearby points
        # Use (1+1)-ES with adaptive sigma
        if best_x is not None and calls_made < budget and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            successes = 0
            trials = 0
            min_sigma = 1e-4
            max_sigma = 5.0

            while calls_made < budget and not check_deadline():
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_eval(y)
                trials += 1

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    sigma *= 1.2
                    sigma = min(sigma, max_sigma)
                else:
                    sigma *= 0.9
                    sigma = max(sigma, min_sigma)

                # Adapt sigma based on success rate over last 20 trials
                if trials % 20 == 0:
                    success_rate = successes / 20.0
                    if success_rate > 0.3:
                        sigma *= 1.3
                    elif success_rate < 0.1:
                        sigma *= 0.7
                    sigma = np.clip(sigma, min_sigma, max_sigma)
                    successes = 0

                # If we've converged (sigma very small), restart locally
                if sigma < min_sigma * 2:
                    break

        # Phase 3: Multiple restarts with local refinement
        n_restarts = 15
        for r in range(n_restarts):
            if check_deadline() or calls_made >= budget:
                break

            # Choose restart point: mix of random and perturbed best
            if r % 3 == 0:
                # Pure random restart
                x0 = rng.uniform(-5, 5, dim)
            elif r % 3 == 1:
                # Perturb current best
                x0 = best_x + rng.uniform(-2, 2, dim)
                x0 = np.clip(x0, -5, 5)
            else:
                # Perturb current best with smaller scale
                x0 = best_x + rng.uniform(-1, 1, dim)
                x0 = np.clip(x0, -5, 5)

            fx0 = safe_eval(x0)

            # Local (1+1)-ES refinement
            x = x0.copy()
            fx = fx0
            sigma = 0.8
            min_sigma = 1e-5
            max_sigma = 3.0
            successes = 0
            trials = 0

            # Limit local search budget per restart
            local_budget = min(30, budget - calls_made)
            for _ in range(local_budget):
                if check_deadline() or calls_made >= budget:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_eval(y)
                trials += 1

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    sigma *= 1.25
                    sigma = min(sigma, max_sigma)
                else:
                    sigma *= 0.85
                    sigma = max(sigma, min_sigma)

                if trials % 15 == 0:
                    success_rate = successes / 15.0
                    if success_rate > 0.35:
                        sigma *= 1.2
                    elif success_rate < 0.1:
                        sigma *= 0.75
                    sigma = np.clip(sigma, min_sigma, max_sigma)
                    successes = 0

                if sigma < min_sigma * 2:
                    break

        # Phase 4: Fine-tuning around the best solution found
        if best_x is not None and calls_made < budget and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            min_sigma = 1e-6
            max_sigma = 0.5
            successes = 0
            trials = 0

            fine_budget = min(50, budget - calls_made)
            for _ in range(fine_budget):
                if check_deadline() or calls_made >= budget:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_eval(y)
                trials += 1

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    sigma *= 1.15
                    sigma = min(sigma, max_sigma)
                else:
                    sigma *= 0.9
                    sigma = max(sigma, min_sigma)

                if trials % 10 == 0:
                    success_rate = successes / 10.0
                    if success_rate > 0.4:
                        sigma *= 1.1
                    elif success_rate < 0.1:
                        sigma *= 0.8
                    sigma = np.clip(sigma, min_sigma, max_sigma)
                    successes = 0

                if sigma < min_sigma * 2:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
