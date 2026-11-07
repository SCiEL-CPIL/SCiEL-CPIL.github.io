"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive step size for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls = 0

        def check_time():
            return time.time() > deadline

        # Use a fixed seed for reproducibility across instances
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Multi-start: try several initial points with different strategies
        # Rastrigin has many local minima, so we need good global exploration
        # Strategy: combine random restarts with a CMA-ES-like local search

        # First, do a quick random scan to find a good starting region
        # We'll allocate some budget for exploration and some for exploitation

        # Phase 1: Random exploration to find promising regions
        # Use a grid-like sampling combined with random points
        n_explore = min(50, budget // 10)
        for i in range(n_explore):
            if check_time() or calls >= budget:
                break
            # Mix of random and structured points
            if i % 3 == 0:
                x = rng.uniform(-5, 5, dim)
            elif i % 3 == 1:
                # Points near the origin (Rastrigin global min is often near 0 after shift)
                x = rng.uniform(-2, 2, dim)
            else:
                # Points at moderate distance
                x = rng.uniform(-3.5, 3.5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback: evaluate one point
            x = np.zeros(dim)
            best_f = func(x)
            calls += 1
            best_x = x.copy()

        # Phase 2: Local optimization from the best point using adaptive step size
        # Use a (1+lambda)-ES with adaptive step size
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # Start with a moderate step size

        # Use a population-based approach for better convergence
        # CMA-ES-like: maintain a population and adapt covariance
        # For simplicity and speed, use a simpler adaptive method

        # Adaptive (1+1)-ES with success rule
        lambda_ = 8  # Population size
        successes = 0
        window = 15  # Window for success rate

        while calls < budget:
            if check_time():
                break

            # Generate lambda_ offspring
            offspring = []
            for _ in range(lambda_):
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls += 1
                if calls >= budget or check_time():
                    break
                offspring.append((fy, y))

            if not offspring:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])
            best_offspring_f, best_offspring_x = offspring[0]

            # Update best
            if best_offspring_f < best_f:
                best_f = best_offspring_f
                best_x = best_offspring_x.copy()

            # Update current point and step size
            if best_offspring_f < fx:
                x = best_offspring_x
                fx = best_offspring_f
                successes += 1
            else:
                successes = 0

            # Adapt sigma based on success rate
            # 1/5 success rule: target success rate of 1/5
            # If success rate > 1/5, increase sigma; otherwise decrease
            recent_successes = min(successes, window)
            success_rate = recent_successes / window
            if success_rate > 0.2:
                sigma *= 1.2
            elif success_rate < 0.1:
                sigma *= 0.8
            else:
                sigma *= 0.95

            # Clamp sigma
            sigma = np.clip(sigma, 0.01, 5.0)

            # If we've been stuck for a while, try a restart
            # Check if we've made no progress in the last 50 evaluations
            if calls > 50:
                # Simple stagnation check: if best hasn't improved in last 30 calls
                # We'll track this more carefully
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
