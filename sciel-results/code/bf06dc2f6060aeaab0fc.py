"""Rewritten optimizer using Rank-1 + Rank-mu natural gradient descent with diagonal step-size."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Rank-1 + Rank-mu Natural Gradient Descent with diagonal step-size vector.
    
    Models the search distribution as an axis-aligned ellipsoid (diagonal covariance)
    and updates it via natural gradient descent on the log-diagonal parameters.
    This avoids full covariance matrix overhead while adapting per-dimension scales
    to rapidly flatten anisotropic landscapes like the Discus function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        B = self.budget

        # Initial point in the center of the box
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Diagonal step-size (sigma per dimension), start with a moderate value
        sigma = np.full(d, 1.5)

        # Population size
        mu = 10  # number of offspring per generation
        # Learning rates for natural gradient updates
        lr_rank1 = 0.25 / (mu * d)      # rank-1 update rate
        lr_rankmu = 0.1 / (mu * d)      # rank-mu update rate

        # Counter for successive improvements (for adaptation)
        count_improve = 0
        max_count = 100

        calls_used = 1  # first evaluation

        for gen in range(1, B - mu + 1):
            if calls_used >= B:
                break

            # Generate mu offspring
            offspring = []
            offspring_f = []
            for i in range(mu):
                if calls_used >= B:
                    break
                # Sample from N(x, diag(sigma^2))
                z = rng.standard_normal(d)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                offspring.append(y)
                offspring_f.append(fy)

            if len(offspring) == 0:
                break

            # Sort by fitness
            indices = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_f = [offspring_f[i] for i in indices]

            # Update best
            if sorted_f[0] < best_fx:
                best_fx = sorted_f[0]
                best_x = sorted_offspring[0].copy()

            # Natural gradient update for log(sigma)
            # Let s = log(sigma), update s via natural gradient
            s = np.log(sigma)

            # Rank-1 update: direction towards best offspring
            # y_best = sorted_offspring[0]
            # Direction in original space: (y_best - x) / sigma^2 (weighted)
            y_best = np.array(sorted_offspring[0])
            # In the natural gradient framework for diagonal covariance:
            # The rank-1 update to log(sigma) is proportional to (y_best - x) / sigma
            # More precisely, for a diagonal Gaussian, the natural gradient of 
            # log N(y; x, diag(sigma^2)) w.r.t. log(sigma) involves (y-x)/sigma
            # We use a simplified but effective update:
            
            # Rank-1: move sigma towards the best offspring direction
            # If y_best is in direction d from x, increase sigma in that direction
            delta_r1 = (y_best - x) / (sigma + 1e-10)
            s += lr_rank1 * delta_r1

            # Rank-mu update: average of top mu/2 (or all mu) offspring
            # Use weighted average of top half
            n_top = max(1, mu // 2)
            top_offspring = np.array(sorted_offspring[:n_top])
            # Weights: linearly decreasing
            weights = np.arange(n_top, 0, -1, dtype=float)
            weights /= weights.sum()
            
            # Average direction from x
            avg_y = np.average(top_offspring, axis=0, weights=weights[:, None])
            delta_rm = (avg_y - x) / (sigma + 1e-10)
            s += lr_rankmu * delta_rm

            # Clamp sigma to reasonable bounds
            sigma = np.exp(np.clip(s, np.log(0.01), np.log(10.0)))

            # Update mean x: move towards best offspring
            # Use a step size that shrinks over time
            step_size = 1.0 / (1 + gen * 0.01)
            x = x + step_size * (y_best - x)
            x = np.clip(x, -5, 5)

            # Track improvement for early stopping or adaptation
            if sorted_f[0] < fx:
                fx = sorted_f[0]
                count_improve = 0
            else:
                count_improve += 1

            # If no improvement for a long time, reduce sigma globally
            if count_improve > max_count:
                sigma *= 0.7
                count_improve = 0

        # Final evaluation at best_x if we have budget left
        if calls_used < B:
            # Do a few local refinements around best_x
            local_sigma = sigma * 0.3
            for _ in range(B - calls_used):
                if calls_used >= B:
                    break
                z = rng.standard_normal(d)
                y = best_x + local_sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                    local_sigma *= 0.9
                else:
                    local_sigma *= 0.95

        return best_x, best_fx
# EVOLVE-BLOCK-END
