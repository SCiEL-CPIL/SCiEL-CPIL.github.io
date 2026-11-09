"""Rewritten optimizer for BBOB-style continuous optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based (1+lambda)-ES with adaptive step size,
    structured initialization, and coordinate refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        deadline = time.time() + 280.0  # safety margin before 300s limit

        # --- Structured initialization: sample a few diverse starting points ---
        # Include origin, random points, and points near boundaries
        init_candidates = []
        init_candidates.append(np.zeros(self.dim))
        init_candidates.append(rng.uniform(-5, 5, self.dim))
        init_candidates.append(rng.uniform(-5, 5, self.dim))
        init_candidates.append(np.full(self.dim, 0.5) * rng.choice([-1, 1], self.dim))
        init_candidates.append(rng.uniform(-3, 3, self.dim))

        # Evaluate all init candidates
        best_x = None
        best_f = np.inf
        for xc in init_candidates:
            xc = np.clip(xc, -5, 5)
            fc = func(xc)
            if fc < best_f:
                best_f = fc
                best_x = xc.copy()

        calls_used = len(init_candidates)
        remaining = self.budget - calls_used
        if remaining <= 0:
            return best_x, best_f

        # --- Main search: (1+lambda)-ES with adaptive sigma ---
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        lam = 4  # population size for offspring
        success_count = 0
        total_trials = 0

        # Phase 1: Population-based ES
        while calls_used < self.budget and time.time() < deadline:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Generate lam offspring
            offspring = []
            for _ in range(lam):
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                offspring.append((y, fy))
                if calls_used >= self.budget:
                    break

            # Select best offspring
            best_off_x, best_off_f = min(offspring, key=lambda t: t[1])

            total_trials += len(offspring)
            if best_off_f < fx:
                x = best_off_x
                fx = best_off_f
                success_count += 1
            else:
                # Keep x, fx unchanged
                pass

            # Adapt sigma based on success rate (1/5 rule variant)
            if total_trials > 10:
                rate = success_count / total_trials
                if rate > 0.2:
                    sigma *= 1.3
                elif rate < 0.1:
                    sigma *= 0.7
                else:
                    sigma *= 0.95
                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 10.0)
                # Reset counters periodically
                if total_trials > 50:
                    success_count = 0
                    total_trials = 0

        # --- Phase 2: Coordinate descent refinement ---
        # Use remaining budget for local refinement
        remaining = self.budget - calls_used
        if remaining > 0 and time.time() < deadline:
            # Coordinate descent with line search
            step = max(sigma, 0.1)
            for _ in range(min(remaining, 100)):
                if calls_used >= self.budget or time.time() > deadline - 2.0:
                    break
                improved = False
                for d in range(self.dim):
                    if calls_used >= self.budget or time.time() > deadline - 2.0:
                        break
                    # Try positive and negative steps
                    for direction in [1, -1]:
                        if calls_used >= self.budget:
                            break
                        x_trial = x.copy()
                        x_trial[d] = np.clip(x_trial[d] + direction * step, -5, 5)
                        f_trial = func(x_trial)
                        calls_used += 1
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            improved = True
                            break
                if not improved:
                    step *= 0.7
                    if step < 0.001:
                        break
                else:
                    step *= 0.9

        return x, fx
# EVOLVE-BLOCK-END
