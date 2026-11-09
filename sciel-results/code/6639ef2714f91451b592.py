import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._start_time = None
        self._deadline = None
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        self._start_time = time.time()
        self._deadline = self._start_time + 320
        self._call_count = 0
        self.budget_remaining = self.budget
        self._best_x = np.zeros(self.dim)
        self._best_f = np.inf
        self._rng = np.random.default_rng(42)

        def safe_call(x):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                return 1e10
            self._call_count += 1
            self.budget_remaining -= 1
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # Generate Sobol sequence for global exploration points
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            sobol_points = sampler.random(500)
            sobol_points = sobol_points * 10.0 - 5.0
        except ImportError:
            sobol_points = self._rng.uniform(-5, 5, (500, self.dim))

        sobol_idx = 0
        total_sobol = len(sobol_points)

        # Phase 1: Quick global exploration with 15 Sobol points
        for i in range(15):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            x = sobol_points[sobol_idx]
            sobol_idx += 1
            safe_call(x)

        # Phase 2: Interleaved (1+1)-ES with periodic global jumps
        # Strategy: Run (1+1)-ES, but every K steps without improvement,
        # jump to a new Sobol point to escape local minima
        sigma = 0.8
        sigma_min = 1e-7
        sigma_max = 4.0
        stagnation_limit = 20
        stagnation_counter = 0
        jump_interval = 25  # Check for jump every 25 ES steps

        es_step = 0
        while self._call_count < self.budget and time.time() < self._deadline:
            # Periodic global jump to handle multimodality
            if es_step > 0 and es_step % jump_interval == 0 and sobol_idx < total_sobol:
                new_x = sobol_points[sobol_idx]
                sobol_idx += 1
                f_new = safe_call(new_x)
                if f_new < self._best_f:
                    # Found a better basin, reset ES
                    sigma = 0.8
                    stagnation_counter = 0
                    es_step = 0
                else:
                    # Keep current best, continue ES
                    pass
                es_step += 1
                continue

            # (1+1)-ES step
            noise = self._rng.standard_normal(self.dim)
            candidate_x = self._best_x + sigma * noise
            candidate_x = np.clip(candidate_x, -5.0, 5.0)

            f_cand = safe_call(candidate_x)
            es_step += 1

            if f_cand < self._best_f:
                # Accept: move to candidate
                self._best_f = f_cand
                self._best_x = candidate_x
                stagnation_counter = 0
                # Slowly increase sigma to explore more
                sigma = min(sigma * 1.03, sigma_max)
            else:
                # Reject: stay at current best
                stagnation_counter += 1
                # Decrease sigma to refine
                sigma = max(sigma * 0.95, sigma_min)

                # If heavily stagnated, do a larger jump
                if stagnation_counter >= stagnation_limit:
                    if sobol_idx < total_sobol:
                        jump_x = sobol_points[sobol_idx]
                        sobol_idx += 1
                    else:
                        jump_x = self._rng.uniform(-5, 5, self.dim)
                    f_jump = safe_call(jump_x)
                    if f_jump < self._best_f:
                        self._best_f = f_jump
                        self._best_x = jump_x
                        sigma = 1.0
                        stagnation_counter = 0
                        es_step = 0
                    else:
                        # Try a random restart with larger sigma
                        sigma = min(sigma * 3.0, sigma_max)
                        stagnation_counter = 0

        return self._best_x, self._best_f
