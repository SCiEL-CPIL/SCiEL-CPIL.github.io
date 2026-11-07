"""Combined (1+1)-ES with Hooke-Jeeves refinement for BBOB f12 (Bent Cigar)."""
import numpy as np
import time


class Optimizer:
    """(1+1)-ES for global exploration, Hooke-Jeeves for local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls_made = 1

        # Phase 1: (1+1)-ES for global exploration
        sigma = 2.5
        min_sigma = 1e-5
        max_sigma = 8.0
        no_improve = 0
        es_budget = int(budget * 0.5)  # Use half the budget for ES

        for _ in range(es_budget - 1):
            if time.time() > deadline or calls_made >= budget:
                break
            z = rng.standard_normal(dim)
            y = np.clip(x + sigma * z, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
                sigma = min(sigma * 1.15, max_sigma)
                no_improve = 0
            else:
                sigma = max(sigma * 0.92, min_sigma)
                no_improve += 1
                if no_improve >= 30:
                    sigma = min(sigma * 2.0, max_sigma)
                    no_improve = 0

        # Phase 2: Hooke-Jeeves for local refinement
        remaining = budget - calls_made
        if remaining > 0 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            step_sizes = np.ones(dim) * 0.5
            min_step = 1e-8
            max_step = 10.0

            hj_calls = 0
            while hj_calls < remaining and time.time() < deadline:
                progress = False
                for i in range(dim):
                    if hj_calls >= remaining or time.time() > deadline:
                        break

                    # Try positive direction
                    trial = x.copy()
                    trial[i] += step_sizes[i]
                    trial = np.clip(trial, -5, 5)
                    trial_f = func(trial)
                    calls_made += 1
                    hj_calls += 1

                    if trial_f < fx:
                        x, fx = trial, trial_f
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()
                        progress = True
                        step_sizes[i] = min(step_sizes[i] * 1.5, max_step)
                        # Continue in same direction
                        while hj_calls < remaining and time.time() < deadline:
                            trial = x.copy()
                            trial[i] += step_sizes[i]
                            trial = np.clip(trial, -5, 5)
                            trial_f = func(trial)
                            calls_made += 1
                            hj_calls += 1
                            if trial_f < fx:
                                x, fx = trial, trial_f
                                if fx < best_f:
                                    best_f, best_x = fx, x.copy()
                            else:
                                break
                    else:
                        step_sizes[i] *= 0.5

                        # Try negative direction
                        trial = x.copy()
                        trial[i] -= step_sizes[i]
                        trial = np.clip(trial, -5, 5)
                        trial_f = func(trial)
                        calls_made += 1
                        hj_calls += 1

                        if trial_f < fx:
                            x, fx = trial, trial_f
                            if fx < best_f:
                                best_f, best_x = fx, x.copy()
                            progress = True
                            step_sizes[i] = min(step_sizes[i] * 1.5, max_step)
                            # Continue in same direction
                            while hj_calls < remaining and time.time() < deadline:
                                trial = x.copy()
                                trial[i] -= step_sizes[i]
                                trial = np.clip(trial, -5, 5)
                                trial_f = func(trial)
                                calls_made += 1
                                hj_calls += 1
                                if trial_f < fx:
                                    x, fx = trial, trial_f
                                    if fx < best_f:
                                        best_f, best_x = fx, x.copy()
                                else:
                                    break
                        else:
                            step_sizes[i] = max(step_sizes[i] * 0.5, min_step)

                # If no progress, check convergence
                if not progress:
                    if np.all(step_sizes <= min_step * 1.1):
                        break
                    # Try a larger random perturbation
                    if hj_calls < remaining and time.time() < deadline:
                        trial = x + rng.uniform(-1, 1, dim)
                        trial = np.clip(trial, -5, 5)
                        trial_f = func(trial)
                        calls_made += 1
                        hj_calls += 1
                        if trial_f < fx:
                            x, fx = trial, trial_f
                            if fx < best_f:
                                best_f, best_x = fx, x.copy()
                            step_sizes = np.ones(dim) * 0.5
                        else:
                            step_sizes = np.maximum(step_sizes * 0.5, min_step)

        return best_x, best_f
