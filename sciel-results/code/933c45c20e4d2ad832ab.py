"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Parallel portfolio of adaptive (1+λ)-ES restarts for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill

        # Global best tracking
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        # Portfolio: multiple independent ES runs
        n_restarts = 8
        lambda_ = 4  # offspring per generation per ES
        rng = np.random.default_rng(12345)

        # Initialize multiple starting points spread across the box
        start_points = []
        for i in range(n_restarts):
            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = np.array([2.0] * dim)
            elif i == 2:
                x0 = np.array([-2.0] * dim)
            else:
                x0 = rng.uniform(-4, 4, dim)
            start_points.append(x0)

        # Evaluate initial points
        for x0 in start_points:
            if calls_made >= budget or time.time() > deadline:
                break
            fx = func(np.clip(x0, -5, 5))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        # Each ES maintains its own state
        es_states = []
        for i, x0 in enumerate(start_points):
            if i >= len(start_points):
                break
            # Initialize ES around its start point
            sigma = 1.0
            x_es = x0.copy()
            f_es = best_f if i == 0 else func(np.clip(x0, -5, 5))
            if i > 0:
                calls_made += 1
            es_states.append({
                'x': x_es.copy(),
                'f': f_es,
                'sigma': sigma,
                'seed': 1000 + i
            })

        # Main loop: interleave steps across all ES instances
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Select which ES to step next (round-robin with adaptation)
            # Prioritize ES that have made recent progress
            for state in es_states:
                if calls_made >= budget or time.time() > deadline:
                    break

                x = state['x']
                sigma = state['sigma']
                f_current = state['f']

                # Generate lambda_ offspring
                offspring = []
                offspring_f = []
                for j in range(lambda_):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    # Adaptive step size based on success rate (simple heuristic)
                    step = sigma * rng.normal(0, 1, dim)
                    x_new = x + step
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls_made += 1
                    offspring.append(x_new)
                    offspring_f.append(f_new)

                    # Update global best
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()

                if not offspring:
                    continue

                # Select best offspring
                best_off_idx = np.argmin(offspring_f)
                x_best_off = offspring[best_off_idx]
                f_best_off = offspring_f[best_off_idx]

                # Update ES state
                if f_best_off < f_current:
                    # Success: accept and adapt sigma up
                    state['x'] = x_best_off
                    state['f'] = f_best_off
                    state['sigma'] = min(sigma * 1.05, 5.0)
                else:
                    # Failure: keep current, adapt sigma down
                    state['sigma'] = max(sigma * 0.9, 0.01)

                # Occasionally re-seed from global best if stuck
                if state['sigma'] < 0.1 and f_current > best_f + 1.0:
                    state['x'] = best_x + rng.normal(0, 0.5, dim)
                    state['x'] = np.clip(state['x'], -5, 5)
                    state['sigma'] = 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
