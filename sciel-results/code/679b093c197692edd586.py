import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = None

    def __call__(self, func):
        start_time = time.time()
        self.deadline = start_time + 290  # Safety margin

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            f = float(func(x_clipped))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        def one_plus_one_es(x0, f0, max_calls):
            nonlocal best_x, best_f, calls
            x = x0.copy()
            f = f0
            sigma = 0.8
            step_scale = 1.0
            gen = 0

            while calls < budget and time.time() < self.deadline:
                # Generate candidate
                y = rng.standard_normal(dim)
                y /= (np.linalg.norm(y) + 1e-15)
                x_trial = x + sigma * step_scale * y
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = eval_func(x_trial)

                if f_trial < f:
                    f = f_trial
                    x = x_trial
                    # Increase step size slightly
                    sigma *= 1.02
                    gen = 0
                else:
                    gen += 1
                    # Decrease step size
                    sigma *= 0.85
                    # Adapt step_scale based on Rosenbrock valley structure
                    # In Rosenbrock, the valley is along y = x^2 - 1
                    # The gradient in the valley is roughly aligned with the direction
                    # where the second derivative is large
                    # For a point in the valley, the narrow direction is roughly
                    # perpendicular to the valley axis.
                    # A simple heuristic: if we're in the valley region (small f),
                    # make steps smaller to avoid escaping
                    # But since we don't know f_opt, we use relative improvement
                    # If we've had many failures, shrink more
                    if gen > 20:
                        step_scale *= 0.95
                    elif gen > 10:
                        step_scale *= 0.98

                # Clamp sigma
                sigma = np.clip(sigma, 1e-10, 10.0)
                step_scale = np.clip(step_scale, 0.1, 5.0)

        # Phase 1: Interleaved (1+1)-ES from 4 diverse seeds
        # Allocate ~60% of budget to this phase
        phase1_budget = int(budget * 0.6)
        calls_per_es = phase1_budget // 4

        # Diverse starting points
        starts = [
            np.zeros(dim),
            np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
            np.array([-3.0, -3.0, -3.0, -3.0, -3.0]),
            rng.uniform(-4, 4, dim),
        ]

        # Run interleaved ES
        es_states = []
        for i, s in enumerate(starts):
            s = np.clip(s, -5, 5)
            f_s = eval_func(s)
            es_states.append((s, f_s, 0.8, 1.0, 0))

        # Interleave: one step from each ES in round-robin
        total_es_calls = 0
        max_rounds = calls_per_es

        for round_ in range(max_rounds):
            for idx in range(4):
                if calls >= phase1_budget:
                    break
                if time.time() >= self.deadline:
                    break

                x, f, sigma, step_scale, gen = es_states[idx]

                # Generate candidate
                y = rng.standard_normal(dim)
                y /= (np.linalg.norm(y) + 1e-15)
                x_trial = x + sigma * step_scale * y
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = eval_func(x_trial)

                if f_trial < f:
                    f = f_trial
                    x = x_trial
                    sigma *= 1.02
                    gen = 0
                else:
                    gen += 1
                    sigma *= 0.85
                    if gen > 20:
                        step_scale *= 0.95
                    elif gen > 10:
                        step_scale *= 0.98

                sigma = np.clip(sigma, 1e-10, 10.0)
                step_scale = np.clip(step_scale, 0.1, 5.0)
                es_states[idx] = (x, f, sigma, step_scale, gen)

        # Phase 2: Nelder-Mead polish from the best point
        if calls < budget and time.time() < self.deadline:
            x0 = best_x.copy() if best_x is not None else np.zeros(dim)
            f0 = best_f
            n = dim

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]

            # Initialize simplex points
            for i in range(n):
                x_i = x0.copy()
                step = 0.5 if abs(x0[i]) < 1e-6 else 0.5 * np.sign(x0[i])
                x_i[i] += step
                x_i = np.clip(x_i, -5, 5)
                f_i = eval_func(x_i)
                simplex.append(x_i)
                fvals.append(f_i)

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            # Main NM loop
            while calls < budget and time.time() < self.deadline:
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check for convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = eval_func(x_r)

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = eval_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = eval_func(x_c)

                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = eval_func(simplex[i])

        # Phase 3: Final (1+1)-ES polish from best point if budget remains
        if calls < budget and time.time() < self.deadline:
            one_plus_one_es(best_x.copy() if best_x is not None else np.zeros(dim), best_f, budget - calls)

        return best_x, best_f
