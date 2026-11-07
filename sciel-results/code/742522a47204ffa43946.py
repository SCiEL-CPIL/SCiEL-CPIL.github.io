"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive simplex and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track all evaluations
        all_points = []
        all_values = []
        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            all_points.append(x.copy())
            all_values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial exploration: sample diverse points
        n_initial = 20
        for i in range(n_initial):
            if call_count >= budget:
                break
            # Mix of random and structured points
            if i == 0:
                x = np.zeros(dim)
            elif i == 1:
                x = rng.uniform(-5, 5, dim)
            elif i < 6:
                # Axis-aligned points
                x = np.zeros(dim)
                x[i - 2] = rng.choice([-3, -1, 1, 3])
            else:
                x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)

        # Sort initial points by value
        if len(all_points) > 0:
            idx = np.argsort(all_values)
            all_points = [all_points[i] for i in idx]
            all_values = [all_values[i] for i in idx]

        # Multi-start Nelder-Mead from best initial points
        n_starts = min(5, len(all_points))
        for s in range(n_starts):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            # Build initial simplex from best points
            simplex = [all_points[0].copy()]
            simplex_f = [all_values[0]]
            for j in range(1, min(s + 2, dim + 1, len(all_points))):
                simplex.append(all_points[j].copy())
                simplex_f.append(all_values[j])

            # If we don't have enough points, add small perturbations
            while len(simplex) < dim + 1:
                base = simplex[0]
                new_pt = base + 0.5 * rng.standard_normal(dim)
                new_pt = np.clip(new_pt, -5, 5)
                if new_pt not in simplex:
                    simplex.append(new_pt)
                    simplex_f.append(evaluate(new_pt))

            # Nelder-Mead
            alpha, gamma, rho = 1.0, 2.0, -0.5
            max_iter = min(200, budget - call_count)
            for it in range(max_iter):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = evaluate(x_r)

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = evaluate(x_c)
                    if f_c < worst_f:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = evaluate(simplex[i])

                # Check convergence
                if len(simplex_f) > 1 and abs(max(simplex_f) - min(simplex_f)) < 1e-10:
                    break

        # Final polish: local search around best point
        if best_x is not None and call_count < budget and time.time() < deadline:
            # Try small perturbations
            for step in [0.1, 0.01, 0.001]:
                if call_count >= budget or time.time() > deadline:
                    break
                for axis in range(dim):
                    if call_count >= budget or time.time() > deadline:
                        break
                    for direction in [-1, 1]:
                        if call_count >= budget or time.time() > deadline:
                            break
                        x_try = best_x.copy()
                        x_try[axis] += direction * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = evaluate(x_try)
                        if f_try < best_f:
                            best_x = x_try.copy()
                            best_f = f_try

        return best_x, best_f
# EVOLVE-BLOCK-END
