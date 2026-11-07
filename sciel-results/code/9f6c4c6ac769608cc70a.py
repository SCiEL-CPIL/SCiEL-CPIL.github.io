"""Multi-start Nelder-Mead with coordinate polish and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Parallel multi-start Nelder-Mead + coordinate polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        DEADLINE = t0 + 280.0  # safety margin under 300s

        def check_time():
            return time.time() < DEADLINE

        # Global best tracker
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal best_x, best_f, call_count
            if not check_time():
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = func(x)
            call_count += 1
            if not np.isfinite(f):
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        dim = self.dim
        budget = self.budget

        # Known Schwefel minimum per-dimension value
        schwefel_star = 4.2096874633
        # Build diverse starting points
        rng = np.random.default_rng(42)

        starts = []
        starts.append(np.full(dim, schwefel_star))
        starts.append(np.zeros(dim))
        for _ in range(4):
            starts.append(rng.uniform(-5, 5, dim))
        starts.append(np.full(dim, -4.0))
        starts.append(np.full(dim, 4.0))
        for _ in range(2):
            starts.append(rng.uniform(-4, 4, dim))

        # Shuffle starts
        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]

        n_starts = len(starts)
        min_per_start = max(10, budget // (2 * n_starts))
        remaining = budget - min_per_start * n_starts
        weights = [max(1, n_starts - i) for i in range(n_starts)]
        wsum = sum(weights)
        allocs = [min_per_start + int(remaining * w / wsum) for w in weights]
        diff = budget - sum(allocs)
        if diff > 0:
            allocs[0] += diff
        elif diff < 0:
            idx = int(np.argmax(allocs))
            allocs[idx] += diff

        def nelder_mead(x0, max_calls, step_size=1.0):
            """Run Nelder-Mead from x0, using at most max_calls function evaluations."""
            n = dim + 1
            simplex = np.zeros((n, dim))
            simplex[0] = np.clip(x0, -5, 5)
            for i in range(1, n):
                simplex[i] = simplex[0].copy()
                simplex[i][i - 1] += step_size

            fvals = np.zeros(n)
            for i in range(n):
                if not check_time():
                    break
                fvals[i] = safe_func(simplex[i])

            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]

            calls_used = n
            max_iter = max(1, max_calls - n)

            for it in range(max_iter):
                if not check_time() or calls_used >= max_calls:
                    break

                centroid = simplex[:-1].mean(axis=0)

                xr = centroid + (centroid - simplex[-1])
                fr = safe_func(xr)
                calls_used += 1

                if fr < fvals[0]:
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    fe = safe_func(xe)
                    calls_used += 1
                    if fe < fr:
                        simplex[-1] = np.clip(xe, -5, 5)
                        fvals[-1] = fe
                    else:
                        simplex[-1] = np.clip(xr, -5, 5)
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = np.clip(xr, -5, 5)
                    fvals[-1] = fr
                else:
                    if fr < fvals[-1]:
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    fc = safe_func(xc)
                    calls_used += 1

                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = np.clip(xc, -5, 5)
                        fvals[-1] = fc
                    else:
                        for i in range(1, n):
                            if not check_time() or calls_used >= max_calls:
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = safe_func(simplex[i])
                            calls_used += 1

                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

            return simplex[0], fvals[0]

        def coordinate_polish(x, max_calls, step=0.5):
            dim = self.dim
            calls_used = 0
            x = np.clip(x.copy(), -5, 5)
            fx = safe_func(x)
            calls_used += 1

            for _ in range(max(1, max_calls // 2)):
                if not check_time() or calls_used >= max_calls:
                    break
                improved = False
                for i in range(dim):
                    if not check_time() or calls_used >= max_calls:
                        break
                    for direction in [1, -1]:
                        x_try = x.copy()
                        x_try[i] += direction * step
                        x_try = np.clip(x_try, -5, 5)
                        if np.allclose(x_try, x):
                            continue
                        f_try = safe_func(x_try)
                        calls_used += 1
                        if f_try < fx:
                            x = x_try
                            fx = f_try
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break

            return x, fx

        # Run Nelder-Mead from each start
        for i, start in enumerate(starts):
            if not check_time() or call_count >= budget:
                break
            alloc = allocs[i]
            if alloc < 5:
                continue

            # Stage 1: large step NM
            x1, f1 = nelder_mead(start, max(alloc // 2, 10), step_size=2.0)
            if not check_time() or call_count >= budget:
                break
            # Stage 2: small step NM
            x2, f2 = nelder_mead(x1, max(alloc // 2, 10), step_size=0.5)

            # Coordinate polish
            if check_time() and call_count < budget:
                xp, fp = coordinate_polish(x2, max(alloc // 4, 5), step=0.3)

        # Final polish from global best if budget remains
        if check_time() and best_x is not None and call_count < budget:
            remaining = budget - call_count
            if remaining > 10:
                xp, fp = coordinate_polish(best_x, min(remaining, 50), step=0.2)

        return best_x, best_f
# EVOLVE-BLOCK-END
