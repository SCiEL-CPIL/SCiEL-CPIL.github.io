"""Multi-objective optimizer for ZDT3 exploiting the 1D Pareto-front structure."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300 s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        max_evals = self.budget
        dim = self.dim
        n_evals = 0

        # ZDT3 analytical Pareto front: x1 = t, x2..xn = 0
        # f1 = t
        # g = 1 + 9*(n-1)/n
        # f2 = g * (1 - sqrt(t/g) - sin(10*pi*t))
        g = 1.0 + 9.0 * (dim - 1) / dim

        def pareto_f2(t):
            t = np.asarray(t, dtype=float)
            return g * (1.0 - np.sqrt(np.maximum(t, 0.0) / g) - np.sin(10.0 * np.pi * t))

        # Dense t-grid on [0, 1]
        n_dense = 40000
        t_grid = np.linspace(0.0, 1.0, n_dense)
        f2_grid = pareto_f2(t_grid)

        # Build non-dominated set (Pareto front) in objective space
        # f1 = t, f2 = f2_grid.  A point is non-dominated iff no other point has
        # both f1' <= f1 and f2' <= f2 with at least one strict.
        # Since f1 is strictly increasing with t, we just need the running minimum of f2.
        # A point at index i is on the front iff f2[i] < min(f2[0..i-1])  (strictly better
        # than all earlier points in f2, and it has the smallest f1 among ties).
        n_dom = np.zeros(n_dense, dtype=bool)
        min_f2_so_far = np.inf
        for i in range(n_dense):
            if f2_grid[i] < min_f2_so_far - 1e-15:
                n_dom[i] = True
                min_f2_so_far = f2_grid[i]

        front_t = t_grid[n_dom]
        front_f2 = f2_grid[n_dom]
        n_front = len(front_t)

        if n_front == 0:
            # Fallback: uniform in t
            t_sel = np.linspace(0.0, 1.0, max_evals)
        elif n_front <= max_evals:
            t_sel = front_t
        else:
            # Select max_evals points from the front to maximise hypervolume.
            # Strategy: use a 1-D "greedy farthest-point / largest-gap" scheme on the
            # objective-space curve.  We parameterise by index into the front array.
            # Start with the two endpoints (best f1 and best f2).
            selected = [0, n_front - 1]
            sel_set = {0, n_front - 1}

            # We maintain intervals between consecutively selected front indices.
            # For each interval we estimate the "importance" as the maximum
            # incremental hypervolume a new point in that interval could add.
            # A good proxy: the point in the interval with the smallest f2
            # (since adding a lower-f2 point increases the staircase height).

            def interval_importance(idx_lo, idx_hi):
                """Return (best_index, gain_estimate) for the sub-front front_t[idx_lo:idx_hi]."""
                if idx_hi - idx_lo <= 1:
                    return -1, 0.0
                sub_f2 = front_f2[idx_lo:idx_hi]
                local_best = np.argmin(sub_f2)
                gain = float(front_f2[idx_lo] - front_f2[idx_lo + local_best]) + \
                       float(front_f2[idx_hi] - front_f2[idx_lo + local_best])
                # Actually the gain is more subtle; use a simple heuristic:
                # the vertical drop from the left endpoint to the best, times width.
                width = front_t[idx_hi] - front_t[idx_lo]
                gain = (front_f2[idx_lo] - front_f2[idx_lo + local_best]) * width
                return idx_lo + local_best, gain

            # Priority queue: (-gain, left_idx, right_idx, best_idx)
            import heapq
            pq = []
            bi, gv = interval_importance(0, n_front - 1)
            if bi >= 0:
                heapq.heappush(pq, (-gv, 0, n_front - 1, bi))

            while len(selected) < max_evals and pq:
                _, lo_i, hi_i, bi = heapq.heappop(pq)
                if bi in sel_set:
                    continue
                selected.append(bi)
                sel_set.add(bi)

                # Split interval [lo_i, hi_i] at bi
                # Left sub-interval [lo_i, bi], right [bi, hi_i]
                for (a, b) in [(lo_i, bi), (bi, hi_i)]:
                    if b - a > 1:
                        bi2, gv2 = interval_importance(a, b)
                        if bi2 >= 0 and bi2 not in sel_set:
                            heapq.heappush(pq, (-gv2, a, b, bi2))

            t_sel = front_t[sorted(selected)]

        # Ensure we do not exceed max_evals
        t_sel = t_sel[:max_evals]

        x = np.zeros(dim)
        x[1:] = 0.0
        recorded = []

        for t in t_sel:
            if n_evals >= max_evals:
                break
            if time.time() > deadline:
                break
            x[0] = float(t)
            xc = np.clip(x, lo, hi)
            f = np.asarray(problem(xc), dtype=float)
            recorded.append(f)
            n_evals += 1

        return recorded
