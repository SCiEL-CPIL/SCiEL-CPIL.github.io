import time
import numpy as np


def construct():
    start = time.time()
    DEADLINE = start + 138.0  # safety margin before 150s kill

    def C2(x):
        return x * (x - 1) // 2

    def init(n, rng):
        color = rng.integers(0, 5, size=(n, n))
        color = np.maximum(color, color.T)
        np.fill_diagonal(color, -1)
        A = []
        common = []
        for c in range(5):
            Ac = (color == c)
            np.fill_diagonal(Ac, False)
            A.append(Ac)
            cc = (Ac @ Ac.T).astype(np.int64)
            np.fill_diagonal(cc, 0)
            common.append(cc)
        total = 0
        iu = np.triu_indices(n, k=1)
        for c in range(5):
            total += int(C2(common[c][iu]).sum())
        return color, A, common, total

    def flip(color, A, common, u, v, c_new):
        c_old = int(color[u, v])
        if c_old == c_new:
            return 0
        n = color.shape[0]
        b_idx = np.delete(np.arange(n), [u, v])
        delta_total = 0
        for c, sign in ((c_old, -1), (c_new, 1)):
            Ac = A[c]
            cc = common[c]
            old_u = cc[u, b_idx].copy()
            new_u = old_u + sign * Ac[b_idx, v]
            delta_total += int((C2(new_u) - C2(old_u)).sum())
            cc[u, b_idx] = new_u
            cc[b_idx, u] = new_u
            old_v = cc[v, b_idx].copy()
            new_v = old_v + sign * Ac[b_idx, u]
            delta_total += int((C2(new_v) - C2(old_v)).sum())
            cc[v, b_idx] = new_v
            cc[b_idx, v] = new_v
            Ac[u, v] = (sign == 1)
            Ac[v, u] = (sign == 1)
        color[u, v] = c_new
        color[v, u] = c_new
        return delta_total

    def local_search(n, rng, time_budget):
        t0 = time.time()
        color, A, common, total = init(n, rng)
        best_total = total
        best_color = color.copy()
        no_improve = 0
        restarts = 0
        while time.time() - t0 < time_budget:
            u = int(rng.integers(0, n))
            v = int(rng.integers(0, n))
            if u == v:
                continue
            c_new = int(rng.integers(0, 5))
            c_old = int(color[u, v])
            old_total = total
            d = flip(color, A, common, u, v, c_new)
            total = old_total + d
            if total < best_total:
                best_total = total
                best_color = color.copy()
                no_improve = 0
                if total == 0:
                    break
            else:
                # annealing: occasionally accept worsening to escape local minima
                if d > 0 and rng.random() < 0.03:
                    no_improve += 1
                else:
                    flip(color, A, common, u, v, c_old)  # revert
                    total = old_total
                    no_improve += 1
            if no_improve > 40000:
                no_improve = 0
                restarts += 1
                if restarts > 200 or time.time() - t0 > time_budget:
                    break
                color, A, common, total = init(n, rng)
                if total < best_total:
                    best_total = total
                    best_color = color.copy()
                if best_total == 0:
                    break
        if best_total == 0:
            return best_color
        return None

    best_n = 0
    best_colors = None
    rng = np.random.default_rng(12345)

    # Try known-feasible 26 first to guarantee a strong score, then push to 27/28.
    targets = [26, 27, 28]
    for n in targets:
        remaining = DEADLINE - time.time()
        if remaining < 8.0:
            break
        budget = min(remaining * 0.6, 55.0)
        if budget < 6.0:
            budget = remaining * 0.5
        col = local_search(n, rng, budget)
        if col is not None:
            best_n = n
            best_colors = col
            # keep trying larger n with leftover time
            continue
        else:
            # not solved; try smaller feasible targets already covered (26) -> stop pushing
            break

    if best_colors is None:
        # Fallback: guaranteed valid small construction (K11 circular).
        n = 11
        best_colors = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(i + 1, n):
                best_colors[i, j] = min(j - i, n - j + i) - 1
        best_colors = np.maximum(best_colors, best_colors.T)
        best_n = n

    colors = [int(best_colors[i, j]) for i in range(best_n) for j in range(i + 1, best_n)]
    return {'n': best_n, 'colors': colors}
