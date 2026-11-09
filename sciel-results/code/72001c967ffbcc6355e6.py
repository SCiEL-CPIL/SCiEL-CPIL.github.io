"""Multi-objective optimizer on RE21 (4 vars, 2 obj) combining surrogate-guided search with evolutionary exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _hv2(pts, ref):
    """Exact 2D hypervolume of points strictly below ref."""
    if pts is None or len(pts) == 0:
        return 0.0
    p = np.asarray(pts, dtype=np.float64)
    p = p[(p[:, 0] < ref[0]) & (p[:, 1] < ref[1])]
    if len(p) == 0:
        return 0.0
    order = np.lexsort((p[:, 1], p[:, 0]))
    p = p[order]
    f1 = p[:, 0]
    f2 = p[:, 1]
    x_right = np.concatenate([f1[1:], [ref[0]]])
    y_top = np.concatenate([f2, [ref[1]]])
    y_bottom = np.concatenate([[ref[1]], f2[:-1]])
    h = np.maximum(0.0, y_top - y_bottom)
    w = np.maximum(0.0, x_right - f1)
    return float(np.sum(h * w))


def _marginal2(allpts, ref, c):
    """Marginal hypervolume contribution of candidate c given set allpts."""
    if c[0] >= ref[0] or c[1] >= ref[1]:
        return 0.0
    if len(allpts) == 0:
        return (ref[0] - c[0]) * (ref[1] - c[1])
    if np.any((allpts[:, 0] <= c[0]) & (allpts[:, 1] <= c[1]) &
              ((allpts[:, 0] < c[0]) | (allpts[:, 1] < c[1]))):
        return 0.0
    mask = allpts[:, 0] <= c[0]
    if np.any(mask):
        x_left = allpts[mask, 0].max()
        y_left = allpts[mask, 1].min()
    else:
        x_left = c[0]
        y_left = ref[1]
    return max(0.0, c[0] - x_left) * max(0.0, y_left - c[1])


def _rbf_predict(X, Y, x, ls_cache=None):
    """Gaussian RBF surrogate with cached length scale."""
    m = len(X)
    if m == 1:
        return float(Y[0])
    diff = X - x
    sq = np.sum(diff * diff, axis=1)
    if ls_cache is not None:
        ls = ls_cache
    else:
        k = min(30, m)
        idx = np.random.choice(m, k, replace=False)
        ds = []
        for a in range(k):
            for b in range(a + 1, k):
                ds.append(np.sum((X[idx[a]] - X[idx[b]]) ** 2))
        med = np.median(ds) if ds else 1.0
        ls = max(1e-6, np.sqrt(med) / 2.0)
    w = np.exp(-sq / (2.0 * ls * ls))
    wsum = w.sum()
    if wsum <= 0:
        return float(np.mean(Y))
    return float(np.dot(w, Y) / wsum)


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 270.0
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        ref = np.array([3000.0, 0.06], dtype=np.float64)
        d = self.dim
        B = int(self.budget)

        def tleft():
            return deadline - time.time()

        def ev(x):
            f = np.asarray(problem(x), dtype=np.float64).ravel()
            return f[0], f[1]

        # Archive: list of (x, f) for non-dominated points
        arch_x = []
        arch_f = []

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def add_to_archive(x_new, f_new):
            """Add to archive if non-dominated; remove dominated points."""
            f_new = np.asarray(f_new, dtype=np.float64)
            if f_new[0] >= ref[0] or f_new[1] >= ref[1]:
                return False
            if len(arch_f) == 0:
                arch_x.append(x_new.copy())
                arch_f.append(f_new.copy())
                return True
            # Check if dominated
            for f_exist in arch_f:
                if dominates(f_exist, f_new):
                    return False
            # Remove dominated points
            new_xs = []
            new_fs = []
            for i in range(len(arch_f)):
                if not dominates(f_new, arch_f[i]):
                    new_xs.append(arch_x[i])
                    new_fs.append(arch_f[i])
            new_xs.append(x_new.copy())
            new_fs.append(f_new.copy())
            arch_x.clear()
            arch_f.clear()
            for i in range(len(new_xs)):
                arch_x.append(new_xs[i])
                arch_f.append(new_fs[i])
            return True

        # Phase 1: Initial diverse sampling
        n_init = min(30, B)
        for i in range(n_init):
            if tleft() < 30.0:
                break
            # Mix of random and Latin hypercube-style points
            if i < 10:
                # Stratified grid
                n_per_dim = 3
                idx = i
                x = np.empty(d)
                for dd in range(d):
                    pos = (idx // (n_per_dim ** dd)) % n_per_dim
                    x[dd] = lo[dd] + span[dd] * ((pos + 0.5 + 0.3 * np.random.random()) / n_per_dim)
            else:
                x = lo + span * np.random.random(d)
            x = np.clip(x, lo, hi)
            f = ev(x)
            add_to_archive(x, f)

        # Cache for RBF length scale
        ls_cache = None
        ls_update_interval = 20
        eval_count = n_init

        # Phase 2: Main loop - surrogate + evolutionary candidate generation
        while len(arch_f) < B and tleft() > 10.0:
            # Update ls cache periodically
            if eval_count % ls_update_interval == 0 and len(arch_x) >= 4:
                X_arr = np.array(arch_x)
                m = len(X_arr)
                k = min(30, m)
                idx = np.random.choice(m, k, replace=False)
                ds = []
                for a in range(k):
                    for b in range(a + 1, k):
                        ds.append(np.sum((X_arr[idx[a]] - X_arr[idx[b]]) ** 2))
                med = np.median(ds) if ds else 1.0
                ls_cache = max(1e-6, np.sqrt(med) / 2.0)

            # Build candidate pool
            candidates = []

            # 1. Differential evolution-style mutation from archive
            if len(arch_x) >= 3:
                n_de = min(12, len(arch_x))
                for _ in range(n_de):
                    i1 = np.random.randint(len(arch_x))
                    i2 = np.random.randint(len(arch_x))
                    i3 = np.random.randint(len(arch_x))
                    if len(set([i1, i2, i3])) < 3:
                        continue
                    F = 0.5 + 0.5 * np.random.random()
                    x = arch_x[i1] + F * (arch_x[i2] - arch_x[i3])
                    x = np.clip(x, lo, hi)
                    candidates.append(x)

            # 2. RBF-guided local search around archive points
            if len(arch_x) >= 2:
                n_local = min(10, len(arch_x))
                for j in np.random.choice(len(arch_x), min(n_local, len(arch_x)), replace=False):
                    base = arch_x[j]
                    for _ in range(2):
                        scale = 0.1 + 0.1 * np.random.random()
                        x = base + span * scale * np.random.standard_normal(d)
                        x = np.clip(x, lo, hi)
                        candidates.append(x)

            # 3. Pure random exploration
            n_rand = 8
            for _ in range(n_rand):
                x = lo + span * np.random.random(d)
                candidates.append(x)

            # 4. Targeted exploration toward gaps in the Pareto front
            if len(arch_f) >= 2:
                F_arr = np.array(arch_f)
                # Find the "gap" in f1 coverage
                f1_vals = sorted(F_arr[:, 0])
                if len(f1_vals) >= 2:
                    gaps = []
                    for i in range(len(f1_vals) - 1):
                        gap = f1_vals[i + 1] - f1_vals[i]
                        mid_f1 = (f1_vals[i] + f1_vals[i + 1]) / 2
                        gaps.append((gap, mid_f1))
                    if gaps:
                        gaps.sort(reverse=True)
                        _, target_f1 = gaps[0]
                        # Find point with closest f1 to target
                        idxs = np.argsort(np.abs(F_arr[:, 0] - target_f1))
                        for k in idxs[:2]:
                            base = arch_x[k]
                            x = base + span * 0.2 * np.random.standard_normal(d)
                            x = np.clip(x, lo, hi)
                            candidates.append(x)

            # Predict objectives using RBF
            if len(arch_x) >= 2 and len(candidates) > 0:
                X_arr = np.array(arch_x)
                Y0 = np.array([f[0] for f in arch_f])
                Y1 = np.array([f[1] for f in arch_f])
                pred = np.empty((len(candidates), 2))
                for ci in range(len(candidates)):
                    pred[ci, 0] = _rbf_predict(X_arr, Y0, candidates[ci], ls_cache)
                    pred[ci, 1] = _rbf_predict(X_arr, Y1, candidates[ci], ls_cache)

                # Score by marginal hypervolume contribution
                F_current = np.array(arch_f)
                contribs = np.array([_marginal2(F_current, ref, pred[ci]) for ci in range(len(pred))])
                order = np.argsort(-contribs)
            else:
                order = np.random.permutation(len(candidates))

            # Evaluate top candidates
            n_eval_this_iter = min(6, len(candidates), B - len(arch_f))
            for rank in order[:n_eval_this_iter]:
                if tleft() < 5.0:
                    break
                x = candidates[rank]
                f = ev(x)
                eval_count += 1
                add_to_archive(x, f)

        # Return all evaluated points (evaluator records all returned objective vectors)
        # We return the archive points; evaluator uses all recorded vectors
        if len(arch_f) > 0:
            return np.array(arch_f)
        return np.array([])
# EVOLVE-BLOCK-END
