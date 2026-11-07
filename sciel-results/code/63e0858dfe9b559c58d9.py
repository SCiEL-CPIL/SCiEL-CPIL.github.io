"""Multi-objective optimizer on DTLZ2 (3 objectives, 12 vars)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _sobol_matrix(n, dim, seed=0):
    """Generate an n x dim Sobol quasi-random matrix in [0,1) using a simple
    bit-reversal / Gray-code based construction (no scipy dependency needed)."""
    # Use scipy if available for true Sobol, else fall back to Halton-like
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        return sampler.random(n)
    except Exception:
        # Fallback: Halton sequence
        def halton(index, base):
            f, r = 1.0, 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        pts = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2
            for i in range(n):
                pts[i, d] = halton(i + 1, base)
        return pts


def _non_dominated_mask(F):
    """Return boolean mask of non-dominated points in F (n x m)."""
    n = len(F)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # j dominates i if F[j] <= F[i] in all and < in at least one
        better = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        better[i] = False  # self not counted
        if np.any(better):
            mask[i] = False
    return mask


def _crowding_distance(F, mask):
    """Compute crowding distance for non-dominated points."""
    n = len(F)
    m = F.shape[1]
    cd = np.zeros(n)
    indices = np.where(mask)[0]
    k = len(indices)
    if k <= 2:
        cd[indices] = np.inf
        return cd
    for j in range(m):
        order = np.argsort(F[indices, j])
        cd[indices[order[0]]] = np.inf
        cd[indices[order[-1]]] = np.inf
        fmin = F[indices[order[0]], j]
        fmax = F[indices[order[-1]], j]
        rng = fmax - fmin
        if rng < 1e-15:
            continue
        for i in range(1, k - 1):
            cd[indices[order[i]]] += (
                F[indices[order[i + 1]], j] - F[indices[order[i - 1]], j]
            ) / rng
    return cd


def _nelder_mead_scalarize(problem, x0, weights, bounds, max_iter=80, tol=1e-10):
    """Minimize weighted sum of objectives via Nelder-Mead (bounded via clipping)."""
    from scipy.optimize import minimize
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def obj(x):
        x = np.clip(x, lo, hi)
        f = np.array(problem(x), dtype=float)
        return float(np.dot(weights, f))

    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": max_iter, "xatol": tol, "fatol": tol})
    return np.clip(res.x, lo, hi), obj(res.x)


class Optimizer:
    """NSGA-II-style multi-objective optimizer with Sobol init and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270  # safety margin under 300s / 330s limits

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim, n_obj = self.dim, self.n_obj

        # --- Phase 1: Structured initialization ---
        pop_size = min(40, max(10, self.budget // 5))
        init_n = min(pop_size, self.budget)
        X = _sobol_matrix(init_n, dim, seed=42) * span + lo
        F = np.array([np.array(problem(x), dtype=float) for x in X], dtype=float)
        used = init_n
        all_F = [F.copy()]  # keep all evaluated points for hypervolume

        # --- Phase 2: NSGA-II-style evolution ---
        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated front + crowding
            mask = _non_dominated_mask(F)
            cd = _crowding_distance(F, mask)

            # Select parents (prefer non-dominated, high crowding)
            n_parents = max(2, pop_size // 2)
            parent_idx = np.where(mask)[0]
            if len(parent_idx) == 0:
                parent_idx = np.arange(len(F))
            # Sort by crowding desc, pick top
            parent_idx = parent_idx[np.argsort(cd[parent_idx])[::-1]]
            parent_idx = parent_idx[:min(n_parents, len(parent_idx))]

            # Generate children via crossover + mutation
            n_children = min(pop_size, self.budget - used)
            if n_children <= 0:
                break

            children_X = []
            children_F = []
            for _ in range(n_children):
                if time.time() > deadline:
                    break
                i, j = (np.random.randint(len(parent_idx)),
                        np.random.randint(len(parent_idx)))
                pi, pj = parent_idx[i], parent_idx[j]
                # Simulated binary crossover (simplified: blend)
                alpha = np.random.rand()
                child = alpha * X[pi] + (1 - alpha) * X[pj]
                # Gaussian mutation
                sigma = 0.05 * np.random.rand()
                child = child + sigma * span * np.random.randn(dim)
                child = np.clip(child, lo, hi)
                children_X.append(child)
                children_F.append(np.array(problem(child), dtype=float))
                used += 1

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)
            all_F.append(children_F)

            # Combine and select
            X_comb = np.vstack([X, children_X])
            F_comb = np.vstack([F, children_F])
            mask_comb = _non_dominated_mask(F_comb)
            cd_comb = _crowding_distance(F_comb, mask_comb)

            # Sort: non-dominated first (by crowding desc), then dominated
            order = np.lexsort((
                -cd_comb,
                mask_comb.astype(int) * -1  # non-dominated first
            ))
            # Actually: sort by (is_dominated, -crowding)
            sort_keys = np.lexsort((-cd_comb, mask_comb.astype(int) * -1))
            X = X_comb[sort_keys][:pop_size]
            F = F_comb[sort_keys][:pop_size]

        # --- Phase 3: Local polish on best non-dominated points ---
        if time.time() < deadline - 30:
            mask = _non_dominated_mask(F)
            best_idx = np.where(mask)[0]
            if len(best_idx) == 0:
                best_idx = np.arange(len(F))
            # Pick up to 8 diverse points to polish
            cd = _crowding_distance(F, mask)
            best_idx = best_idx[np.argsort(cd[best_idx])[::-1]]
            best_idx = best_idx[:min(8, len(best_idx))]

            # Use a few scalarization directions
            n_obj_local = n_obj
            directions = []
            for _ in range(4):
                w = np.random.dirichlet(np.ones(n_obj_local))
                directions.append(w)
            # Add uniform direction
            directions.append(np.ones(n_obj_local) / n_obj_local)

            polish_budget = self.budget - used
            per_dir = max(1, polish_budget // len(directions)) if polish_budget > 0 else 0

            for w in directions:
                if used >= self.budget or time.time() > deadline:
                    break
                # Polish the best point with this weight
                x0 = X[best_idx[0]].copy()
                if used < self.budget:
                    x_polished, _ = _nelder_mead_scalarize(
                        problem, x0, w, bounds,
                        max_iter=min(60, max(10, polish_budget // 2)),
                        tol=1e-12
                    )
                    f_polished = np.array(problem(x_polished), dtype=float)
                    used += 1
                    all_F.append(f_polished.reshape(1, -1))

        return X
# EVOLVE-BLOCK-END
