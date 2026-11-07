"""Optimizer for BBOB f23 (Katsuura): LHS global search + Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _lhs(n, dim, seed):
    """Latin Hypercube Sampling: n points in [0,1)^dim."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((n, dim))
    for d in range(dim):
        perm = rng.permutation(n)
        pts[:, d] = (perm + rng.rand(n)) / n
    return pts


def _nelder_mead(func, x0, dim, budget, calls_ref, deadline, best_ref):
    """Nelder-Mead refinement. Updates calls_ref[0] and best_ref[0/1]."""
    # Initial simplex
    simplex = [x0.copy()]
    for d in range(dim):
        xi = x0.copy()
        step = 0.05 if abs(x0[d]) < 4.9 else 0.01
        xi[d] += step
        simplex.append(np.clip(xi, -5.0, 5.0))

    fs = []
    for i in range(dim + 1):
        if calls_ref[0] >= budget or time.time() > deadline:
            break
        f = func(simplex[i])
        calls_ref[0] += 1
        fs.append(f)
        if f < best_ref[1]:
            best_ref[1] = f
            best_ref[0] = simplex[i].copy()

    if len(fs) < dim + 1:
        return best_ref[0]

    # Sort simplex by fitness
    order = np.argsort(fs)
    simplex = [simplex[i] for i in order]
    fs = [fs[i] for i in order]

    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink

    while calls_ref[0] < budget and time.time() < deadline:
        # Sort
        order = np.argsort(fs)
        simplex = [simplex[i] for i in order]
        fs = [fs[i] for i in order]

        # Convergence check
        if abs(fs[0] - fs[-1]) < 1e-12:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, -5.0, 5.0)
        if calls_ref[0] >= budget or time.time() > deadline:
            break
        fr = func(xr)
        calls_ref[0] += 1
        if fr < best_ref[1]:
            best_ref[1] = fr
            best_ref[0] = xr.copy()

        if fs[0] <= fr < fs[-2]:
            # Accept reflection
            simplex[-1] = xr
            fs[-1] = fr
        elif fr < fs[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, -5.0, 5.0)
            if calls_ref[0] >= budget or time.time() > deadline:
                break
            fe = func(xe)
            calls_ref[0] += 1
            if fe < best_ref[1]:
                best_ref[1] = fe
                best_ref[0] = xe.copy()
            if fe < fr:
                simplex[-1] = xe
                fs[-1] = fe
            else:
                simplex[-1] = xr
                fs[-1] = fr
        else:
            # Contraction
            if fr < fs[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (simplex[-1] - centroid)
            xc = np.clip(xc, -5.0, 5.0)
            if calls_ref[0] >= budget or time.time() > deadline:
                break
            fc = func(xc)
            calls_ref[0] += 1
            if fc < best_ref[1]:
                best_ref[1] = fc
                best_ref[0] = xc.copy()

            if fc < min(fr, fs[-1]):
                simplex[-1] = xc
                fs[-1] = fc
            else:
                # Shrink
                for i in range(1, dim + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                    if calls_ref[0] >= budget or time.time() > deadline:
                        break
                    fs[i] = func(simplex[i])
                    calls_ref[0] += 1
                    if fs[i] < best_ref[1]:
                        best_ref[1] = fs[i]
                        best_ref[0] = simplex[i].copy()
                if calls_ref[0] >= budget or time.time() > deadline:
                    break

        # If simplex is too small, break
        spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
        if np.max(spread) < 1e-8:
            break

    return best_ref[0]


class Optimizer:
    """LHS global exploration + Nelder-Mead refinement for Katsuura on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget

        calls_ref = [0]
        best_ref = [None, np.inf]

        def record(x, f):
            calls_ref[0] += 1
            if f < best_ref[1]:
                best_ref[1] = f
                best_ref[0] = np.asarray(x, dtype=float).copy()

        # Phase 1: Latin Hypercube global exploration (150 calls)
        n_lhs = 150
        lhs_pts = _lhs(n_lhs, dim, seed=42)
        for i in range(n_lhs):
            if calls_ref[0] >= budget or time.time() > deadline:
                break
            x = lhs_pts[i] * 10.0 - 5.0  # map [0,1)^dim to [-5,5)^dim
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            record(x, f)

        if best_ref[0] is None:
            # Fallback: evaluate center
            x0 = np.zeros(dim)
            f0 = func(x0)
            record(x0, f0)

        # Phase 2: Nelder-Mead refinement from best LHS point
        if calls_ref[0] < budget and time.time() < deadline:
            _nelder_mead(func, best_ref[0].copy(), dim, budget, calls_ref, deadline, best_ref)

        return best_ref[0], best_ref[1]
# EVOLVE-BLOCK-END
