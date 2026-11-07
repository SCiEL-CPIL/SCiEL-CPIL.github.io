"""Black-box continuous optimizer on BBOB-style f23 (Katsuura) with multi-start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _halton(dim, n, rng=None):
    """Generate n Halton sequence points in [0,1]^dim."""
    points = np.zeros((n, dim))
    for d in range(dim):
        base = d + 2
        for i in range(n):
            f = 1.0
            ip = i + 1
            while ip > 0:
                ip, rem = divmod(ip, base)
                f /= base
                points[i, d] += rem * f
    return points


def _nelder_mead(func, x0, budget, dim, rng, deadline):
    """
    Nelder-Mead with adaptive simplex.
    Returns (best_x, best_f, calls_used).
    """
    # Initial simplex
    simplex = np.zeros((dim + 1, dim))
    simplex[0] = x0
    for i in range(1, dim + 1):
        simplex[i] = x0.copy()
        simplex[i, i - 1] += 0.5  # small perturbation along each axis

    fvals = np.array([func(np.clip(s, -5, 5)) for s in simplex])
    calls = dim + 1
    best_idx = np.argmin(fvals)
    best_x = np.clip(simplex[best_idx], -5, 5).copy()
    best_f = fvals[best_idx]

    alpha, gamma, rho = 1.0, 2.0, 0.5

    while calls < budget and time.time() < deadline:
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        best_idx = 0
        worst_idx = dim
        best_f = fvals[0]
        worst_f = fvals[-1]

        # Centroid of all but worst
        centroid = np.mean(simplex[:dim], axis=0)

        # Reflection
        xr = centroid + alpha * (centroid - simplex[worst_idx])
        xr = np.clip(xr, -5, 5)
        fr = func(xr)
        calls += 1

        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, -5, 5)
            fe = func(xe)
            calls += 1
            if fe < fr:
                simplex[worst_idx] = xe
                fvals[worst_idx] = fe
            else:
                simplex[worst_idx] = xr
                fvals[worst_idx] = fr
        elif fr < fvals[-2]:
            simplex[worst_idx] = xr
            fvals[worst_idx] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (simplex[worst_idx] - centroid)
            xc = np.clip(xc, -5, 5)
            fc = func(xc)
            calls += 1
            if fc < min(fr, fvals[-1]):
                simplex[worst_idx] = xc
                fvals[worst_idx] = fc
            else:
                # Shrink
                for i in range(1, dim + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = func(np.clip(simplex[i], -5, 5))
                    calls += 1
                    if calls >= budget:
                        break

        # Track best
        cur_best_idx = np.argmin(fvals)
        if fvals[cur_best_idx] < best_f:
            best_f = fvals[cur_best_idx]
            best_x = np.clip(simplex[cur_best_idx], -5, 5).copy()

        # Check convergence
        if np.max(fvals) - np.min(fvals) < 1e-10:
            break

    return best_x, best_f, calls


class Optimizer:
    """Multi-start Nelder-Mead with Halton initialization for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s

        rng = np.random.default_rng(42)

        # Phase 1: Halton-based space-filling exploration
        # Use a good fraction of budget for initial exploration
        n_halton = min(80, total_budget // 5)
        halton_points = _halton(dim, n_halton, rng)
        # Map from [0,1]^dim to [-5,5]^dim
        halton_points = halton_points * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        calls_used = 0
        candidates = []

        for i in range(n_halton):
            if time.time() > deadline - 120:  # leave time for Nelder-Mead
                break
            x = halton_points[i]
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Multi-start Nelder-Mead from top-k diverse points
        # Select diverse points from top candidates
        selected = []
        min_dist = 1.5  # minimum distance between selected points
        for fx, x in candidates:
            if len(selected) >= 8:
                break
            if all(np.linalg.norm(x - sx) > min_dist for _, sx in selected):
                selected.append((fx, x))
            if time.time() > deadline - 60:
                break

        # If not enough diverse points, add more from top
        if len(selected) < 5:
            for fx, x in candidates:
                if (fx, x) not in selected:
                    selected.append((fx, x))
                if len(selected) >= 5:
                    break

        # Allocate remaining budget among Nelder-Mead runs
        remaining_budget = total_budget - calls_used
        if remaining_budget <= 0:
            return best_x, best_f

        n_runs = len(selected)
        budget_per_run = max(30, remaining_budget // n_runs)

        for i, (fx, x0) in enumerate(selected):
            if calls_used >= total_budget:
                break
            if time.time() > deadline - 30:
                break

            # Reduce budget for later runs if time is tight
            run_budget = min(budget_per_run, total_budget - calls_used)
            if run_budget < 20:
                break

            bx, bf, calls_in_run = _nelder_mead(
                func, x0, run_budget, dim, rng, deadline
            )
            calls_used += calls_in_run

            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
