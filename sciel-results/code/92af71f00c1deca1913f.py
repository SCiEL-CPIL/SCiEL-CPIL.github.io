"""Sphere packing in a unit cube: 100 equal spheres, maximize radius."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 150.0
SAFETY = 20.0  # safety margin in seconds


def _compute_r(c):
    """Compute the valid radius for centres c (N,3)."""
    # min pairwise distance / 2
    d = np.sqrt(np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    r_pair = d.min() / 2.0
    # distance to walls
    r_wall = (0.5 - np.abs(c)).min()
    return min(r_pair, r_wall)


def _initial_grid():
    """Structured initial packing: densest lattice-like arrangement."""
    # Use a 5x5x4 arrangement (100 points) with small jitter to break symmetry
    rng = np.random.default_rng(42)
    nx, ny, nz = 5, 5, 4
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    # Add tiny jitter to break perfect symmetry
    grid += rng.uniform(-1e-4, 1e-4, size=grid.shape)
    return grid[:N]


def _force_directed(c, time_budget):
    """Repulsion-based force-directed optimization with smooth objective."""
    rng = np.random.default_rng(123)
    c = c.copy()
    best_c = c.copy()
    best_r = _compute_r(c)
    t_end = time.time() + time_budget
    temp = 0.05
    cool = 0.995

    while time.time() < t_end:
        # Compute forces: repulsion between close pairs + wall repulsion
        d = np.sqrt(np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=-1))
        np.fill_diagonal(d, np.inf)

        # Repulsive force (inverse square, capped)
        d_safe = np.maximum(d, 1e-10)
        force = np.zeros_like(c)
        for i in range(N):
            for j in range(i + 1, N):
                diff = c[i] - c[j]
                dist = d[i, j]
                if dist < 0.3:  # only interact when close
                    f = 1.0 / (dist * dist + 1e-6)
                    f = min(f, 10.0)
                    force[i] += f * diff / dist
                    force[j] -= f * diff / dist

        # Wall repulsion
        for i in range(N):
            for k in range(3):
                if c[i, k] > 0.45:
                    force[i, k] -= 5.0 * (c[i, k] - 0.45)
                elif c[i, k] < -0.45:
                    force[i, k] += 5.0 * (-0.45 - c[i, k])

        # Move with temperature
        step = temp * 0.01
        c_new = c + step * force + rng.normal(0, temp * 0.005, size=c.shape)
        # Clamp to cube
        c_new = np.clip(c_new, -0.499, 0.499)

        r_new = _compute_r(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
        # Accept with Metropolis criterion
        if r_new >= best_r or rng.random() < np.exp((r_new - best_r) / (temp + 1e-10)):
            c = c_new
        temp *= cool

    return best_c, best_r


def _logsumexp_objective(flat, c, alpha):
    """Smooth surrogate: maximize log-sum-exp of negative distances."""
    c = flat.reshape(N, 3)
    # Pairwise distances
    d = np.sqrt(np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    # Wall distances
    wall = 0.5 - np.abs(c)
    wall_min = wall.min(axis=1)  # (N,)
    # Combine: we want to maximize min of all constraints
    # Use log-sum-exp of negative values (smooth min approximation)
    neg_d = -d[~np.isinf(d)]
    neg_wall = -wall_min
    all_neg = np.concatenate([neg_d, neg_wall])
    # log-sum-exp
    m = all_neg.max()
    if np.isfinite(m):
        lse = m + np.log(np.sum(np.exp(all_neg - m)))
    else:
        lse = 0.0
    return lse  # minimize this (negative of smooth min)


def _polish_lbfgs(c, time_budget):
    """Polish with L-BFGS on smooth log-sum-exp surrogate."""
    t_end = time.time() + time_budget
    best_c = c.copy()
    best_r = _compute_r(c)

    for alpha in [50, 100, 200, 400]:
        if time.time() > t_end - 5:
            break
        flat = c.flatten()

        def obj(flat_x):
            if time.time() > t_end:
                return 0.0
            return _logsumexp_objective(flat_x, c, alpha)

        def jac(flat_x):
            # Numerical gradient
            eps = 1e-7
            g = np.zeros_like(flat_x)
            f0 = obj(flat_x)
            for i in range(len(flat_x)):
                flat_x[i] += eps
                f1 = obj(flat_x)
                flat_x[i] -= eps
                g[i] = (f1 - f0) / eps
            return g

        result = minimize(
            obj, flat, method='L-BFGS-B', jac=jac,
            bounds=[(-0.499, 0.499)] * (3 * N),
            options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        c_new = np.clip(result.x.reshape(N, 3), -0.499, 0.499)
        r_new = _compute_r(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
        c = c_new

    return best_c, best_r


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE - SAFETY

    # 1. Initial structured packing
    c = _initial_grid()
    best_c = c.copy()
    best_r = _compute_r(c)

    # 2. Force-directed / simulated annealing phase
    fd_time = min(40.0, (deadline - time.time()) * 0.5)
    if fd_time > 5:
        c, r = _force_directed(c, fd_time)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    # 3. L-BFGS polish phase
    polish_time = max(5.0, deadline - time.time() - 5)
    if polish_time > 5:
        c, r = _polish_lbfgs(best_c, polish_time)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    # 4. Final validation
    best_c = np.clip(best_c, -0.499, 0.499)
    final_r = _compute_r(best_c)

    # Ensure valid output
    if not np.all(np.isfinite(best_c)):
        return _initial_grid()

    return best_c
