"""Pack 200 equal circles in the unit square [-1/2, 1/2]^2 to maximize common radius.

Strategy:
  1. Start from a square grid (valid seed).
  2. Use L-BFGS-B to minimize a smooth surrogate of the packing objective.
     The surrogate uses a soft-min (log-sum-exp) over pair distances and wall
     distances so the objective is differentiable.
  3. After the main optimization, run a final polish pass with tighter
     tolerances and more iterations.
  4. Enforce a wall-clock deadline so we always return a valid result.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 200
DEADLINE = 120.0          # seconds
SAFETY_MARGIN = 10.0      # return at least this many seconds before the limit


def _soft_min(values, temperature=0.01):
    """Approximate min via log-sum-exp (smooth)."""
    # To avoid overflow, shift by max
    v = np.asarray(values, dtype=np.float64)
    v_max = np.max(v)
    return v_max + temperature * np.log(np.sum(np.exp((v - v_max) / temperature)))


def _packing_loss(x, temperature=0.01):
    """
    Loss = -soft_min(min_pair_dist/2, min_wall_dist)
    We want to maximize the minimum radius, so we minimize the negative.

    x: flat array of length 2*N, centres in [-1/2, 1/2]^2
    """
    c = x.reshape(N, 2)

    # Pairwise distances (only i<j)
    # For 200 points, 200*199/2 = 19900 pairs — fine
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists[np.triu_indices(N, k=1)]
    min_pair = np.min(pair_dists)

    # Wall distances: distance to each of the 4 sides
    # x in [-1/2, 1/2], so wall dist for x-axis = 0.5 - |x|
    wall_dists = np.concatenate([
        0.5 - np.abs(c[:, 0]),
        0.5 - np.abs(c[:, 1]),
    ])
    min_wall = np.min(wall_dists)

    # The radius is min(min_pair/2, min_wall)
    # We use soft-min over the two candidate radii
    r_pair = min_pair / 2.0
    # Combine via soft-min for smoothness
    # Actually, let's just take the raw min for the gradient to be meaningful,
    # but use a soft penalty for violations.

    # Better approach: maximize min(r_pair, min_wall)
    # Use a smooth approximation: soft-min of the two values
    r = _soft_min(np.array([r_pair, min_wall]), temperature=temperature)

    # Penalty for any overlap or wall violation (hard constraints)
    # If any pair distance < 2*r or wall dist < r, add penalty
    # But we're maximizing r, so just return -r and let the optimizer handle it.
    # To be safe, add a large penalty if constraints are violated.
    penalty = 0.0
    # Check if any pair distance < 2 * current r estimate
    # Use the actual min, not soft
    if min_pair < 2 * r - 1e-12:
        penalty += (2 * r - min_pair) ** 2 * 100
    if min_wall < r - 1e-12:
        penalty += (r - min_wall) ** 2 * 100

    return -r + penalty


def _packing_loss_grad(x, temperature=0.01):
    """Numerical gradient (central difference) for the loss."""
    eps = 1e-6
    f0 = _packing_loss(x, temperature)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        grad[i] = (_packing_loss(xp, temperature) - _packing_loss(xm, temperature)) / (2 * eps)
    return grad


def construct():
    t_start = time.time()

    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    x0 = grid[:N].flatten()

    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * N)

    best_x = x0.copy()
    best_r = _compute_radius(best_x.reshape(N, 2))

    # Phase 1: L-BFGS-B with moderate temperature
    for temp in [0.05, 0.02, 0.01, 0.005]:
        if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
            break
        try:
            res = minimize(
                _packing_loss,
                best_x,
                args=(temp,),
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 500,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxfun': 2000,
                    'disp': False,
                },
            )
            if res.success or res.fun < -_compute_radius(best_x.reshape(N, 2)):
                new_x = res.x
                new_r = _compute_radius(new_x.reshape(N, 2))
                if new_r > best_r:
                    best_x = new_x
                    best_r = new_r
        except Exception:
            pass

    # Phase 2: Polish with very low temperature and more iterations
    if time.time() - t_start < DEADLINE - SAFETY_MARGIN:
        try:
            res = minimize(
                _packing_loss,
                best_x,
                args=(0.001,),
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 2000,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxfun': 5000,
                    'disp': False,
                },
            )
            new_x = res.x
            new_r = _compute_radius(new_x.reshape(N, 2))
            if new_r > best_r:
                best_x = new_x
                best_r = new_r
        except Exception:
            pass

    # Final: local nudge to fix any tiny violations
    c = best_x.reshape(N, 2)
    c = _fix_violations(c)
    best_x = c.flatten()

    return best_x.reshape(N, 2)


def _compute_radius(c):
    """Compute the actual max radius for centre array c."""
    # Pairwise min distance
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)

    # Wall distances
    wall_dists = np.concatenate([
        0.5 - np.abs(c[:, 0]),
        0.5 - np.abs(c[:, 1]),
    ])
    min_wall = np.min(wall_dists)

    return min(min_pair / 2.0, min_wall)


def _fix_violations(c, max_iter=100):
    """Push overlapping circles apart and keep them inside the square."""
    c = c.copy()
    for _ in range(max_iter):
        # Check wall violations
        for i in range(N):
            for axis in range(2):
                if c[i, axis] < -0.5:
                    c[i, axis] = -0.5
                elif c[i, axis] > 0.5:
                    c[i, axis] = 0.5

        # Check pair overlaps: if dist < 2*r, push apart
        r = _compute_radius(c)
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                d = c[i] - c[j]
                dist = np.linalg.norm(d)
                min_dist = 2 * r
                if dist < min_dist - 1e-12:
                    if dist < 1e-12:
                        # Coincident: nudge randomly
                        c[i] += np.random.normal(0, 1e-4, 2)
                        c[j] -= np.random.normal(0, 1e-4, 2)
                    else:
                        direction = d / dist
                        push = (min_dist - dist) / 2.0 + 1e-6
                        c[i] += direction * push
                        c[j] -= direction * push
                    moved = True

        if not moved:
            break

        # Re-clamp to square
        c = np.clip(c, -0.5, 0.5)

    return c
