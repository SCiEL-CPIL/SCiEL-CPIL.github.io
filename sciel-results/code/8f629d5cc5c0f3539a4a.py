"""Optimized packing of 50 equal circles in a unit square using smooth optimization."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 50


def _score(c):
    """Compute the packing radius for center array c of shape (N,2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Distance to walls
    wall = 0.5 - np.max(np.abs(c), axis=1)
    if np.any(wall < -1e-12):
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    r = min(min_pair, np.min(wall))
    if r < -1e-12:
        return 0.0
    return max(r, 0.0)


def _smooth_min(values, temperature=0.05):
    """Smooth approximation of min using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    m = np.min(values)
    if m < -1e10:
        return m
    # Avoid overflow: use stable log-sum-exp
    exp_vals = np.exp((values - m) / temperature)
    s = np.sum(exp_vals)
    if s <= 0 or not np.isfinite(s):
        return m
    return m + temperature * np.log(s) / temperature  # Wait, this is just m + temperature * log(sum(exp(...)))
    # Actually: soft_min = m + temperature * log(sum(exp((v-m)/temperature)))
    # But that's not quite right. Let me fix:
    # soft_min(v) ≈ m + temperature * log( sum(exp((v_i - m)/temperature)) )
    # No wait, that gives m + temperature * log(N) when all equal, which is wrong.
    # Correct: soft_min = -temperature * log( sum(exp(-v_i / temperature)) )
    # Let me redo this properly.


def _soft_min(values, temperature=0.05):
    """Soft minimum: -T * log(sum(exp(-v_i/T))). As T->0, approaches min."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    # Stable computation
    m = np.min(values)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(-(values - m) / temperature)
    s = np.sum(exp_vals)
    if s <= 0 or not np.isfinite(s):
        return m
    return m - temperature * np.log(s)


def _objective(x, temperature=0.05):
    """Objective to minimize: -soft_min of all constraints (distances/2 and wall distances)."""
    c = x.reshape(N, 2)
    # Wall distances
    wall = 0.5 - np.max(np.abs(c), axis=1)
    # Pairwise distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_half = dist / 2.0
    # Collect all constraint values
    all_vals = np.concatenate([wall, pair_half[np.triu_indices(N, k=1)]])
    # We want to maximize the minimum, so minimize the negative soft-min
    return -_soft_min(all_vals, temperature)


def _objective_grad(x, temperature=0.05):
    """Gradient of the objective for L-BFGS-B."""
    c = x.reshape(N, 2)
    # Wall distances
    wall = 0.5 - np.max(np.abs(c), axis=1)
    # Pairwise distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_half = dist / 2.0

    all_vals = np.concatenate([wall, pair_half[np.triu_indices(N, k=1)]])
    m = np.min(all_vals)
    exp_vals = np.exp(-(all_vals - m) / temperature)
    s = np.sum(exp_vals)
    if s <= 0 or not np.isfinite(s):
        # Fallback to hard min gradient
        return None

    weights = exp_vals / s
    # Gradient of soft_min w.r.t. all_vals is -weights
    # d(soft_min)/d(v_i) = -exp(-(v_i-m)/T) / sum(exp(-(v_j-m)/T)) = -weights_i

    grad_all = -weights

    # Now back-propagate to x
    grad_x = np.zeros_like(x)

    # Wall contributions: wall_i = 0.5 - max(|c_i0|, |c_i1|)
    # d(wall_i)/d(c_ij) = -sign(c_ij) if |c_ij| == max(|c_i|)
    abs_c = np.abs(c)
    max_abs = np.max(abs_c, axis=1)
    for i in range(N):
        w = grad_all[i]
        for j in range(2):
            if abs_c[i, j] == max_abs[i]:
                grad_x[2 * i + j] = -w * np.sign(c[i, j])
            elif abs_c[i, j] > 0 and abs_c[i, j] == max_abs[i]:
                grad_x[2 * i + j] = -w * np.sign(c[i, j])

    # Pair contributions
    idx = np.triu_indices(N, k=1)
    for k in range(len(idx[0])):
        i, j = idx[0][k], idx[1][k]
        w = grad_all[N + k]
        # pair_half_kj = dist(c_i, c_j) / 2
        # d(dist)/d(c_i) = (c_i - c_j) / dist
        # d(pair_half)/d(c_i) = (c_i - c_j) / (2 * dist)
        d = dist[i, j]
        if d > 1e-12:
            diff_ij = c[i] - c[j]
            grad_x[2 * i:2 * i + 2] += w * diff_ij / d
            grad_x[2 * j:2 * j + 2] += w * (-diff_ij) / d

    return grad_x


def _initial_grid():
    """Create an initial grid arrangement."""
    m = int(np.ceil(np.sqrt(N)))
    # Use a slightly perturbed grid to break symmetry
    np.random.seed(42)
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    # Add small random perturbation
    grid += np.random.normal(0, 0.005, grid.shape)
    return grid


def _random_initial():
    """Random initial configuration."""
    np.random.seed(int(time.time() * 1000) % 10000)
    c = np.random.uniform(-0.45, 0.45, (N, 2))
    return c


def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill

    best_c = None
    best_score = 0.0

    # Start with grid
    c = _initial_grid()
    best_c = c.copy()
    best_score = _score(c)

    # Try optimization from multiple starting points
    starts = [c.copy()]

    # Add a few random starts
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        starts.append(np.random.uniform(-0.45, 0.45, (N, 2)))

    # Also try a hexagonal-like arrangement
    # Hexagonal packing in a square
    hex_c = []
    row = 0
    while len(hex_c) < N:
        y = -0.45 + row * 0.12
        if y > 0.45:
            break
        offset = 0.06 if row % 2 == 0 else 0.0
        x = -0.45 + offset
        col = 0
        while x < 0.45 and len(hex_c) < N:
            hex_c.append([x, y])
            x += 0.12
            col += 1
        row += 1
    if len(hex_c) >= N:
        starts.append(np.array(hex_c[:N]))

    for start in starts:
        if time.time() > deadline - 10:
            break

        x0 = start.flatten()

        # Use L-BFGS-B with bounds
        bounds = [(-0.5, 0.5)] * (2 * N)

        try:
            # First pass with higher temperature
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if time.time() > deadline - 5:
                break

            c_opt = result.x.reshape(N, 2)
            s = _score(c_opt)

            if s > best_score:
                best_score = s
                best_c = c_opt.copy()

            # Second pass with lower temperature for refinement
            if time.time() > deadline - 10:
                break

            result2 = minimize(
                _objective,
                c_opt.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
            )
            if time.time() > deadline - 5:
                break

            c_opt2 = result2.x.reshape(N, 2)
            s2 = _score(c_opt2)

            if s2 > best_score:
                best_score = s2
                best_c = c_opt2.copy()

        except Exception:
            continue

    # Final polish: local refinement using pairwise distance maximization
    if time.time() < deadline - 5 and best_c is not None:
        try:
            x0 = best_c.flatten()
            bounds = [(-0.5, 0.5)] * (2 * N)
            # Use a very low temperature for final polish
            result3 = minimize(
                lambda x: _objective(x, temperature=0.01),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
            )
            c_polished = result3.x.reshape(N, 2)
            s3 = _score(c_polished)
            if s3 > best_score:
                best_score = s3
                best_c = c_polished.copy()
        except Exception:
            pass

    # Ensure validity
    if best_c is None:
        best_c = _initial_grid()

    # Clamp to square
    best_c = np.clip(best_c, -0.5 + 1e-10, 0.5 - 1e-10)

    return best_c


# EVOLVE-BLOCK-START
# (The construct function and helpers above are the evolved block)
# EVOLVE-BLOCK-END
