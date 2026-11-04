"""Pack 100 equal spheres in a unit cube, maximizing common radius.

Algorithm:
  1. Initialize from an FCC lattice (best known densest sphere packing).
  2. Energy minimization with a potential of the form
         E = sum_{i<j} 1/d_ij^p  +  w * sum_i 1/bd_i^p
     using L-BFGS-B, starting at moderate p and gradually increasing p
     to sharpen the objective toward the true max-min distance.
  3. A local "pair-push" refinement step that nudges each pair that is
     closer than current best + margin apart, keeping points in-bounds.
  4. Repeat polish passes while wall-clock time allows.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 100
CUBE = 0.5  # cube is [-0.5, 0.5]^3

_DEADLINE = 125.0  # safety margin before 150 s kill


def _compute_radius(centers):
    """Return the largest common radius for the given centres."""
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    if n < 2:
        return float('inf')
    # pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, np.inf)
    pair_min = np.sqrt(np.min(dist_sq))
    # boundary distances
    abs_c = np.abs(centers)
    boundary_dists = np.min(CUBE - abs_c, axis=1)
    boundary_min = np.min(boundary_dists)
    return float(min(pair_min, boundary_min))


def _energy_and_gradient(centers_flat, p, w):
    """Compute energy and gradient for sum 1/d^p + w * sum 1/bd^p."""
    c = centers_flat.reshape(N, 3)
    grad = np.zeros_like(c)

    # pairwise
    diff = c[:, None, :] - c[None, :, :]  # (N,N,3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, np.inf)
    dist_sq = np.maximum(dist_sq, 1e-14)

    # dE/dx_i = -p * sum_{j != i} (c_i - c_j) / dist_ij^(p+2)
    # Because d/dx_i [ dist_ij^(-p) ] = -p * dist_ij^(-p-1) * d(dist_ij)/d x_i
    #   = -p * dist_ij^(-p-1) * (x_i - x_j) / dist_ij
    #   = -p * (x_i - x_j) / dist_ij^(p+2)
    # Note: diff[i,j] = c_i - c_j
    inv_dist_pow = (dist_sq ** (-(p + 2) / 2.0))  # 1 / dist_ij^(p+2)
    weighted_diff = diff * inv_dist_pow[:, :, None]
    np.fill_diagonal(weighted_diff.diagonal(0), 0.0)
    # set diagonal to zero properly
    for i in range(N):
        weighted_diff[i, i, :] = 0.0

    pair_grad = -p * np.sum(weighted_diff, axis=1)
    grad += pair_grad

    # boundary: for each point i, dist to nearest face along each axis
    # d/dx of 1/(0.5 - x)^p = p / (0.5 - x)^(p+1)   when x <= 0
    #                         (nearest face is x = -0.5)
    # d/dx of 1/(0.5 + x)^p = -p / (0.5 + x)^(p+1)  when x >= 0
    # For each coordinate k:
    #   d_left  = 0.5 - c[i,k]
    #   d_right = 0.5 + c[i,k]
    #   d_min = min(d_left, d_right)
    #   if d_left <= d_right:  grad_k +=  w * p / d_left^(p+1)
    #   else:                  grad_k -=  w * p / d_right^(p+1)

    abs_c = np.abs(c)
    # for each point and axis, compute the signed gradient
    # We handle axes independently
    for k in range(3):
        d_left = CUBE - c[:, k]   # distance to face at -0.5
        d_right = CUBE + c[:, k]  # distance to face at +0.5
        # nearest face
        mask_left = d_left <= d_right
        # gradient contribution
        g = np.zeros(N)
        g[mask_left] = w * p / np.maximum(d_left[mask_left], 1e-14) ** (p + 1)
        g[~mask_left] = -w * p / np.maximum(d_right[~mask_left], 1e-14) ** (p + 1)
        grad[:, k] += g

    # energy
    energy = np.sum(1.0 / dist_sq[np.triu_indices(N, k=1)] ** (p / 2.0))
    # boundary energy
    # distance to nearest face for each point:
    # for each point: min over 6 faces
    # = min(0.5-|x|, 0.5-|y|, 0.5-|z|)
    boundary_dists = np.min(CUBE - np.abs(c), axis=1)
    boundary_dists = np.maximum(boundary_dists, 1e-14)
    energy += w * np.sum(1.0 / boundary_dists ** p)

    return energy, grad.ravel()


def _generate_initial_positions_fcc(rng):
    """Generate an FCC lattice initialisation for N points in [-0.5, 0.5]^3."""
    # FCC: points at all integer lattice points where all coords are even or all odd
    # (i.e., (i,j,k) with i+j+k even) in a unit cell, scaled.
    # Standard FCC in unit cell [0,1]^3 has 4 atoms at:
    #   (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    # and 12 face centers etc. Actually the 4-atom basis is:
    #   (0, 0, 0)
    #   (0, 0.5, 0.5)
    #   (0.5, 0, 0.5)
    #   (0.5, 0.5, 0)
    # in a cubic lattice of side 1.

    # We need ~100 points. 5^3 = 125 lattice points, 4 per cell → 500 points.
    # Use a 3×3×3 cubic lattice of cells → 4×27 = 108 points. Trim to 100.

    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])

    n_cells = 3  # 3×3×3 = 27 cells, 4 per cell = 108 points
    points = []
    for iz in range(n_cells):
        for iy in range(n_cells):
            for ix in range(n_cells):
                origin = np.array([ix, iy, iz])
                for b in basis:
                    points.append(origin + b)

    points = np.array(points)  # shape (108, 3)

    # Scale to fit in [-0.5, 0.5]^3
    # The lattice spans [0, n_cells) = [0,3]. We want it in [-0.5, 0.5].
    # Shift center: subtract 1.0 (center of [0,3) is 1.0)
    # Scale: divide by 3.0 so it fits in [-0.5, 0.5]
    points = (points - n_cells / 2.0) / n_cells  # now in [-0.5, 0.5]

    # Add small random perturbation to break symmetry
    points += rng.uniform(-0.01, 0.01, size=points.shape)

    # Clip to bounds
    points = np.clip(points, -CUBE + 0.001, CUBE - 0.001)

    if len(points) > N:
        # Remove excess points (last ones)
        points = points[:N]

    return points


def _generate_initial_positions_grid(rng):
    """Fallback: grid initialisation."""
    m = 5  # 5×5×5 = 125, take first 100
    ticks = (np.arange(m) + 0.5) / m - CUBE
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    centers += rng.uniform(-0.02, 0.02, size=centers.shape)
    centers = np.clip(centers, -CUBE + 0.001, CUBE - 0.001)
    return centers


def _local_pair_push(centers, margin, max_iters=200, step_scale=0.1):
    """Local refinement: push apart pairs closer than current_min + margin."""
    c = centers.copy()
    bounds = np.array([-CUBE + 1e-9, CUBE - 1e-9])

    for it in range(max_iters):
        improved = False
        diff = c[:, None, :] - c[None, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)

        # Find the minimum pair distance
        min_dist = np.min(dist)
        # Also check boundary
        boundary_dists = np.min(CUBE - np.abs(c), axis=1)
        cur_min = min(min_dist, np.min(boundary_dists))

        # Target: all distances should be > cur_min + margin
        # Push apart pairs with d < cur_min + margin
        iu = np.triu_indices(N, k=1)
        pair_dists = dist[iu]
        close_mask = pair_dists < cur_min + margin

        if not np.any(close_mask):
            # Check boundary
            for i in range(N):
                bd = np.min(CUBE - np.abs(c[i]))
                if bd < cur_min + margin:
                    improved = True
                    # Push towards center
                    c[i] = c[i] * 0.5
            if not improved:
                break

        # Move close pairs apart
        for idx in np.where(close_mask)[0]:
            i, j = iu[0][idx], iu[1][idx]
            d = c[i] - c[j]
            dist_ij = np.linalg.norm(d)
            if dist_ij < 1e-14:
                d = rng_dir()
                dist_ij = 1.0
            direction = d / dist_ij
            needed = cur_min + margin - dist_ij
            step = min(needed * step_scale, 0.02)
            new_ci = c[i] + direction * step
            new_cj = c[j] - direction * step
            # Keep in bounds
            new_ci = np.clip(new_ci, bounds[0], bounds[1])
            new_cj = np.clip(new_cj, bounds[0], bounds[1])
            c[i] = new_ci
            c[j] = new_cj
            improved = True

        # Boundary push
        for i in range(N):
            for k in range(3):
                d_to_neg = CUBE - c[i, k]  # distance to face at -0.5
                d_to_pos = CUBE + c[i, k]  # distance to face at +0.5
                d_min = min(d_to_neg, d_to_pos)
                if d_min < cur_min + margin:
                    if d_to_neg <= d_to_pos:
                        c[i, k] += min((cur_min + margin - d_to_neg) * step_scale, 0.01)
                    else:
                        c[i, k] -= min((cur_min + margin - d_to_pos) * step_scale, 0.01)
                    c[i, k] = np.clip(c[i, k], bounds[0], bounds[1])
                    improved = True

        if not improved:
            break

    return c


def construct():
    start_time = time.time()
    rng = np.random.RandomState(12345)

    best_centers = None
    best_radius = -np.inf

    bounds = [(-CUBE + 1e-9, CUBE - 1e-9)] * (N * 3)

    # Try different initializations
    inits = [
        _generate_initial_positions_fcc,
        _generate_initial_positions_grid,
    ]

    for init_fn in inits:
        if time.time() - start_time > _DEADLINE:
            break
        try:
            centers = init_fn(rng)
        except Exception:
            centers = _generate_initial_positions_grid(rng)

        current = centers.copy()
        cur_radius = _compute_radius(current)

        # Progressively increase power p for energy minimization
        powers = [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0]

        for p in powers:
            if time.time() - start_time > _DEADLINE - 5.0:
                break

            x0 = current.ravel()
            try:
                res = minimize(
                    lambda x: _energy_and_gradient(x, p, 1.0)[0],
                    x0,
                    jac=lambda x: _energy_and_gradient(x, p, 1.0)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 300,
                        'maxfun': 800,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'disp': False,
                    },
                )
                new_centers = res.x.reshape(N, 3)
                new_radius = _compute_radius(new_centers)
                if new_radius > cur_radius:
                    cur_radius = new_radius
                    current = new_centers
            except Exception:
                pass

        # Local pair-push refinement
        if time.time() - start_time < _DEADLINE - 3.0:
            try:
                refined = _local_pair_push(current, margin=0.005, max_iters=100, step_scale=0.05)
                ref_radius = _compute_radius(refined)
                if ref_radius > cur_radius:
                    cur_radius = ref_radius
                    current = refined
            except Exception:
                pass

        if cur_radius > best_radius:
            best_radius = cur_radius
            best_centers = current.copy()

    # Final polish with high power on the best solution
    if best_centers is not None and time.time() - start_time < _DEADLINE - 3.0:
        for p_polish in [20.0, 30.0]:
            if time.time() - start_time > _DEADLINE - 2.0:
                break
            x0 = best_centers.ravel()
            try:
                res = minimize(
                    lambda x: _energy_and_gradient(x, p_polish, 1.0)[0],
                    x0,
                    jac=lambda x: _energy_and_gradient(x, p_polish, 1.0)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 200,
                        'maxfun': 500,
                        'ftol': 1e-13,
                        'gtol': 1e-10,
                        'disp': False,
                    },
                )
                polished = res.x.reshape(N, 3)
                pol_radius = _compute_radius(polished)
                if pol_radius > best_radius:
                    best_radius = pol_radius
                    best_centers = polished
            except Exception:
                pass

    if best_centers is None:
        # Fallback grid
        m = 5
        ticks = (np.arange(m) + 0.5) / m - CUBE
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers
