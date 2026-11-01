"""Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.

Strategy:
  1. Rigid-body + per-point radial breathing parameterization of a high-symmetry
     seed lattice (FCC-like). This reduces the effective search space by keeping
     the lattice structure intact while allowing global rotation, translation and
     local breathing, preventing the optimizer from deforming the lattice into
     asymmetric local minima.
  2. Soft-min (log-sum-exp) surrogate for the radius with an analytic gradient
     (using only unique pairs i<j to avoid double counting), optimized with
     L-BFGS-B.
  3. Simulated annealing with overlap energy to escape local minima.
  4. Lloyd's algorithm (Voronoi centroid moves) to equalize clearance.
  5. Final high-precision polish.

Entry point: construct() -> (N, 3) array of centres in [-1/2, 1/2]^3.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """Compute the largest common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    if len(c) == 0:
        return 0.0
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _softmin_radius_and_grad(c, t):
    """
    Soft-min radius and its gradient via log-sum-exp.
    Uses only unique pairs (i<j) to avoid double counting.
    softmin(x) = m - t*log(sum(exp(-(x-m)/t)))
    Returns (radius, grad) with grad shape (N,3).
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)

    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_vals = pair_dists / 2.0

    wall_vals = (0.5 - np.abs(c)).ravel()
    all_vals = np.concatenate([pair_vals, wall_vals])

    m = np.min(all_vals)
    exp_vals = np.exp(-(all_vals - m) / t)
    s = np.sum(exp_vals)
    radius = m - t * np.log(s)
    weights = exp_vals / s

    grad = np.zeros_like(c)
    n_pairs = len(pair_vals)
    pair_w = weights[:n_pairs]
    pi, pj = iu
    safe_dists = np.where(pair_dists > 1e-10, pair_dists, 1.0)
    diff_pair = c[pi] - c[pj]
    factor = pair_w / (2.0 * safe_dists)
    np.add.at(grad, pi, -factor[:, np.newaxis] * diff_pair)
    np.add.at(grad, pj, factor[:, np.newaxis] * diff_pair)

    wall_w = weights[n_pairs:].reshape(N, 3)
    signs = np.sign(c)
    signs[np.abs(c) < 1e-10] = 0.0
    grad += wall_w * signs
    return radius, grad


def _fcc_seed():
    """FCC-like seed lattice with 50 points in [-0.5,0.5]^3."""
    pts = []
    # Simple cubic lattice corners + face centers of a 3x3x3 sub-division
    a = 1.0 / 3.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x = -0.5 + i * a
                y = -0.5 + j * a
                z = -0.5 + k * a
                pts.append((x, y, z))
    # face centers
    for i in range(4):
        for j in range(4):
            x = -0.5 + i * a
            y = -0.5 + j * a
            pts.append((x, -0.5 + 0.5 * a, -0.5 + 3 * 0.5 * a))  # placeholder
    # Just take a clean grid of 50
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _breathed_config(base, rot, trans, breath):
    """Apply rigid transform (rotation R, translation t) + per-point radial breathing."""
    c = base @ rot.T + trans
    # radial breathing: scale each point about the cloud centroid
    centroid = c.mean(axis=0)
    r = c - centroid
    # breath[i] scales point i radially
    scale = 1.0 + breath
    c = centroid + r * scale[:, np.newaxis]
    return c


def _rigid_breath_obj(params, base, t_soft):
    """Objective for rigid+breath parameterization."""
    n_b = len(base)
    # params: [3 rot (axis-angle) , 3 trans, n_b breath]
    rot_vec = params[:3]
    trans = params[3:6]
    breath = params[6:6 + n_b]
    # build rotation from axis-angle
    theta = np.linalg.norm(rot_vec)
    if theta < 1e-12:
        R = np.eye(3)
    else:
        axis = rot_vec / theta
        c = np.cos(theta)
        s = np.sin(theta)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
    c = _breathed_config(base, R, trans, breath)
    # penalty if outside cube
    if np.any(c < -0.5 + 1e-9) or np.any(c > 0.5 - 1e-9):
        return 1e8
    r, _ = _softmin_radius_and_grad(c, t_soft)
    return -r


def _repulsion_anneal(c_init, deadline, max_iters=4000):
    """Simulated annealing with overlap energy."""
    c = c_init.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    r_target = best_r + 0.005
    temp = 0.04
    decay = 0.9985
    pert_scale = 0.006

    for it in range(max_iters):
        if time.time() > deadline:
            break
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        overlap_pair = np.maximum(0.0, 2.0 * r_target - dists)
        wall_dists = 0.5 - np.abs(c)
        overlap_wall = np.maximum(0.0, r_target - wall_dists)
        E = np.sum(overlap_pair ** 2) + np.sum(overlap_wall ** 2)

        move = np.random.normal(0.0, pert_scale, size=c.shape)
        c_new = np.clip(c + move, -0.49, 0.49)
        diff_new = c_new[:, np.newaxis, :] - c_new[np.newaxis, :, :]
        dists_new = np.sqrt(np.sum(diff_new * diff_new, axis=-1))
        np.fill_diagonal(dists_new, np.inf)
        overlap_pair_new = np.maximum(0.0, 2.0 * r_target - dists_new)
        overlap_wall_new = np.maximum(0.0, r_target - (0.5 - np.abs(c_new)))
        E_new = np.sum(overlap_pair_new ** 2) + np.sum(overlap_wall_new ** 2)

        dE = E_new - E
        if dE < 0 or np.random.random() < np.exp(-dE / max(temp, 1e-6)):
            c = c_new
            r_check = _compute_radius(c)
            if r_check > best_r:
                best_r = r_check
                best_c = c.copy()
                pert_scale *= 0.96
            else:
                pert_scale = min(pert_scale * 1.02, 0.018)
        else:
            pert_scale = min(pert_scale * 1.03, 0.018)
        temp *= decay
        if it % 300 == 0 and E < 1e-4:
            r_target += 0.002
    return best_c, best_r


def _lloyds(c_init, deadline, max_iters=80):
    """Lloyd's algorithm: move each centre toward its Voronoi cell centroid."""
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    eps = 0.2
    boundary = []
    for ax in range(3):
        for sgn in [-1, 1]:
            p = np.zeros(3)
            p[ax] = sgn * (0.5 + eps)
            boundary.append(p)
    for i in range(3):
        for j in range(i + 1, 3):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    p = np.zeros(3)
                    p[i] = si * (0.5 + eps)
                    p[j] = sj * (0.5 + eps)
                    boundary.append(p)
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                boundary.append(np.array([sx * (0.5 + eps), sy * (0.5 + eps), sz * (0.5 + eps)]))
    boundary = np.array(boundary)

    for it in range(max_iters):
        if time.time() > deadline:
            break
        all_pts = np.vstack([c, boundary])
        try:
            vor = Voronoi(all_pts)
        except Exception:
            break
        # For each of the first N points, find its Voronoi cell and clip to cube
        new_c = c.copy()
        for i in range(N):
            region = vor.regions[vor.point_region[i]]
            if len(region) == 0 or -1 in region:
                continue
            verts = vor.vertices[region]
            # centroid of cell, clip to cube
            centroid = verts.mean(axis=0)
            centroid = np.clip(centroid, -0.49, 0.49)
            new_c[i] = c[i] + 0.5 * (centroid - c[i])
        new_c = np.clip(new_c, -0.49, 0.49)
        r_new = _compute_radius(new_c)
        if r_new >= best_r:
            c = new_c
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
    return best_c, best_r


def _softmin_polish(c_init, deadline, t_start=0.02, t_end=0.001, max_iter=200):
    """L-BFGS-B on soft-min surrogate with annealed temperature."""
    x0 = c_init.flatten()
    t = t_start
    for stage in range(4):
        t = t_start * (t_end / t_start) ** (stage / 3.0)
        def obj(x):
            c = x.reshape(N, 3)
            if np.any(c < -0.5 + 1e-9) or np.any(c > 0.5 - 1e-9):
                return 1e8
            r, _ = _softmin_radius_and_grad(c, t)
            return -r
        def grad(x):
            c = x.reshape(N, 3)
            if np.any(c < -0.5 + 1e-9) or np.any(c > 0.5 - 1e-9):
                return np.ones_like(x)
            _, g = _softmin_radius_and_grad(c, t)
            return -g.ravel()
        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                           bounds=[(-0.49, 0.49)] * (N * 3),
                           options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-9})
            x0 = res.x
        except Exception:
            pass
        if time.time() > deadline:
            break
    c = x0.reshape(N, 3)
    return c


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    best_c = None
    best_r = -1.0

    def consider(c):
        nonlocal best_c, best_r
        if c is None:
            return
        c = np.clip(c, -0.4999, 0.4999)
        if not np.all(np.isfinite(c)):
            return
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()

    base = _fcc_seed()

    # Stage 1: rigid-body + breathing search from several random orientations
    for restart in range(3):
        if time.time() > deadline - 20:
            break
        np.random.seed(100 + restart)
        rot_vec = np.random.normal(0, 0.4, 3)
        trans = np.random.normal(0, 0.02, 3)
        breath = np.random.normal(0, 0.05, len(base))
        params = np.concatenate([rot_vec, trans, breath])
        n_b = len(base)
        bounds = [(-1.0, 1.0)] * 3 + [(-0.1, 0.1)] * 3 + [(-0.3, 0.3)] * n_b
        try:
            res = minimize(_rigid_breath_obj, params, args=(base, 0.02),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 150, 'ftol': 1e-12})
            c = _breathed_config(base,
                                 _axis_angle_to_R(res.x[:3]),
                                 res.x[3:6], res.x[6:6 + n_b])
            consider(c)
        except Exception:
            pass

    # Stage 2: simulated annealing from best so far
    if best_c is None:
        best_c = base.copy()
    if time.time() < deadline - 25:
        c, r = _repulsion_anneal(best_c, deadline - 25)
        consider(c)

    # Stage 3: Lloyd's algorithm
    if time.time() < deadline - 10:
        c, r = _lloyds(best_c, deadline - 10)
        consider(c)

    # Stage 4: soft-min polish
    if time.time() < deadline - 2:
        c = _softmin_polish(best_c, deadline - 2)
        consider(c)

    if best_c is None:
        return base
    return np.clip(best_c, -0.4999, 0.4999)


def _axis_angle_to_R(v):
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    axis = v / theta
    c = np.cos(theta)
    s = np.sin(theta)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)

# EVOLVE-BLOCK-END
