import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 300


def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    for attempt in range(12):
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * (0.88 + 0.045 * attempt)
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x, y, r))
        if len(rows) < n_points:
            continue
        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]
        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.04, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]
        d = np.linalg.norm(selected[:, np.newaxis, :] - selected[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)
        if r > best_r:
            best_r = r
            best_pts = selected.copy()
    if best_pts is None:
        spacing = 2.0 / (np.sqrt(n_points) + 1.0)
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]
    return best_pts


def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_pair, min_bound)


def _soft_min_logsumexp(values, T):
    """Stable soft minimum: -T * logsumexp(-values/T)"""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    neg_vals = -values / T
    max_neg = np.max(neg_vals)
    if max_neg > 700:
        return values.min()
    exp_shifted = np.exp(neg_vals - max_neg)
    log_sum = np.log(np.sum(exp_shifted)) + max_neg
    return -T * log_sum


def _objective_and_grad(x, T):
    """
    Objective: minimize -softmin(constraints) where constraints are:
      - pairwise half-distances d_ij/2 for i<j
      - boundary distances 1 - |c_i|
    """
    pts = x.reshape(-1, 2)
    n = len(pts)

    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms

    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    iu = np.triu_indices(n, k=1)
    pair_half = d[iu] / 2.0

    all_vals = np.concatenate([pair_half, boundary])

    obj = -_soft_min_logsumexp(all_vals, T)

    neg_vals_T = -all_vals / T
    max_nv = np.max(neg_vals_T)
    if max_nv > 700:
        min_idx = np.argmin(all_vals)
        grad_all = np.zeros_like(all_vals)
        grad_all[min_idx] = -1.0
    else:
        exp_shifted = np.exp(neg_vals_T - max_nv)
        weights = exp_shifted / np.sum(exp_shifted)
        grad_all = -weights

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)

    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]

    W = np.zeros((n, n))
    n_pairs = len(pair_half)
    W[iu] = grad_all[:n_pairs]
    W[iu[1], iu[0]] = grad_all[:n_pairs]

    weighted_unit = W[:, :, np.newaxis] * unit_vecs
    grad_pairs = 0.5 * np.sum(weighted_unit, axis=1)

    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[n_pairs:][:, np.newaxis]

    grad_pts = grad_pairs + grad_boundary
    grad_obj = grad_pts.ravel()

    return obj, grad_obj


def _delaunay_stress_layout(pts, n_steps=5000, deadline=None):
    """
    Force-directed layout using Delaunay triangulation edges.
    Minimizes stress: sum over edges of (d_ij - target)^2.
    Projects points back into unit disk after each step.
    """
    n = len(pts)
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    
    # Build Delaunay triangulation
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        edge_list = list(edges)
        # Also add boundary edges (connect points near the circle)
        norms = np.linalg.norm(pos, axis=1)
        boundary_idx = np.where(norms > 0.85)[0]
        for i in range(len(boundary_idx)):
            for j in range(i+1, len(boundary_idx)):
                a, b = boundary_idx[i], boundary_idx[j]
                d = np.linalg.norm(pos[a] - pos[b])
                if d < 0.15:  # only connect nearby boundary points
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edges:
                        edges.add((a, b))
                        edge_list.append((a, b))
    except Exception:
        # Fallback: use k-nearest neighbors
        from scipy.spatial import cKDTree
        tree = cKDTree(pos)
        edge_list = []
        for i in range(n):
            _, idx = tree.query(pos[i], k=7)
            for j in idx:
                if j != i:
                    a, b = min(i, j), max(i, j)
                    if (a, b) not in edge_list:
                        edge_list.append((a, b))
    
    if len(edge_list) == 0:
        return best_pos, best_r
    
    # Target edge length: approximate optimal spacing
    target_len = 2.0 * 0.053  # ~2*r where r is best known radius
    
    # Adaptive learning rate
    lr = 0.01
    cooldown = 0.999
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute stress gradient
        forces = np.zeros_like(pos)
        
        for (a, b) in edge_list:
            diff = pos[a] - pos[b]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                continue
            # Stress: want d to be close to target_len
            # Gradient of (d - target)^2 w.r.t. pos[a]: 2*(d-target)/d * (pos[a]-pos[b])
            stress_grad = 2.0 * (d - target_len) / d * diff
            forces[a] += stress_grad
            forces[b] -= stress_grad
        
        # Add wall constraint force
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.05
        for i in range(n):
            if norms[i] > 1.0 - wall_margin:
                excess = norms[i] - (1.0 - wall_margin)
                if excess > 0:
                    f_mag = 10.0 * excess
                    if norms[i] > 1e-10:
                        direction = -pos[i] / norms[i]
                    else:
                        direction = np.array([1.0, 0.0])
                    forces[i] += f_mag * direction
        
        # Update positions
        pos = pos + lr * forces
        
        # Project back into unit disk
        norms = np.linalg.norm(pos, axis=1)
        for i in range(n):
            if norms[i] > 1.0:
                pos[i] = pos[i] / norms[i] * 0.999
        
        # Check improvement periodically
        if step % 100 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
        
        # Decay learning rate
        lr *= cooldown
    
    return best_pos, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Phase 1: Multiple hexagonal initializations with Delaunay stress layout
    best_pts = None
    best_r = 0.0

    for seed in [42, 123, 7, 999, 555]:
        if time.time() > deadline - 40.0:
            break

        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)

        # Run Delaunay stress-minimization layout
        stress_pts, stress_r = _delaunay_stress_layout(
            pts,
            n_steps=3000,
            deadline=deadline - 35.0
        )

        if stress_r > r:
            pts = stress_pts
            r = stress_r

        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Phase 2: Gradient-based refinement with soft-min (annealing schedule)
    if time.time() < deadline - 15.0 and best_pts is not None:
        # More refined temperature schedule
        temperatures = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 5e-05, 2e-05]
        x0 = best_pts.ravel()

        for T in temperatures:
            if time.time() > deadline - 12.0:
                break
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, T)[0],
                    x0,
                    jac=lambda x: _objective_and_grad(x, T)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 200,
                        'ftol': 1e-13,
                        'gtol': 1e-11,
                        'maxls': 60,
                        'maxfun': 500,
                    }
                )
                new_pts = result.x.reshape(-1, 2)
                norms = np.linalg.norm(new_pts, axis=1)
                if np.any(norms >= 1.0) or not np.all(np.isfinite(new_pts)):
                    continue
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
                    x0 = new_pts.ravel()
            except Exception:
                continue

    # Phase 3: Final ultra-low-temperature polish passes
    if time.time() < deadline - 5.0:
        for T_final in [1e-05, 5e-06]:
            if time.time() > deadline - 3.0:
                break
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, T_final)[0],
                    x0,
                    jac=lambda x: _objective_and_grad(x, T_final)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 400,
                        'ftol': 1e-15,
                        'gtol': 1e-13,
                        'maxls': 100,
                        'maxfun': 1000,
                    }
                )
                new_pts = result.x.reshape(-1, 2)
                norms = np.linalg.norm(new_pts, axis=1)
                if np.all(norms < 1.0) and np.all(np.isfinite(new_pts)):
                    new_r = _compute_radius(new_pts)
                    if new_r > best_r:
                        best_r = new_r
                        best_pts = new_pts.copy()
                        x0 = new_pts.ravel()
            except Exception:
                pass

    # Ensure validity
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)

    return best_pts
