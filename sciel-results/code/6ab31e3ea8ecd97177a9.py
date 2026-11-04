import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0

    for attempt in range(14):
        base_spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        spacing = base_spacing * (0.95 + 0.015 * attempt)

        k = int(1.0 / spacing) + 5
        rows = []
        rot_angle = rng.uniform(0, 2 * np.pi)
        cos_a = np.cos(rot_angle)
        sin_a = np.sin(rot_angle)
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                r = np.hypot(x_rot, y_rot)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x_rot, y_rot, r))

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

        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)

        if r > best_r:
            best_r = r
            best_pts = selected.copy()

    if best_pts is None:
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
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
    d = cdist(pts, pts)
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


def _get_delaunay_edges(pts):
    """Get Delaunay triangulation edges as a list of (i,j) pairs."""
    try:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback: all pairs (expensive but safe)
        n = len(pts)
        return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _objective_and_grad(x, T, edge_indices=None):
    """
    Objective: minimize -softmin(constraints)
    If edge_indices is provided, only consider those pairs for the pairwise term.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)

    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms

    # Pairwise distances
    if edge_indices is None:
        d = cdist(pts, pts)
        iu = np.triu_indices(n, k=1)
        pair_half = d[iu] / 2.0
    else:
        pair_half = np.array([np.linalg.norm(pts[i] - pts[j]) / 2.0 for i, j in edge_indices])

    all_vals = np.concatenate([pair_half, boundary])

    obj = -_soft_min_logsumexp(all_vals, T)

    # Gradient
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

    grad_pts = np.zeros_like(pts)

    # Boundary gradient
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[len(pair_half):][:, np.newaxis]
    grad_pts += grad_boundary

    # Pairwise gradient
    if edge_indices is None:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]

        W = np.zeros((n, n))
        n_pairs = len(pair_half)
        iu = np.triu_indices(n, k=1)
        W[iu] = grad_all[:n_pairs]
        W[iu[1], iu[0]] = grad_all[:n_pairs]

        weighted_unit = W[:, :, np.newaxis] * unit_vecs
        grad_pairs = 0.5 * np.sum(weighted_unit, axis=1)
        grad_pts += grad_pairs
    else:
        for k, (i, j) in enumerate(edge_indices):
            w = grad_all[k]
            if w == 0:
                continue
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                continue
            unit_vec = diff / d
            grad_pts[i] += 0.5 * w * unit_vec
            grad_pts[j] -= 0.5 * w * unit_vec

    return obj, grad_pts.ravel()


def _force_relax(pts, n_steps=2000, deadline=None):
    """Fast force-directed relaxation with adaptive target separation using spatial hashing."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)

    r_est = best_r

    lr = 0.02
    cooldown = 0.998

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        forces = np.zeros_like(pos)

        target = 2.0 * (r_est + 0.001)

        cell_size = max(target, 0.01)
        grid = {}
        for i in range(n):
            cx = int(pos[i, 0] / cell_size)
            cy = int(pos[i, 1] / cell_size)
            key = (cx, cy)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)

        for i in range(n):
            cx = int(pos[i, 0] / cell_size)
            cy = int(pos[i, 1] / cell_size)

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (cx + dx, cy + dy)
                    if key not in grid:
                        continue
                    for j in grid[key]:
                        if j <= i:
                            continue
                        diff = pos[i] - pos[j]
                        dist = np.linalg.norm(diff)
                        if dist < target and dist > 1e-10:
                            excess = target - dist
                            force_mag = 4.0 * excess**3 / dist
                            unit_vec = diff / dist
                            forces[i] += force_mag * unit_vec
                            forces[j] -= force_mag * unit_vec

        # Boundary force
        norms = np.linalg.norm(pos, axis=1)
        boundary_force = 0.0
        for i in range(n):
            if norms[i] > 1.0 - r_est:
                excess = norms[i] - (1.0 - r_est)
                force_mag = 4.0 * excess**3 / max(norms[i], 1e-10)
                unit_vec = pos[i] / norms[i]
                forces[i] -= force_mag * unit_vec

        pos += lr * forces
        lr *= cooldown

        # Check validity
        norms = np.linalg.norm(pos, axis=1)
        if np.any(norms >= 1.0) or not np.all(np.isfinite(pos)):
            pos = best_pos.copy()
            continue

        r = _compute_radius(pos)
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
            r_est = best_r

    return best_pos, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Initial hexagonal packing with multiple seeds
    best_pts = None
    best_r = 0.0
    for seed in [42, 0, 123, 7]:
        if time.time() > deadline - 40.0:
            break
        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = _hex_init(N, seed=42)
        best_r = _compute_radius(best_pts)

    # Force relaxation pass
    if time.time() < deadline - 30.0:
        best_pts, best_r = _force_relax(best_pts, n_steps=3000, deadline=deadline - 30.0)

    x0 = best_pts.ravel()

    # Get Delaunay edges for sparse optimization
    edges = _get_delaunay_edges(best_pts)

    # Temperature annealing schedule
    temperatures = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]

    for idx, T in enumerate(temperatures):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 15.0:
            break

        if idx < 3:
            maxiter = 400
            maxfun = 1000
        elif idx < 5:
            maxiter = 300
            maxfun = 800
        else:
            maxiter = 200
            maxfun = 500

        try:
            result = minimize(
                lambda x: _objective_and_grad(x, T, edges)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T, edges)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': maxiter,
                    'ftol': 1e-13,
                    'gtol': 1e-11,
                    'maxls': 50,
                    'maxfun': maxfun,
                }
            )
            elapsed = time.time() - start_time
            if elapsed > deadline - 10.0:
                break
            new_pts = result.x.reshape(-1, 2)
            norms = np.linalg.norm(new_pts, axis=1)
            if np.any(norms >= 1.0) or not np.all(np.isfinite(new_pts)):
                continue
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
                x0 = new_pts.ravel()
                # Update Delaunay edges
                edges = _get_delaunay_edges(best_pts)
        except Exception:
            continue

    # Final high-precision polish with very low temperature
    if time.time() < deadline - 8.0:
        try:
            T_final = 0.00002
            edges = _get_delaunay_edges(best_pts)
            result = minimize(
                lambda x: _objective_and_grad(x, T_final, edges)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T_final, edges)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxls': 100,
                    'maxfun': 1500,
                }
            )
            new_pts = result.x.reshape(-1, 2)
            norms = np.linalg.norm(new_pts, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(new_pts)):
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass

    # One more polish pass if time allows
    if time.time() < deadline - 5.0:
        try:
            T_final2 = 0.00001
            edges = _get_delaunay_edges(best_pts)
            result = minimize(
                lambda x: _objective_and_grad(x, T_final2, edges)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T_final2, edges)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-14,
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
        except Exception:
            pass

    # Ensure validity
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)

    return best_pts
