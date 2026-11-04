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

    # Try a few different spacings and rotations
    for attempt in range(14):
        # Spacing estimation
        # Area per circle ~ pi * r^2. Total area ~ pi * 1^2.
        # N * pi * r^2 <= pi => r <= 1/sqrt(N)
        # Hex packing density is pi / (2*sqrt(3)) ~ 0.9069
        # So r ~ 0.95 / sqrt(N)
        base_spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        spacing = base_spacing * (0.95 + 0.015 * attempt)

        k = int(1.0 / spacing) + 5
        rows = []
        # Use a fixed rotation per attempt to break symmetry
        rot_angle = rng.uniform(0, 2 * np.pi)
        cos_a = np.cos(rot_angle)
        sin_a = np.sin(rot_angle)
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                # Apply rotation
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

        # Add noise for attempts > 0
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
        # Fallback
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
        # Sparse case
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
        # Full pairwise
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
        # Sparse pairwise
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
    """Fast force-directed relaxation with adaptive target separation."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)

    # Estimate initial radius
    r_est = best_r

    lr = 0.02
    cooldown = 0.998

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        forces = np.zeros_like(pos)

        # Pairwise repulsion using Hertzian overlap^4 potential
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)

        # Adaptive target separation: try to push to current best radius
        # We use a slightly larger target to encourage growth
        target = 2.0 * (r_est + 0.001) 
        mask = dists < target
        # Safe distances
        safe_dists = np.where(mask, dists, 1.0)

        # Hertzian overlap: (target - dist)^4 when dist < target
        excess = np.where(mask, target - dists, 0.0)
        overlap4 = excess ** 4

        # Force magnitude: derivative of (target-d)^4 / d = 4*(target-d)^3
        force_mag = np.where(mask, 4.0 * excess**3 / safe_dists, 0.0)

        # Unit vectors
        unit_vecs = diff / safe_dists[:, :, np.newaxis]

        # Sum forces
        forces += np.sum(force_mag[:, :, np.newaxis] * unit_vecs, axis=1)

        # Boundary force: push inward if too close to wall
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.03
        for i in range(n):
            if norms[i] > 1.0 - wall_margin:
                excess_b = norms[i] - (1.0 - wall_margin)
                if excess_b > 0:
                    f_mag = 50.0 * excess_b**3
                    if norms[i] > 1e-10:
                        direction = -pos[i] / norms[i]
                    else:
                        direction = np.array([1.0, 0.0])
                    forces[i] += f_mag * direction

        pos = pos + lr * forces

        # Project back into unit disk
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.999

        # Update best periodically
        if step % 50 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
                r_est = r

        lr *= cooldown

    # Final evaluation
    final_r = _compute_radius(pos)
    if final_r > best_r:
        best_r = final_r
        best_pos = pos.copy()

    return best_pos, best_r

def _get_active_edges(pts, max_edges=1500):
    """Get the closest pairs of points using Delaunay triangulation."""
    n = len(pts)
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
        edge_list = list(edges)
    except Exception:
        # Fallback: use k-nearest neighbors
        edge_list = []
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        for i in range(n):
            nn = np.argsort(d[i])[:5]
            for j in nn:
                a, b = min(i, j), max(i, j)
                if (a, b) not in edge_list:
                    edge_list.append((a, b))
    
    if len(edge_list) > max_edges:
        # Sort by distance and keep closest
        pts_arr = np.array(pts)
        edge_dists = [np.linalg.norm(pts_arr[i] - pts_arr[j]) for i, j in edge_list]
        order = np.argsort(edge_dists)
        edge_list = [edge_list[k] for k in order[:max_edges]]
    
    return edge_list

def _local_polish(pts, deadline=None):
    """High-precision local polish using active-set Newton-like iterations."""
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Build active edge list
    edge_list = _get_active_edges(best_pts)
    
    # Multi-temperature polish
    temperatures = [0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000005]
    x0 = best_pts.ravel()
    
    for T in temperatures:
        if deadline is not None and time.time() > deadline:
            break
        try:
            # Recompute edges periodically
            if len(temperatures) > 3:
                edge_list = _get_active_edges(best_pts)
            
            result = minimize(
                lambda x: _objective_and_grad(x, T, edge_list)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T, edge_list)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-15,
                    'gtol': 1e-13,
                    'maxls': 100,
                    'maxfun': 800,
                    'disp': False,
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
    
    # Final high-precision polish with very low temperature
    if deadline is None or time.time() < deadline - 2.0:
        try:
            edge_list = _get_active_edges(best_pts)
            T_final = 1e-6
            result = minimize(
                lambda x: _objective_and_grad(x, T_final, edge_list)[0],
                best_pts.ravel(),
                jac=lambda x: _objective_and_grad(x, T_final, edge_list)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-16,
                    'gtol': 1e-14,
                    'maxls': 150,
                    'maxfun': 1500,
                    'disp': False,
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
    
    # NEW: Active Set Tightening
    # Identify active constraints: boundary points and min-distance pairs
    if deadline is None or time.time() < deadline - 1.0:
        try:
            active_pts = best_pts.copy()
            current_r = _compute_radius(active_pts)
            
            for _ in range(20):
                # Find min constraint
                norms = np.linalg.norm(active_pts, axis=1)
                min_norm_idx = np.argmax(norms)
                min_boundary = 1.0 - norms[min_norm_idx]
                
                d = cdist(active_pts, active_pts)
                np.fill_diagonal(d, np.inf)
                min_pair_val, min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
                min_pair = d[min_pair_val, min_pair_idx] / 2.0
                
                if min_pair < min_boundary:
                    # Pair is limiting
                    i, j = min_pair_val, min_pair_idx
                    diff = active_pts[i] - active_pts[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-10: continue
                    unit_vec = diff / dist
                    # Move i and j apart
                    step_size = 1e-5
                    active_pts[i] += 0.5 * step_size * unit_vec
                    active_pts[j] -= 0.5 * step_size * unit_vec
                else:
                    # Boundary is limiting
                    i = min_norm_idx
                    norm_i = np.linalg.norm(active_pts[i])
                    if norm_i < 1e-10: continue
                    unit_vec = -active_pts[i] / norm_i
                    # Move i inward? No, we want to maximize r.
                    # If boundary is limiting, we should move it OUTWARD? 
                    # Wait, r = min(1-|c|, d/2).
                    # If 1-|c| is the min, increasing r requires decreasing |c|? No.
                    # r is the radius of the circles. The circles must be INSIDE.
                    # So 1-|c| >= r. To increase r, we need to increase 1-|c|? No, that means moving IN.
                    # But if we move in, we might violate pair distances.
                    # Actually, if boundary is the bottleneck, it means the point is too close to the edge.
                    # To allow a larger r, we must move the point INWARD.
                    # But wait, if we move inward, 1-|c| increases.
                    # So yes, move inward.
                    step_size = 1e-5
                    active_pts[i] += step_size * unit_vec # unit_vec is inward (-pos/norm)
                
                # Project into disk
                norms = np.linalg.norm(active_pts, axis=1)
                mask_out = norms > 1.0
                if np.any(mask_out):
                    active_pts[mask_out] = active_pts[mask_out] / norms[mask_out, np.newaxis] * 0.9999

                new_r = _compute_radius(active_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = active_pts.copy()
                else:
                    break
        except Exception:
            pass

    return best_pts, best_r

def _anneal_restart(pts, seed, deadline=None):
    """Perturb the current best and re-relax to escape local minima."""
    rng = np.random.default_rng(seed)
    n = len(pts)
    best_r = _compute_radius(pts)
    
    # Perturbation scale
    scale = 0.005
    
    perturbed = pts + rng.normal(0, scale, (n, 2))
    # Project into disk
    norms = np.linalg.norm(perturbed, axis=1)
    mask_out = norms > 0.999
    if np.any(mask_out):
        perturbed[mask_out] = perturbed[mask_out] / norms[mask_out, np.newaxis] * 0.999
    
    # Force relax
    relaxed_pts, relaxed_r = _force_relax(perturbed, n_steps=800, deadline=deadline)
    
    if relaxed_r > best_r:
        return relaxed_pts, relaxed_r
    
    return pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin for 150s limit

    best_pts = None
    best_r = 0.0

    # Phase 1: Multiple initializations with force relaxation
    seeds = [42, 123, 7, 99, 256, 777]
    for seed in seeds:
        if time.time() > deadline - 60.0:
            break

        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)

        # Force relaxation
        relaxed_pts, relaxed_r = _force_relax(pts, n_steps=1500, deadline=deadline - 50.0)

        if relaxed_r > r:
            pts = relaxed_pts
            r = relaxed_r

        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)

    # Phase 2: Global optimization with soft-min (full pairwise)
    if time.time() < deadline - 25.0:
        temperatures = [0.02, 0.01, 0.005, 0.002, 0.001]
        x0 = best_pts.ravel()

        for T in temperatures:
            if time.time() > deadline - 15.0:
                break
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, T)[0],
                    x0,
                    jac=lambda x: _objective_and_grad(x, T)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 80,
                        'ftol': 1e-13,
                        'gtol': 1e-11,
                        'maxls': 50,
                        'maxfun': 250,
                        'disp': False,
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
                continue

    # Phase 3: Local polish with sparse active-set gradient
    if time.time() < deadline - 15.0 and best_pts is not None:
        polished_pts, polished_r = _local_polish(best_pts, deadline=deadline - 10.0)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()

    # Phase 4: Annealing restarts to escape local minima
    if time.time() < deadline - 10.0 and best_pts is not None:
        for restart_seed in [1001, 2002, 3003, 4004, 5005]:
            if time.time() > deadline - 8.0:
                break
            try:
                new_pts, new_r = _anneal_restart(best_pts, seed=restart_seed, deadline=deadline - 8.0)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
                    # After improvement, do a quick polish
                    if time.time() < deadline - 5.0:
                        polished_pts, polished_r = _local_polish(best_pts, deadline=deadline - 4.0)
                        if polished_r > best_r:
                            best_r = polished_r
                            best_pts = polished_pts.copy()
            except Exception:
                pass

    # Final polish if time remains
    if time.time() < deadline - 3.0 and best_pts is not None:
        try:
            polished_pts, polished_r = _local_polish(best_pts, deadline=deadline - 2.0)
            if polished_r > best_r:
                best_r = polished_r
                best_pts = polished_pts.copy()
        except Exception:
            pass

    # Final validity check
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
    
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)

    # Ensure all points are strictly inside the unit circle
    norms = np.linalg.norm(best_pts, axis=1)
    mask_out = norms >= 0.999999
    if np.any(mask_out):
        best_pts[mask_out] = best_pts[mask_out] / norms[mask_out, np.newaxis] * 0.999999

    return best_pts
