import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    for attempt in range(10):
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * (0.90 + 0.05 * attempt)
        k = int(1.0 / spacing) + 4
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
            selected = selected + rng.normal(0, spacing * 0.03, selected.shape)
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
        k = int(1.0 / spacing) + 4
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


def _ring_init(n_points, seed=0):
    """Generate concentric ring initialization with jitter."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    
    for attempt in range(8):
        # Estimate number of rings
        # Area per circle ~ pi*r^2, total area = pi
        # n * pi * r^2 <= pi => r <= 1/sqrt(n)
        # Circumference of ring k ~ 2*pi*k*r_spacing
        # Points on ring k ~ 2*pi*k*r_spacing / (2*r) where r is circle radius
        r_est = 1.0 / np.sqrt(n_points) * (0.95 + 0.02 * attempt)
        
        points = []
        # Center point
        points.append([0.0, 0.0])
        
        ring_idx = 1
        while len(points) < n_points:
            # Ring radius
            ring_r = ring_idx * r_est * 2.0 * 0.577  # hex spacing factor
            if ring_r >= 1.0 - r_est:
                break
            # Number of points on this ring
            # Circumference / (2*r_est) approximately
            n_ring = max(6, int(2.0 * np.pi * ring_r / (2.0 * r_est)))
            # Adjust to not exceed remaining
            remaining = n_points - len(points)
            n_ring = min(n_ring, remaining)
            
            angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False) + rng.uniform(0, 2*np.pi/n_ring)
            for a in angles:
                x = ring_r * np.cos(a)
                y = ring_r * np.sin(a)
                if np.hypot(x, y) < 1.0 - r_est:
                    points.append([x, y])
            
            ring_idx += 1
        
        if len(points) < n_points:
            # Fill remaining with random points
            while len(points) < n_points:
                x = rng.uniform(-0.95, 0.95)
                y = rng.uniform(-0.95, 0.95)
                if np.hypot(x, y) < 1.0 - r_est:
                    points.append([x, y])
        
        pts = np.array(points[:n_points])
        
        # Add small jitter
        jitter = rng.normal(0, r_est * 0.05, pts.shape)
        pts = pts + jitter
        
        # Ensure inside
        norms = np.linalg.norm(pts, axis=1)
        mask = norms < 1.0 - r_est * 0.5
        if np.sum(mask) < n_points:
            continue
        pts = pts[mask][:n_points]
        
        # Compute quality
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(pts, axis=1).max()
        r = min(min_pair, min_bound)
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    if best_pts is None:
        return _hex_init(n_points, seed)
    return best_pts


def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    sq_norms = np.sum(pts * pts, axis=1)
    cross = pts @ pts.T
    sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * cross
    np.maximum(sq_dists, 0.0, out=sq_dists)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair = np.sqrt(sq_dists.min()) / 2.0
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
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
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
    
    grad_obj = np.zeros_like(x)
    n_pairs = len(pair_half)
    
    safe_dists = np.where(d > 1e-10, d, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    W = np.zeros((n, n))
    W[iu] = grad_all[:n_pairs]
    W[iu[1], iu[0]] = grad_all[:n_pairs]
    
    grad_pairs = 0.5 * np.einsum('ij,ijc->ic', W, unit_vecs)
    
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[n_pairs:][:, np.newaxis]
    
    grad_pts = grad_pairs + grad_boundary
    grad_obj = grad_pts.ravel()
    
    return obj, grad_obj


def _active_set_newton(pts, n_iter=200, tol=1e-12):
    """
    Newton-like local solve on the active constraint set.
    Identify points involved in binding constraints and solve a small
    quadratic program to increase the minimum constraint.
    """
    pts = pts.copy()
    n = len(pts)
    
    for it in range(n_iter):
        norms = np.linalg.norm(pts, axis=1)
        boundary = 1.0 - norms
        
        sq_norms = np.sum(pts * pts, axis=1)
        cross = pts @ pts.T
        sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * cross
        np.maximum(sq_dists, 0.0, out=sq_dists)
        np.fill_diagonal(sq_dists, np.inf)
        dists = np.sqrt(sq_dists)
        pair_half = dists / 2.0
        
        min_pair_val = pair_half.min()
        min_boundary_val = boundary.min()
        min_val = min(min_pair_val, min_boundary_val)
        
        # Find active set: constraints within 2% of minimum
        threshold = min_val + 0.02 * max(min_val, 1e-6)
        
        active_points = set()
        active_pairs = []
        
        for i in range(n):
            for j in range(i+1, n):
                if pair_half[i, j] < threshold:
                    active_pairs.append((i, j))
                    active_points.add(i)
                    active_points.add(j)
        
        for i in range(n):
            if boundary[i] < threshold:
                active_points.add(i)
        
        if len(active_points) < 2 or len(active_points) > 50:
            # Too many or too few active points; use simple nudge
            if min_pair_val < min_boundary_val:
                idx = np.unravel_index(np.argmin(pair_half), pair_half.shape)
                i, j = idx
                if i == j:
                    continue
                vec = pts[i] - pts[j]
                dist = np.linalg.norm(vec)
                if dist < 1e-10:
                    continue
                unit = vec / dist
                step = 0.0003
                pts[i] += step * unit
                pts[j] -= step * unit
            else:
                i = np.argmin(boundary)
                norm_i = norms[i]
                if norm_i < 1e-10:
                    continue
                unit = pts[i] / norm_i
                pts[i] -= 0.0003 * unit
            
            new_norms = np.linalg.norm(pts, axis=1)
            mask = new_norms >= 1.0
            if np.any(mask):
                pts[mask] *= 0.999 / new_norms[mask, np.newaxis]
            continue
        
        # Solve small QP for active points
        active_idx = list(active_points)
        n_act = len(active_idx)
        
        # Build gradient and Hessian approximation
        # For each active point, compute the net force from active constraints
        forces = np.zeros((n_act, 2))
        
        for (i, j) in active_pairs:
            if i in active_idx and j in active_idx:
                ii = active_idx.index(i)
                jj = active_idx.index(j)
                vec = pts[i] - pts[j]
                dist = np.linalg.norm(vec)
                if dist < 1e-10:
                    continue
                unit = vec / dist
                # Weight by how close to minimum
                w = max(0, 1.0 - (pair_half[i,j] - min_val) / max(threshold - min_val, 1e-10))
                forces[ii] += 0.5 * w * unit
                forces[jj] -= 0.5 * w * unit
        
        for i in active_idx:
            ii = active_idx.index(i)
            if boundary[i] < threshold:
                norm_i = norms[i]
                if norm_i > 1e-10:
                    w = max(0, 1.0 - (boundary[i] - min_val) / max(threshold - min_val, 1e-10))
                    forces[ii] -= w * pts[i] / norm_i
        
        # Step: move each active point in the direction of the force
        step_size = 0.0005
        for ii, idx in enumerate(active_idx):
            pts[idx] += step_size * forces[ii]
        
        # Ensure validity
        new_norms = np.linalg.norm(pts, axis=1)
        mask = new_norms >= 1.0
        if np.any(mask):
            pts[mask] *= 0.999 / new_norms[mask, np.newaxis]
        
        # Check convergence
        if it > 10:
            prev_min = min_val
            new_norms2 = np.linalg.norm(pts, axis=1)
            new_boundary = 1.0 - new_norms2
            sq_norms2 = np.sum(pts * pts, axis=1)
            cross2 = pts @ pts.T
            sq_dists2 = sq_norms2[:, np.newaxis] + sq_norms2[np.newaxis, :] - 2.0 * cross2
            np.maximum(sq_dists2, 0.0, out=sq_dists2)
            np.fill_diagonal(sq_dists2, np.inf)
            new_pair = np.sqrt(sq_dists2).min() / 2.0
            new_min = min(new_pair, new_boundary.min())
            if abs(new_min - prev_min) < tol:
                break
    
    return pts


def _tighten_min_constraint(pts, n_iter=50, step_size=0.001):
    """Directly push the most constrained points to increase the minimum constraint."""
    pts = pts.copy()
    for _ in range(n_iter):
        norms = np.linalg.norm(pts, axis=1)
        boundary = 1.0 - norms
        
        sq_norms = np.sum(pts * pts, axis=1)
        cross = pts @ pts.T
        sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * cross
        np.maximum(sq_dists, 0.0, out=sq_dists)
        np.fill_diagonal(sq_dists, np.inf)
        pair_half = np.sqrt(sq_dists) / 2.0
        
        min_pair_idx = np.unravel_index(np.argmin(pair_half), pair_half.shape)
        min_boundary_idx = np.argmin(boundary)
        
        min_pair_val = pair_half[min_pair_idx]
        min_boundary_val = boundary[min_boundary_idx]
        
        if min_pair_val < min_boundary_val:
            i, j = min_pair_idx
            if i == j:
                continue
            vec = pts[i] - pts[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                continue
            unit = vec / dist
            pts[i] += step_size * unit
            pts[j] -= step_size * unit
        else:
            i = min_boundary_idx
            norm_i = norms[i]
            if norm_i < 1e-10:
                continue
            unit = pts[i] / norm_i
            pts[i] -= step_size * unit
        
        new_norms = np.linalg.norm(pts, axis=1)
        mask = new_norms >= 1.0
        if np.any(mask):
            pts[mask] *= 0.999 / new_norms[mask, np.newaxis]
    
    return pts


def _anneal_restart(pts, rng, scale=0.01):
    """Perturb the solution slightly and return."""
    perturbed = pts + rng.normal(0, scale, pts.shape)
    norms = np.linalg.norm(perturbed, axis=1)
    mask = norms < 1.0
    perturbed = perturbed[mask]
    if len(perturbed) < N:
        return pts  # Return original if perturbation invalidates too many
    return perturbed[:N]


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pts = None
    best_r = -1.0
    rng = np.random.default_rng(12345)
    
    # Phase 1: Diverse initialization
    init_candidates = []
    
    # Hex lattice with multiple seeds
    for seed in [42, 0, 1, 2, 3]:
        if time.time() > deadline - 90.0:
            break
        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)
        init_candidates.append((r, pts))
    
    # Ring initialization
    if time.time() < deadline - 90.0:
        for seed in [10, 20]:
            pts = _ring_init(N, seed=seed)
            r = _compute_radius(pts)
            init_candidates.append((r, pts))
    
    # Sort by quality
    init_candidates.sort(key=lambda x: -x[0])
    best_r = init_candidates[0][0]
    best_pts = init_candidates[0][1].copy()
    
    # Phase 2: Temperature annealing optimization from best initial
    x0 = best_pts.ravel()
    
    # Temperature schedule: more steps at intermediate T
    temperatures = [0.03, 0.015, 0.008, 0.004, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
    
    for T in temperatures:
        if time.time() > deadline - 30.0:
            break
        try:
            maxiter = 400 if T <= 0.001 else 250
            maxfun = 1200 if T <= 0.001 else 700
            result = minimize(
                lambda x: _objective_and_grad(x, T)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': maxiter,
                    'ftol': 1e-13,
                    'gtol': 1e-11,
                    'maxls': 100,
                    'maxfun': maxfun,
                }
            )
            if time.time() > deadline - 25.0:
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
        except Exception:
            continue
    
    # Phase 3: Anneal restarts - try perturbing and re-optimizing
    if time.time() < deadline - 40.0:
        for restart in range(3):
            if time.time() > deadline - 35.0:
                break
            perturbed = _anneal_restart(best_pts, rng, scale=0.005 * (restart + 1))
            x_pert = perturbed.ravel()
            
            # Quick optimization at medium T
            T_restart = 0.003
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, T_restart)[0],
                    x_pert,
                    jac=lambda x: _objective_and_grad(x, T_restart)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 200,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxls': 80,
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
    
    # Phase 4: Active set Newton refinement
    if time.time() < deadline - 20.0:
        try:
            newton_pts = _active_set_newton(best_pts, n_iter=100, tol=1e-12)
            norms = np.linalg.norm(newton_pts, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(newton_pts)):
                newton_r = _compute_radius(newton_pts)
                if newton_r > best_r:
                    best_r = newton_r
                    best_pts = newton_pts.copy()
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 15.0:
        try:
            T_final = 0.000005
            result = minimize(
                lambda x: _objective_and_grad(x, T_final)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T_final)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-14,
                    'gtol': 1e-13,
                    'maxls': 150,
                    'maxfun': 6000,
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
    
    # Phase 6: Final tighten + Newton
    if time.time() < deadline - 8.0:
        try:
            tightened = _tighten_min_constraint(best_pts, n_iter=80, step_size=0.0003)
            norms = np.linalg.norm(tightened, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(tightened)):
                tight_r = _compute_radius(tightened)
                if tight_r > best_r:
                    best_r = tight_r
                    best_pts = tightened.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 4.0:
        try:
            newton_pts = _active_set_newton(best_pts, n_iter=50, tol=1e-13)
            norms = np.linalg.norm(newton_pts, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(newton_pts)):
                newton_r = _compute_radius(newton_pts)
                if newton_r > best_r:
                    best_r = newton_r
                    best_pts = newton_pts.copy()
        except Exception:
            pass
    
    # Ensure validity
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask):
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)
    
    return best_pts
