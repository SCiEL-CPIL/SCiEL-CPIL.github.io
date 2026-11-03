import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.distance import cdist

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    
    # Try a range of spacings
    for attempt in range(15):
        # Spacing estimation
        # r_max ~ 0.95/sqrt(N), spacing = 2*r_max
        base_spacing = 2.0 / np.sqrt(n_points) * 0.95
        spacing = base_spacing * (0.92 + 0.015 * attempt)
        
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        
        if len(rows) < n_points:
            continue
            
        pts = np.array(rows)
        
        # Apply a single random rotation to the entire lattice
        angle = rng.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        pts = pts @ rot_matrix.T
        
        # Add small noise for non-first attempts to break symmetry further
        if attempt > 0:
            noise_scale = spacing * 0.01
            pts = pts + rng.normal(0, noise_scale, pts.shape)
            
        # Select points inside the unit circle (with margin)
        norms = np.linalg.norm(pts, axis=1)
        # We want to select points that are inside 1 - spacing/2
        # But we can select more and then pick the best N
        # Let's select all that are within 1.0
        mask = norms < 1.0
        pts_inside = pts[mask]
        
        if len(pts_inside) < n_points:
            continue
            
        # Sort by distance to origin? Or just pick random N?
        # Picking closest to origin is good for packing density
        order = np.argsort(norms[mask], kind="stable")
        selected = pts_inside[order[:n_points]]
        
        # Compute quality
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
        spacing = 2.0 / np.sqrt(n_points) * 0.95
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

def _ring_init(n_points, seed=0):
    """Generate concentric ring initialization with jitter."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    
    for attempt in range(10):
        # Estimate spacing
        r_est = 1.0 / np.sqrt(n_points) * (0.95 + 0.02 * attempt)
        spacing = 2.0 * r_est * 0.866 # Hex spacing approx
        
        points = []
        # Center
        points.append([0.0, 0.0])
        
        ring_idx = 1
        while len(points) < n_points:
            ring_r = ring_idx * spacing * 0.866
            if ring_r >= 1.0 - r_est:
                break
            n_ring = max(6, int(2.0 * np.pi * ring_r / spacing))
            n_ring = min(n_ring, n_points - len(points))
            
            angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
            angles += rng.uniform(0, 2*np.pi/n_ring)
            
            for a in angles:
                x = ring_r * np.cos(a)
                y = ring_r * np.sin(a)
                if np.hypot(x, y) < 1.0 - r_est:
                    points.append([x, y])
            
            ring_idx += 1
            
        if len(points) < n_points:
            # Fill with random
            while len(points) < n_points:
                x = rng.uniform(-0.9, 0.9)
                y = rng.uniform(-0.9, 0.9)
                if np.hypot(x, y) < 0.95:
                    points.append([x, y])
                    
        pts = np.array(points[:n_points])
        pts = pts + rng.normal(0, r_est * 0.05, pts.shape)
        
        norms = np.linalg.norm(pts, axis=1)
        mask = norms < 1.0
        if np.sum(mask) < n_points:
            continue
        pts = pts[mask][:n_points]
        
        d = cdist(pts, pts)
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
    """Compute the packing radius for given points using KD-tree for speed."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    min_pair = dists[:, 1].min() / 2.0
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
    Objective: minimize -softmin(constraints)
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

def _force_relax(pts, n_steps=1000, deadline=None):
    """Fast force-directed relaxation with Hertzian (overlap^4) repulsion."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)
    
    r_est = best_r
    lr = 0.005
    cooldown = 0.9995
    
    tree = cKDTree(pos)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
            
        forces = np.zeros_like(pos)
        
        # Use KD-tree to find nearby pairs (within 2.5 * r_est)
        search_radius = 2.5 * r_est
        pairs = tree.query_pairs(search_radius)
        
        for (i, j) in pairs:
            diff = pos[i] - pos[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            target = 2.0 * r_est
            if dist < target:
                # Hertzian: force proportional to overlap^4
                overlap = target - dist
                force_mag = 10.0 * (overlap ** 4) / (target ** 3)
                unit_vec = diff / dist
                forces[i] += force_mag * unit_vec
                forces[j] -= force_mag * unit_vec
        
        # Boundary forces
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.08
        for i in range(n):
            if norms[i] > 1.0 - wall_margin:
                excess_b = norms[i] - (1.0 - wall_margin)
                if excess_b > 0:
                    f_mag = 20.0 * (excess_b ** 2)
                    if norms[i] > 1e-10:
                        direction = -pos[i] / norms[i]
                    else:
                        direction = np.array([1.0, 0.0])
                    forces[i] += f_mag * direction
        
        pos = pos + lr * forces
        
        # Clamp to unit circle
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.9999
            
        # Update tree periodically
        if step % 20 == 0:
            tree = cKDTree(pos)
        
        if step % 30 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
                r_est = r
                lr = min(0.02, lr * 1.02)
                
        lr *= cooldown
        
    return best_pos, best_r

def _delaunay_perturb(pts, n_perturb=5, seed=0, deadline=None):
    """Perturb the configuration by swapping nearby point pairs identified via Delaunay triangulation."""
    rng = np.random.default_rng(seed)
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    
    for attempt in range(n_perturb):
        if deadline is not None and time.time() > deadline:
            break
        
        # Build Delaunay triangulation
        try:
            tri = Delaunay(pos)
        except Exception:
            continue
        
        # Get edges from triangulation
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edges.add((min(simplex[i], simplex[j]), max(simplex[i], simplex[j])))
        
        if len(edges) == 0:
            continue
        
        # Pick a random edge and swap the two points slightly
        edge_list = list(edges)
        i, j = edge_list[rng.integers(len(edge_list))]
        
        # Compute midpoint and swap positions with small random displacement
        mid = (pos[i] + pos[j]) / 2.0
        perp = np.array([-(pos[j][1] - pos[i][1]), pos[j][0] - pos[i][0]])
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 1e-10:
            perp = perp / perp_norm
        
        # Swap with a small perturbation
        new_i = pos[j] + rng.normal(0, 0.002, 2)
        new_j = pos[i] + rng.normal(0, 0.002, 2)
        
        # Ensure they stay inside the unit circle
        if np.linalg.norm(new_i) < 0.999 and np.linalg.norm(new_j) < 0.999:
            pos[i] = new_i
            pos[j] = new_j
        
        # Relax after perturbation
        relaxed_pts, relaxed_r = _force_relax(pos, n_steps=200, deadline=deadline)
        
        if relaxed_r > best_r:
            best_r = relaxed_r
            best_pos = relaxed_pts.copy()
            pos = relaxed_pts.copy()
        else:
            # Revert
            pos = best_pos.copy()
    
    return best_pos, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin: evaluator kills at 150s
    
    best_pts = None
    best_r = 0.0
    
    # Phase 1: Initialization and Force Relaxation
    inits = [
        _hex_init(N, seed=42),
        _hex_init(N, seed=123),
        _ring_init(N, seed=7),
        _hex_init(N, seed=999),
        _ring_init(N, seed=555)
    ]
    
    for idx, pts in enumerate(inits):
        if time.time() > deadline - 50.0:
            break
            
        r = _compute_radius(pts)
        
        # Force relaxation
        relaxed_pts, relaxed_r = _force_relax(pts, n_steps=1200, deadline=deadline - 40.0)
        
        if relaxed_r > r:
            pts = relaxed_pts
            r = relaxed_r
            
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)
        
    # Phase 2: Delaunay-based perturbation to escape local minima
    if time.time() < deadline - 30.0:
        perturbed_pts, perturbed_r = _delaunay_perturb(
            best_pts, n_perturb=8, seed=777, deadline=deadline - 25.0
        )
        if perturbed_r > best_r:
            best_r = perturbed_r
            best_pts = perturbed_pts.copy()
    
    # Phase 3: Global Optimization with Soft-Min
    if time.time() < deadline - 20.0:
        temperatures = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        x0 = best_pts.ravel()
        
        for T in temperatures:
            if time.time() > deadline - 10.0:
                break
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, T)[0],
                    x0,
                    jac=lambda x: _objective_and_grad(x, T)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 150,
                        'ftol': 1e-14,
                        'gtol': 1e-12,
                        'maxls': 60,
                        'maxfun': 400,
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
                
    # Phase 4: Final Polish with High Precision
    if time.time() < deadline - 5.0 and best_pts is not None:
        temperatures = [0.00002, 0.00001, 0.000005]
        x0 = best_pts.ravel()
        
        for T in temperatures:
            if time.time() > deadline - 2.0:
                break
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, T)[0],
                    x0,
                    jac=lambda x: _objective_and_grad(x, T)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 200,
                        'ftol': 1e-15,
                        'gtol': 1e-13,
                        'maxls': 80,
                        'maxfun': 600,
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
                    
    # Final validity check
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)
        
    return best_pts
