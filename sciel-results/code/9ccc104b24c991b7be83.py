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
    
    # Try a range of spacings and rotations
    # Optimal spacing for N circles in unit disk:
    # r_max ~ 0.95/sqrt(N), so spacing = 2*r_max ~ 1.9/sqrt(N)
    for attempt in range(15):
        # Spacing: vary from slightly tight to slightly loose
        base_spacing = 2.0 / np.sqrt(n_points) * 0.95
        spacing = base_spacing * (0.92 + 0.015 * attempt)
        
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                # Add a small random rotation to break symmetry
                angle = rng.uniform(0, 2*np.pi)
                x_rot = x * np.cos(angle) - y * np.sin(angle)
                y_rot = x * np.sin(angle) + y * np.cos(angle)
                r = np.hypot(x_rot, y_rot)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x_rot, y_rot, r))
        
        if len(rows) < n_points:
            continue
            
        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]
        
        # Add noise for non-first attempts
        if attempt > 0:
            noise_scale = spacing * 0.03
            selected = selected + rng.normal(0, noise_scale, selected.shape)
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

def _random_repulsive_init(n_points, seed=0):
    """Initialize with random points and push them apart."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.8, 0.8, size=(n_points, 2))
    
    # Simple repulsion to avoid overlapping starts
    for _ in range(50):
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        # Find closest pairs and push them apart
        min_idx = np.unravel_index(np.argmin(d), d.shape)
        if d[min_idx] < 0.05:
            i, j = min_idx
            diff = pts[i] - pts[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-6:
                diff = diff / norm
                pts[i] += 0.01 * diff
                pts[j] -= 0.01 * diff
                # Keep inside
                pts[i] = np.clip(pts[i], -0.95, 0.95)
                pts[j] = np.clip(pts[j], -0.95, 0.95)
                
    # Project inside
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 0.95
    pts[mask] = pts[mask] / norms[mask, np.newaxis] * 0.95
    
    return pts

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
        n_pairs = len(pair_half)
    else:
        # Sparse case
        # edge_indices is a list of (i, j)
        pair_half = np.array([np.linalg.norm(pts[i] - pts[j]) / 2.0 for i, j in edge_indices])
        n_pairs = len(pair_half)
        
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
            if w == 0: continue
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-10: continue
            unit_vec = diff / d
            grad_pts[i] += 0.5 * w * unit_vec
            grad_pts[j] -= 0.5 * w * unit_vec
            
    return obj, grad_pts.ravel()

def _force_relax(pts, n_steps=2000, deadline=None):
    """Fast force-directed relaxation with adaptive parameters."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)
    
    # Estimate initial radius
    r_est = best_r
    
    lr = 0.02
    cooldown = 0.998
    wall_margin = 0.08
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
            
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Force is proportional to (2*r_est - dist) if dist < 2*r_est
        target = 2.0 * r_est
        mask = dists < target
        
        # Vectorized force calculation
        safe_dists = np.where(mask, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        excess = np.where(mask, target - dists, 0.0)
        force_mag = 3.0 * excess / safe_dists
        
        forces += np.sum(force_mag[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Boundary force
        norms = np.linalg.norm(pos, axis=1)
        for i in range(n):
            if norms[i] > 1.0 - wall_margin:
                excess_b = norms[i] - (1.0 - wall_margin)
                if excess_b > 0:
                    f_mag = 15.0 * excess_b
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
            
        # Update best more frequently
        if step % 20 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
                r_est = r # Update estimate for force strength
                # Adjust learning rate based on improvement
                lr = min(0.05, lr * 1.1)
                
        lr *= cooldown
        
    return best_pos, best_r

def _polish_with_lagrange(pts, deadline=None):
    """Polish using active set / Lagrange multiplier approach on constrained pairs."""
    pos = pts.copy().astype(np.float64)
    n = len(pos)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    
    # Identify active constraints: pairs with distance close to min, and boundary contacts
    for iteration in range(50):
        if deadline is not None and time.time() > deadline:
            break
        
        r = _compute_radius(pos)
        if r <= best_r:
            break
        best_r = r
        best_pos = pos.copy()
        
        # Find active pairs: those with distance <= 2*r + small_tol
        d = cdist(pos, pos)
        np.fill_diagonal(d, np.inf)
        
        # Active pairwise constraints
        active_tol = 2.0 * r + 1e-8
        active_pairs = np.argwhere(d <= active_tol)
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        
        # Active boundary constraints
        norms = np.linalg.norm(pos, axis=1)
        active_boundary = np.where(norms >= 1.0 - r - 1e-8)[0]
        
        # Build linear system for Lagrange multipliers (simplified)
        # For each active constraint, we want to adjust points to equalize distances
        # This is a least-squares problem
        
        # Simple approach: for each active pair, move them apart slightly
        # For each active boundary point, move it inward slightly
        step_size = 0.001 * r
        
        for i, j in active_pairs:
            diff = pos[i] - pos[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            unit_vec = diff / dist
            # Move apart
            pos[i] += step_size * unit_vec
            pos[j] -= step_size * unit_vec
        
        for i in active_boundary:
            norm = np.linalg.norm(pos[i])
            if norm < 1e-10:
                continue
            unit_vec = pos[i] / norm
            # Move inward
            pos[i] -= step_size * unit_vec
        
        # Project back into unit disk
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.999
            
    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pts = None
    best_r = 0.0
    
    # Phase 1: Initialization and Force Relaxation
    # Use multiple initialization strategies
    inits = [
        _hex_init(N, seed=42),
        _hex_init(N, seed=123),
        _random_repulsive_init(N, seed=7),
        _hex_init(N, seed=999),
        _hex_init(N, seed=555)
    ]
    
    for pts in inits:
        if time.time() > deadline - 40.0:
            break
            
        r = _compute_radius(pts)
        
        # Force relaxation
        relaxed_pts, relaxed_r = _force_relax(pts, n_steps=1200, deadline=deadline - 30.0)
        
        if relaxed_r > r:
            pts = relaxed_pts
            r = relaxed_r
            
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)
        
    # Phase 2: Global Optimization with Soft-Min
    if time.time() < deadline - 15.0:
        temperatures = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
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
                
    # Phase 3: Local Polish with Sparse Gradient
    if time.time() < deadline - 5.0 and best_pts is not None:
        # Build Delaunay triangulation to get sparse edges
        try:
            tri = Delaunay(best_pts)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
            edge_list = list(edges)
        except Exception:
            edge_list = []
            
        if edge_list:
            # We will run a few iterations of L-BFGS-B with sparse gradient
            temperatures = [0.0001, 0.00005, 0.00002, 0.00001, 0.000005]
            x0 = best_pts.ravel()
            
            for T in temperatures:
                if time.time() > deadline - 2.0:
                    break
                try:
                    result = minimize(
                        lambda x: _objective_and_grad(x, T, edge_list)[0],
                        x0,
                        jac=lambda x: _objective_and_grad(x, T, edge_list)[1],
                        method='L-BFGS-B',
                        options={
                            'maxiter': 250,
                            'ftol': 1e-15,
                            'gtol': 1e-13,
                            'maxls': 100,
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
                    
    # Phase 4: Active Set Polish
    if time.time() < deadline - 1.0 and best_pts is not None:
        polished_pts, polished_r = _polish_with_lagrange(best_pts, deadline=deadline - 0.5)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
            
    # Final validity check
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)
        
    return best_pts
