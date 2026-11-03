import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, distance
import sys

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    for attempt in range(12):
        # Vary spacing slightly to explore different lattice densities
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * (0.92 + 0.03 * attempt)
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
            # Add small noise to break symmetry and escape local minima
            noise_scale = spacing * (0.02 + 0.01 * attempt)
            selected = selected + rng.normal(0, noise_scale, selected.shape)
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
        r_est = 1.0 / np.sqrt(n_points) * (0.95 + 0.02 * attempt)
        
        points = []
        # Center point
        points.append([0.0, 0.0])
        
        ring_idx = 1
        while len(points) < n_points:
            # Ring radius
            ring_r = ring_idx * r_est * 2.0 * 0.577
            if ring_r >= 1.0 - r_est:
                break
            # Number of points on this ring
            n_ring = max(6, int(2.0 * np.pi * ring_r / (2.0 * r_est)))
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

def _delaunay_refine(pts, n_iter=5, T=0.0001, time_budget=10.0):
    """
    Refine the packing by identifying critical edges via Delaunay triangulation
    and optimizing only the points involved in those edges.
    """
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    
    for _ in range(n_iter):
        if time.time() - start_time > time_budget:
            break
            
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(pts)
        except Exception:
            break
            
        # Get edges from triangulation
        # tri.simplices has shape (n_tri, 3)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    u, v = simplex[i], simplex[j]
                    if u > v:
                        u, v = v, u
                    edges.add((u, v))
        
        # Also include boundary constraints? 
        # For now, focus on pairwise distances.
        
        # Calculate distances for all edges
        edge_list = list(edges)
        if not edge_list:
            break
            
        # Find the shortest edge
        min_dist = np.inf
        min_edge_idx = -1
        for idx, (u, v) in enumerate(edge_list):
            d = np.linalg.norm(pts[u] - pts[v])
            if d < min_dist:
                min_dist = d
                min_edge_idx = idx
                
        if min_edge_idx == -1:
            break
            
        # Identify points involved in edges shorter than min_dist * 1.5 (active set)
        threshold = min_dist * 1.5
        active_points = set()
        for u, v in edge_list:
            d = np.linalg.norm(pts[u] - pts[v])
            if d < threshold:
                active_points.add(u)
                active_points.add(v)
                
        # Also add points close to boundary
        norms = np.linalg.norm(pts, axis=1)
        boundary_dist = 1.0 - norms
        min_boundary = boundary_dist.min()
        for i in range(n):
            if boundary_dist[i] < min_boundary + 0.5 * min_dist:
                active_points.add(i)
                
        if len(active_points) < 2:
            break
            
        active_idx = np.array(list(active_points))
        n_active = len(active_idx)
        
        # Define a local objective that only moves active points
        # We keep fixed points fixed.
        # This is a bit tricky with standard minimize.
        # Instead, we can just run a global minimize but with a very low T 
        # and hope the gradient focuses on the active points.
        # However, a more direct approach is to use SLSQP on the active points 
        # with constraints that they don't overlap with fixed points or each other.
        # Given complexity, let's stick to a "nudged" local minimize on the full set 
        # but with a very aggressive focus on the minimum constraint.
        
        # Alternative: Perturb only active points slightly and re-optimize
        # This is safer and simpler.
        rng = np.random.default_rng(len(active_idx))
        perturbation = rng.normal(0, min_dist * 0.05, (n_active, 2))
        
        new_pts = pts.copy()
        new_pts[active_idx] += perturbation
        
        # Ensure validity
        norms = np.linalg.norm(new_pts, axis=1)
        mask = norms < 1.0
        if not np.all(mask):
            # Scale back if out of bounds
            new_pts[~mask] *= 0.999 / norms[~mask, np.newaxis]
            
        # Optimize this perturbed state
        try:
            result = minimize(
                lambda x: _objective_and_grad(x, T)[0],
                new_pts.ravel(),
                jac=lambda x: _objective_and_grad(x, T)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            opt_pts = result.x.reshape(-1, 2)
            if np.all(np.isfinite(opt_pts)) and np.all(np.linalg.norm(opt_pts, axis=1) < 1.0):
                new_r = _compute_radius(opt_pts)
                old_r = _compute_radius(pts)
                if new_r > old_r:
                    pts = opt_pts
                else:
                    # If it didn't improve, keep original pts for next iteration
                    pass
        except Exception:
            pass
            
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pts = None
    best_r = -1.0
    rng = np.random.default_rng(12345)
    
    # Phase 1: Diverse initialization
    init_candidates = []
    
    for seed in [42, 0, 1, 2, 3]:
        if time.time() > deadline - 90.0:
            break
        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)
        init_candidates.append((r, pts))
    
    if time.time() < deadline - 90.0:
        for seed in [10, 20]:
            pts = _ring_init(N, seed=seed)
            r = _compute_radius(pts)
            init_candidates.append((r, pts))
    
    init_candidates.sort(key=lambda x: -x[0])
    best_r = init_candidates[0][0]
    best_pts = init_candidates[0][1].copy()
    
    x0 = best_pts.ravel()
    
    # Phase 2: Temperature annealing optimization
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
    
    # Phase 3: Anneal restarts
    if time.time() < deadline - 40.0:
        for restart in range(3):
            if time.time() > deadline - 35.0:
                break
            perturbed = best_pts + rng.normal(0, 0.005 * (restart + 1), best_pts.shape)
            norms = np.linalg.norm(perturbed, axis=1)
            mask = norms < 1.0
            perturbed = perturbed[mask]
            if len(perturbed) < N:
                continue
            perturbed = perturbed[:N]
            
            T_restart = 0.003
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, T_restart)[0],
                    perturbed.ravel(),
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
    
    # Phase 4: Delaunay-based refinement (NEW)
    if time.time() < deadline - 20.0:
        try:
            refined_pts = _delaunay_refine(best_pts, n_iter=5, T=0.0001, time_budget=15.0)
            norms = np.linalg.norm(refined_pts, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(refined_pts)):
                refined_r = _compute_radius(refined_pts)
                if refined_r > best_r:
                    best_r = refined_r
                    best_pts = refined_pts.copy()
                    x0 = refined_pts.ravel()
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
    
    # Phase 6: Final tighten
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
    
    # Ensure validity
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask):
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)
    
    return best_pts
