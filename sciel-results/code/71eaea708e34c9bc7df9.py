"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import math


def _compute_facet_data(pts):
    """
    Compute facet normals (circumcenters) and angular distances to nearest site.
    Returns:
        normals: (n_facets, 3) outward unit normals
        radii: (n_facets,) angular distance from each circumcenter to nearest site
        simplices: (n_facets, 3) vertex indices for each facet
        valid_mask: (n_facets,) boolean mask of valid facets
    """
    try:
        hull = ConvexHull(pts)
        tris = hull.simplices
        a = pts[tris[:, 0]]
        b = pts[tris[:, 1]]
        c = pts[tris[:, 2]]
        
        normals = np.cross(b - a, c - a)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            return None, None, None, None
        
        normals = normals[valid]
        norms = norms[valid]
        normals = normals / norms[:, np.newaxis]
        
        a_valid = a[valid]
        dot_products = np.sum(normals * a_valid, axis=1)
        flip = dot_products < 0
        normals[flip] = -normals[flip]
        
        # Compute angular distance from each circumcenter to nearest site
        dots = np.clip(pts @ normals.T, -1.0, 1.0)
        dists = np.arccos(dots)  # shape: (n_sites, n_facets)
        radii = np.min(dists, axis=0)  # shape: (n_facets,)
        
        simplices_valid = tris[valid]
        
        return normals, radii, simplices_valid, valid
    except Exception:
        return None, None, None, None


def _covering_radius(pts):
    """Compute covering radius in radians. Returns (radius, worst_facet_index, worst_point)."""
    normals, radii, simplices, valid = _compute_facet_data(pts)
    if normals is None:
        return np.pi, -1, None
    
    max_idx = np.argmax(radii)
    max_ang = radii[max_idx]
    worst_facet = np.where(valid)[0][max_idx]
    worst_pt = normals[max_idx].copy()
    
    return max_ang, worst_facet, worst_pt


def _soft_covering_radius(pts, temperature=0.05):
    """
    Compute a smooth soft-max approximation of the covering radius.
    Returns (soft_radius, gradient_wrt_pts) where gradient has shape (N, 3).
    
    The soft-max is: T * log(sum(exp(r_i / T)))
    where r_i are the per-facet radii.
    """
    normals, radii, simplices, valid = _compute_facet_data(pts)
    if normals is None:
        return np.pi, np.zeros((N, 3))
    
    # Soft-max for smooth maximum
    shifted = radii - np.max(radii)
    weights = np.exp(shifted / temperature)
    weights /= np.sum(weights)
    soft_radius = np.max(radii) + temperature * np.log(np.sum(weights))
    
    # Compute gradient: d(soft_max)/d(pts)
    # soft_max = T * log(sum(exp(r_i/T)))
    # d(soft_max)/d(x) = sum_i (w_i * dr_i/dx)
    # where w_i = exp(r_i/T) / sum(exp(r_j/T))
    
    # For each facet i, r_i = arccos(min_j <p_j, n_i>)
    # dr_i/d(p_k) = -n_i / sqrt(1 - <p_k, n_i>^2) for the argmin k
    # But we need the gradient through the min over j, which is only non-zero for argmin
    
    # Simplified: for each facet, find the nearest point and compute gradient contribution
    dots = np.clip(pts @ normals.T, -1.0, 1.0)  # (N, n_facets)
    dists = np.arccos(dots)  # (N, n_facets)
    
    # For each facet, find the nearest point index
    nearest_idx = np.argmin(dists, axis=0)  # (n_facets,)
    
    grad = np.zeros((N, 3))
    for i in range(len(radii)):
        if weights[i] < 1e-10:
            continue
        k = nearest_idx[i]
        # dr_i/d(p_k) = -n_i / sqrt(1 - dots[k,i]^2)
        dot_val = dots[k, i]
        denom = np.sqrt(max(1e-12, 1.0 - dot_val * dot_val))
        grad[k] -= (weights[i] / denom) * normals[i]
    
    # Scale by 1/temperature from the soft-max derivative
    grad /= temperature
    
    return soft_radius, grad


def _fibonacci_start():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(rng):
    """Generate random initial points on the sphere."""
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _lloyd_step(pts, n_iter=30):
    """Perform Lloyd's algorithm iterations to minimize covering radius."""
    pts = pts.copy()
    for _ in range(n_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Vectorized circumcenter computation
        tris = hull.simplices
        a = pts[tris[:, 0]]
        b = pts[tris[:, 1]]
        c = pts[tris[:, 2]]
        
        normals = np.cross(b - a, c - a)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            break
        
        normals = normals[valid]
        norms = norms[valid]
        normals = normals / norms[:, np.newaxis]
        
        a_valid = a[valid]
        dot_products = np.sum(normals * a_valid, axis=1)
        flip = dot_products < 0
        normals[flip] = -normals[flip]
        
        circumcenters = normals
        
        new_pts = pts.copy()
        for i in range(len(pts)):
            dists = np.linalg.norm(circumcenters - pts[i], axis=1)
            k = min(5, len(circumcenters))
            idx = np.argsort(dists)[:k]
            centroid = circumcenters[idx].mean(axis=0)
            nc = np.linalg.norm(centroid)
            if nc > 1e-10:
                centroid = centroid / nc
                new_pts[i] = 0.5 * pts[i] + 0.5 * centroid
                new_pts[i] /= np.linalg.norm(new_pts[i])
        
        pts = new_pts
    
    return pts


def _local_optimize(pts, n_iter=100, base_alpha=0.05):
    """Local optimization targeting the worst facet with adaptive step size."""
    pts = pts.copy()
    for it in range(n_iter):
        radius, wf, wpt = _covering_radius(pts)
        if wf < 0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        tri = hull.simplices[wf]
        a_idx, b_idx, c_idx = tri[0], tri[1], tri[2]
        
        # Adaptive step size: reduce if improvement is small
        alpha = base_alpha * (0.5 ** (it // 20))
        
        # Try moving each vertex toward the circumcenter
        best_trial = None
        best_r = radius
        for idx in [a_idx, b_idx, c_idx]:
            for alpha_mult in [1.0, 0.5, 0.25, 0.1]:
                trial = pts.copy()
                new_pt = trial[idx] + (alpha * alpha_mult) * (wpt - trial[idx])
                new_pt /= np.linalg.norm(new_pt)
                trial[idx] = new_pt
                r_trial, _, _ = _covering_radius(trial)
                if r_trial < best_r:
                    best_r = r_trial
                    best_trial = trial.copy()
        
        if best_trial is not None:
            pts = best_trial
        else:
            # No improvement found with current alpha, try smaller steps
            improved = False
            for idx in [a_idx, b_idx, c_idx]:
                for alpha_mult in [0.05, 0.02, 0.01]:
                    trial = pts.copy()
                    new_pt = trial[idx] + (alpha * alpha_mult) * (wpt - trial[idx])
                    new_pt /= np.linalg.norm(new_pt)
                    trial[idx] = new_pt
                    r_trial, _, _ = _covering_radius(trial)
                    if r_trial < best_r:
                        best_r = r_trial
                        best_trial = trial.copy()
                        improved = True
                        break
                if improved:
                    break
            if best_trial is not None:
                pts = best_trial
            else:
                break
    
    return pts


def _polish(pts, n_iter=50):
    """High-precision polish phase with adaptive step-size reduction."""
    pts = pts.copy()
    for it in range(n_iter):
        radius, wf, wpt = _covering_radius(pts)
        if wf < 0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        tri = hull.simplices[wf]
        
        # Try multiple step sizes, moving all three vertices
        best_trial = None
        best_r = radius
        for alpha_mult in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
            alpha = 0.01 * alpha_mult
            trial = pts.copy()
            for idx in tri:
                new_pt = trial[idx] + alpha * (wpt - trial[idx])
                new_pt /= np.linalg.norm(new_pt)
                trial[idx] = new_pt
            
            r_trial, _, _ = _covering_radius(trial)
            if r_trial < best_r:
                best_r = r_trial
                best_trial = trial.copy()
        
        # Also try moving individual vertices
        for idx in tri:
            for alpha_mult in [1.0, 0.5, 0.25, 0.1, 0.05]:
                alpha = 0.01 * alpha_mult
                trial = pts.copy()
                new_pt = trial[idx] + alpha * (wpt - trial[idx])
                new_pt /= np.linalg.norm(new_pt)
                trial[idx] = new_pt
                r_trial, _, _ = _covering_radius(trial)
                if r_trial < best_r:
                    best_r = r_trial
                    best_trial = trial.copy()
        
        if best_trial is not None:
            pts = best_trial
        else:
            break
    
    return pts


def _gradient_descent_step(pts, n_iter=50, lr=0.005, temperature=0.05):
    """
    Use gradient descent on the soft-max covering radius to refine the configuration.
    """
    pts = pts.copy()
    for it in range(n_iter):
        soft_r, grad = _soft_covering_radius(pts, temperature=temperature)
        
        # Apply gradient step
        new_pts = pts - lr * grad
        
        # Project back to unit sphere
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / norms
        
        # Check if this improved the actual (non-soft) covering radius
        r_trial, _, _ = _covering_radius(new_pts)
        r_current, _, _ = _covering_radius(pts)
        
        if r_trial < r_current:
            pts = new_pts
            # Decay learning rate
            lr *= 0.95
        else:
            # Try smaller step
            for scale in [0.5, 0.25, 0.1]:
                new_pts2 = pts - lr * scale * grad
                norms2 = np.linalg.norm(new_pts2, axis=1, keepdims=True)
                new_pts2 = new_pts2 / norms2
                r_trial2, _, _ = _covering_radius(new_pts2)
                if r_trial2 < r_current:
                    pts = new_pts2
                    lr *= 0.95
                    break
            else:
                break
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate multiple initial configurations
    best_pts = None
    best_radius = np.pi
    
    # Fibonacci start
    pts_fib = _fibonacci_start()
    r_fib, _, _ = _covering_radius(pts_fib)
    if r_fib < best_radius:
        best_radius = r_fib
        best_pts = pts_fib.copy()
    
    # Several random starts
    for _ in range(6):
        if time.time() > deadline - 60:
            break
        pts_rand = _random_start(rng)
        r_rand, _, _ = _covering_radius(pts_rand)
        if r_rand < best_radius:
            best_radius = r_rand
            best_pts = pts_rand.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_start()
        best_radius, _, _ = _covering_radius(best_pts)
    
    pts = best_pts.copy()
    
    # Phase 2: Lloyd's algorithm for global improvement
    while time.time() < deadline - 50:
        pts_lloyd = _lloyd_step(pts, n_iter=15)
        r_lloyd, _, _ = _covering_radius(pts_lloyd)
        if r_lloyd < best_radius:
            best_radius = r_lloyd
            best_pts = pts_lloyd.copy()
            pts = pts_lloyd.copy()
        else:
            pts_lloyd = _lloyd_step(pts, n_iter=5)
            r_lloyd, _, _ = _covering_radius(pts_lloyd)
            if r_lloyd < best_radius:
                best_radius = r_lloyd
                best_pts = pts_lloyd.copy()
                pts = pts_lloyd.copy()
            else:
                break
    
    # Phase 3: Gradient descent on soft-max surrogate
    while time.time() < deadline - 30:
        pts_grad = _gradient_descent_step(pts, n_iter=30, lr=0.005, temperature=0.05)
        r_grad, _, _ = _covering_radius(pts_grad)
        if r_grad < best_radius:
            best_radius = r_grad
            best_pts = pts_grad.copy()
            pts = pts_grad.copy()
        else:
            # Try with different temperature
            pts_grad = _gradient_descent_step(pts, n_iter=20, lr=0.003, temperature=0.02)
            r_grad, _, _ = _covering_radius(pts_grad)
            if r_grad < best_radius:
                best_radius = r_grad
                best_pts = pts_grad.copy()
                pts = pts_grad.copy()
            else:
                break
    
    # Phase 4: Local optimization targeting worst facets
    while time.time() < deadline - 15:
        pts_local = _local_optimize(pts, n_iter=50)
        r_local, _, _ = _covering_radius(pts_local)
        if r_local < best_radius:
            best_radius = r_local
            best_pts = pts_local.copy()
            pts = pts_local.copy()
        else:
            pts_local = _local_optimize(pts, n_iter=20)
            r_local, _, _ = _covering_radius(pts_local)
            if r_local < best_radius:
                best_radius = r_local
                best_pts = pts_local.copy()
                pts = pts_local.copy()
            else:
                break
    
    # Phase 5: High-precision polish
    while time.time() < deadline - 5:
        pts_polished = _polish(pts, n_iter=40)
        r_polished, _, _ = _covering_radius(pts_polished)
        if r_polished < best_radius:
            best_radius = r_polished
            best_pts = pts_polished.copy()
            pts = pts_polished.copy()
        else:
            break
    
    # Ensure output is valid
    pts_out = best_pts.copy()
    norms = np.linalg.norm(pts_out, axis=1)
    pts_out = pts_out / norms[:, np.newaxis]
    
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(pts_out[i] - pts_out[j]) < 1e-8:
                pts_out[j] += rng.standard_normal(3) * 0.01
                pts_out[j] /= np.linalg.norm(pts_out[j])
    
    return pts_out
# EVOLVE-BLOCK-END
