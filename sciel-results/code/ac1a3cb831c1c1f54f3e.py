N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.optimize import minimize
import time


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-14] = 1.0
    return pts / norms


def _facet_circumradii(pts):
    """
    Compute circumradii (in radians) for all triangular facets of the convex hull.
    Returns an array of circumradii.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.array([np.pi / 2])
    
    simplices = hull.simplices
    if len(simplices) == 0:
        return np.array([np.pi / 2])
    
    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    mask = norms > 1e-12
    norms[mask] = 1.0 / norms[mask]
    norms[~mask] = 0.0
    normals = normals * norms
    
    dots_p1 = np.sum(normals * p1, axis=1)
    normals[dots_p1 < 0] *= -1
    
    dots1 = np.sum(normals * p1, axis=1)
    dots2 = np.sum(normals * p2, axis=1)
    dots3 = np.sum(normals * p3, axis=1)
    dots1 = np.clip(dots1, -1.0, 1.0)
    dots2 = np.clip(dots2, -1.0, 1.0)
    dots3 = np.clip(dots3, -1.0, 1.0)
    
    angles1 = np.arccos(dots1)
    angles2 = np.arccos(dots2)
    angles3 = np.arccos(dots3)
    
    cr = np.max(np.stack([angles1, angles2, angles3], axis=1), axis=1)
    
    valid = mask.flatten()
    cr = cr[valid]
    
    if len(cr) == 0:
        return np.array([np.pi / 2])
    
    return cr


def _covering_radius(pts):
    """Exact covering radius in radians using ConvexHull."""
    cr = _facet_circumradii(pts)
    return np.max(cr) if len(cr) > 0 else np.pi / 2


def _fast_circumradii(pts, k=3):
    """
    Fast approximate circumradii using k-nearest neighbors.
    For each point, compute the circumradius of the triangle formed by
    the point and its (k-1) nearest neighbors. This avoids ConvexHull.
    """
    n = len(pts)
    if n < 3:
        return np.array([np.pi / 2])
    
    # Compute pairwise angular distances
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # For each point, find k nearest neighbors (excluding self)
    # Set diagonal to inf to exclude self
    np.fill_diagonal(dists, np.inf)
    
    # Get indices of k nearest neighbors for each point
    nn_idx = np.argsort(dists, axis=1)[:, :k]
    
    circumradii = np.zeros(n)
    for i in range(n):
        if nn_idx[i, 0] >= n:
            circumradii[i] = np.pi / 2
            continue
        # Use the point and its two nearest neighbors
        a = pts[i]
        b = pts[nn_idx[i, 0]]
        c = pts[nn_idx[i, 1]]
        
        # Compute circumradius of spherical triangle
        v1 = b - a
        v2 = c - a
        nrm = np.cross(v1, v2)
        nn = np.linalg.norm(nrm)
        if nn < 1e-12:
            circumradii[i] = np.pi / 2
            continue
        nrm = nrm / nn
        if np.dot(nrm, a) < 0:
            nrm = -nrm
        
        d1 = np.clip(np.dot(nrm, a), -1.0, 1.0)
        d2 = np.clip(np.dot(nrm, b), -1.0, 1.0)
        d3 = np.clip(np.dot(nrm, c), -1.0, 1.0)
        
        circumradii[i] = np.max([np.arccos(d1), np.arccos(d2), np.arccos(d3)])
    
    return circumradii


def _smooth_max_circumradii_fast(pts, beta=50.0):
    """
    Smooth approximation of max squared circumradius using fast neighbor-based method.
    """
    cr = _fast_circumradii(pts, k=3)
    if len(cr) == 0:
        return np.pi / 2
    
    cr_sq = cr ** 2
    max_cr_sq = np.max(cr_sq)
    if max_cr_sq < 1e-10:
        return 0.0
    
    shifted = cr_sq - max_cr_sq
    log_sum = max_cr_sq + np.log(np.sum(np.exp(beta * shifted))) / beta
    return log_sum


def _smooth_max_circumradii_exact(pts, beta=50.0):
    """
    Smooth approximation of max squared circumradius using exact ConvexHull method.
    """
    cr = _facet_circumradii(pts)
    if len(cr) == 0:
        return np.pi / 2
    
    cr_sq = cr ** 2
    max_cr_sq = np.max(cr_sq)
    if max_cr_sq < 1e-10:
        return 0.0
    
    shifted = cr_sq - max_cr_sq
    log_sum = max_cr_sq + np.log(np.sum(np.exp(beta * shifted))) / beta
    return log_sum


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _lbfgs_optimize(pts, deadline, max_iter=300, beta=50.0, use_fast=True):
    """
    Optimize using L-BFGS. Uses fast surrogate by default, exact for final polish.
    """
    if use_fast:
        def objective(x):
            p = _normalize(x.reshape(N, 3))
            return _smooth_max_circumradii_fast(p, beta)
    else:
        def objective(x):
            p = _normalize(x.reshape(N, 3))
            return _smooth_max_circumradii_exact(p, beta)
    
    x0 = pts.flatten()
    
    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-7, 'maxls': 40, 'ftol': 1e-9}
        )
        return _normalize(res.x.reshape(N, 3))
    except Exception:
        return pts


def _local_facet_search(pts, deadline, max_iter=300):
    """
    Local search: find worst facet, move its vertices toward the facet center.
    Uses exact ConvexHull for accuracy.
    """
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    step_size = 0.03
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        try:
            hull = ConvexHull(pts)
            simplices = hull.simplices
            if len(simplices) == 0:
                break
        except Exception:
            break
        
        cr = _facet_circumradii(pts)
        if len(cr) == 0:
            break
        
        worst_idx = np.argmax(cr)
        
        if worst_idx >= len(simplices):
            break
        
        facet = simplices[worst_idx]
        p1, p2, p3 = pts[facet[0]], pts[facet[1]], pts[facet[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            break
        n = n / nn
        if np.dot(n, p1) < 0:
            n = -n
        
        new_pts = pts.copy()
        for idx in facet:
            move = (n - new_pts[idx]) * step_size
            new_pts[idx] = _normalize(new_pts[idx] + move)
        
        new_cr = _covering_radius(new_pts)
        if new_cr < best_cr - 1e-13:
            best_cr = new_cr
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            step_size *= 0.97
        else:
            step_size *= 0.5
            if step_size < 1e-7:
                break
    
    return best_pts


def _electrostatic_repulsion(pts, k=8, strength=1.0):
    """
    Apply electrostatic repulsion from k nearest neighbors to spread points.
    """
    n = len(pts)
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    dists = np.arccos(dots)
    np.fill_diagonal(dists, np.inf)
    
    nn_idx = np.argsort(dists, axis=1)[:, :k]
    
    new_pts = pts.copy().astype(float)
    for i in range(n):
        force = np.zeros(3)
        for j in nn_idx[i]:
            if j >= n:
                continue
            # Repulsive force along the direction away from neighbor
            d = pts[i] - pts[j]
            dn = np.linalg.norm(d)
            if dn > 1e-10:
                force += d / dn * strength
        new_pts[i] = _normalize(pts[i] + force * 0.01)
    
    return new_pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Phase 1: Generate diverse starting configurations
    starts = [
        _fibonacci_sphere(N),
        _random_sphere(N, 1),
        _random_sphere(N, 2),
        _random_sphere(N, 3),
        _random_sphere(N, 4),
        _random_sphere(N, 5),
    ]
    
    best_pts = starts[0].copy()
    best_cr = _covering_radius(best_pts)
    
    # Phase 2: L-BFGS with fast surrogate from each start
    for i, s in enumerate(starts):
        if time.time() > deadline - 30:
            break
        try:
            optimized = _lbfgs_optimize(s, deadline - 25, max_iter=250, beta=50.0, use_fast=True)
            cr = _covering_radius(optimized)
            if cr < best_cr:
                best_cr = cr
                best_pts = optimized.copy()
        except Exception:
            pass
    
    # Phase 3: Local facet search from best
    if time.time() < deadline - 25:
        try:
            local_best = _local_facet_search(best_pts, deadline - 20, max_iter=300)
            cr = _covering_radius(local_best)
            if cr < best_cr:
                best_cr = cr
                best_pts = local_best.copy()
        except Exception:
            pass
    
    # Phase 4: L-BFGS polish with exact surrogate
    if time.time() < deadline - 15:
        try:
            polished = _lbfgs_optimize(best_pts, deadline - 10, max_iter=150, beta=80.0, use_fast=False)
            cr = _covering_radius(polished)
            if cr < best_cr:
                best_cr = cr
                best_pts = polished.copy()
        except Exception:
            pass
    
    # Phase 5: Another local facet search pass
    if time.time() < deadline - 10:
        try:
            local_best2 = _local_facet_search(best_pts, deadline - 5, max_iter=200)
            cr = _covering_radius(local_best2)
            if cr < best_cr:
                best_cr = cr
                best_pts = local_best2.copy()
        except Exception:
            pass
    
    # Phase 6: Final L-BFGS with very tight tolerance
    if time.time() < deadline - 5:
        try:
            final_polish = _lbfgs_optimize(best_pts, deadline - 2, max_iter=100, beta=120.0, use_fast=False)
            cr = _covering_radius(final_polish)
            if cr < best_cr:
                best_cr = cr
                best_pts = final_polish.copy()
        except Exception:
            pass
    
    # Final validation: ensure no coincident points
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            min_dist = min(min_dist, d)
    if min_dist < 1e-6:
        rng = np.random.default_rng(42)
        best_pts = _normalize(best_pts + rng.randn(N, 3) * 1e-4)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
