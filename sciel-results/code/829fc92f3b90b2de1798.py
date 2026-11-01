"""Improved program: place 14 points in 3D to minimize (max/min pairwise distance)^2."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 14
D = 3
DEADLINE = 145.0  # safety margin before 150s kill

def _pairwise_dists_sq(pts):
    """Return all pairwise squared distances (upper triangle), shape (n_pairs,)."""
    n = len(pts)
    idx_i, idx_j = np.triu_indices(n, k=1)
    diffs = pts[idx_i] - pts[idx_j]
    return np.sum(diffs * diffs, axis=1)

def _score(pts):
    """Compute (max_dist / min_dist)^2."""
    d2 = _pairwise_dists_sq(pts)
    dmin = np.min(d2)
    dmax = np.max(d2)
    if dmin < 1e-30:
        return 1e12
    return (dmax / dmin) ** 2

def _objective(pts_flat):
    """Objective for optimizer: minimize (max/min)^2 with penalty for near-coincident points."""
    pts = pts_flat.reshape(-1, D)
    d2 = _pairwise_dists_sq(pts)
    dmin = np.min(d2)
    dmax = np.max(d2)
    if dmin < 1e-20:
        return 1e12
    ratio_sq = (dmax / dmin) ** 2
    # Penalty for any pair too close (encourages distinctness)
    penalty = 0.0
    for v in d2:
        if v < 1e-6:
            penalty += (1e-6 - v) ** 2 * 1e6
    return ratio_sq + penalty

def _tetrahedron_init():
    """Create a regular tetrahedron with edge length 1, plus midpoints of edges."""
    # Regular tetrahedron vertices (edge length sqrt(2) for now, will scale)
    # Standard regular tetrahedron centered at origin with circumradius 1:
    v0 = np.array([1, 1, 1], dtype=float)
    v1 = np.array([1, -1, -1], dtype=float)
    v2 = np.array([-1, 1, -1], dtype=float)
    v3 = np.array([-1, -1, 1], dtype=float)
    # Scale so edge length is 1: distance between v0,v1 = sqrt((0)^2+(2)^2+(2)^2)=sqrt(8)=2*sqrt(2)
    # So divide by 2*sqrt(2)
    scale = 1.0 / (2.0 * np.sqrt(2.0))
    verts = np.array([v0, v1, v2, v3]) * scale
    # Edge midpoints (6 edges)
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    mids = np.array([(verts[i] + verts[j]) / 2.0 for i, j in edges])
    # We have 4 + 6 = 10 points. Need 14. Add 4 more points.
    # Add points at centroid of faces? Or just use the tetrahedron vertices scaled differently?
    # Let's add 4 more points: perhaps at the centers of the 4 faces.
    faces = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
    face_centers = np.array([(verts[i]+verts[j]+verts[k])/3.0 for i,j,k in faces])
    # Now we have 4 + 6 + 4 = 14 points
    pts = np.vstack([verts, mids, face_centers])
    return pts

def _random_init():
    """Random initialization in unit cube."""
    return np.random.rand(N, D)

def _optimize_from_start(start_pts, max_iter=5000, tol=1e-10):
    """Run L-BFGS-B from a starting point."""
    x0 = start_pts.flatten()
    result = minimize(_objective, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
    return result.x.reshape(-1, D)

def _optimize_from_gram(start_pts, max_iter=3000, tol=1e-10):
    """
    Optimize using Gram matrix parameterization.
    G = P^T P must be PSD. We optimize over upper triangle of G, then take Cholesky.
    """
    n = N
    # Number of unique elements in symmetric matrix
    idx = np.triu_indices(n)
    m = len(idx[0])
    
    def _gram_objective(g_flat):
        g = np.zeros((n, n))
        g[idx] = g_flat
        g = g + g.T - np.diag(np.diag(g))  # make symmetric
        # Ensure PSD by adding small identity if needed
        eigvals = np.linalg.eigvalsh(g)
        if np.min(eigvals) < -1e-8:
            return 1e10
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(g + 1e-12 * np.eye(n))
        except np.linalg.LinAlgError:
            return 1e10
        pts = L.T  # P such that P^T P = G
        return _objective(pts.flatten())
    
    g0 = start_pts.T @ start_pts
    g0_flat = g0[idx].copy()
    
    result = minimize(_gram_objective, g0_flat, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
    if result.success or _score(start_pts) > 1e10:
        g = np.zeros((n, n))
        g[idx] = result.x
        g = g + g.T - np.diag(np.diag(g))
        try:
            L = np.linalg.cholesky(g + 1e-12 * np.eye(n))
            return L.T
        except np.linalg.LinAlgError:
            pass
    return start_pts

def construct():
    t_start = time.time()
    best_pts = None
    best_score = 1e12
    
    # Multiple starting points
    starts = []
    
    # 1. Tetrahedron-based initialization
    starts.append(_tetrahedron_init())
    
    # 2. Random initializations
    for _ in range(5):
        starts.append(_random_init())
    
    # 3. Sphere-based: points on a sphere with some perturbation
    from scipy.spatial import ConvexHull
    # Use Fibonacci sphere for 14 points
    n_sphere = 14
    golden_angle = np.pi * (3 - np.sqrt(5))
    sphere_pts = np.zeros((n_sphere, 3))
    for i in range(n_sphere):
        z = 1 - 2 * i / (n_sphere - 1)
        r = np.sqrt(1 - z*z)
        theta = golden_angle * i
        sphere_pts[i] = [r * np.cos(theta), r * np.sin(theta), z]
    starts.append(sphere_pts)
    
    # Try each starting point with coordinate optimization
    for i, start in enumerate(starts):
        if time.time() - t_start > DEADLINE - 30:
            break
        try:
            opt_pts = _optimize_from_start(start, max_iter=2000, tol=1e-9)
            sc = _score(opt_pts)
            if sc < best_score:
                best_score = sc
                best_pts = opt_pts
        except Exception:
            pass
    
    # Try Gram matrix optimization from the best coordinate result
    if best_pts is not None and time.time() - t_start < DEADLINE - 20:
        try:
            gram_pts = _optimize_from_gram(best_pts, max_iter=2000, tol=1e-10)
            sc = _score(gram_pts)
            if sc < best_score:
                best_score = sc
                best_pts = gram_pts
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if best_pts is not None and time.time() - t_start < DEADLINE - 10:
        try:
            polished = _optimize_from_start(best_pts, max_iter=5000, tol=1e-12)
            sc = _score(polished)
            if sc < best_score:
                best_score = sc
                best_pts = polished
        except Exception:
            pass
    
    # Ensure distinctness: if any two points are too close, perturb
    if best_pts is None:
        best_pts = _tetrahedron_init()
    
    # Final check: ensure all points are distinct
    d2 = _pairwise_dists_sq(best_pts)
    if np.min(d2) < 1e-10:
        # Perturb coincident points
        for i in range(N):
            for j in range(i+1, N):
                if np.sum((best_pts[i] - best_pts[j])**2) < 1e-10:
                    best_pts[j] += np.random.randn(D) * 1e-5
    
    return best_pts
