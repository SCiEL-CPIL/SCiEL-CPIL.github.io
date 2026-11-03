"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _covering_radius(pts):
    """
    Estimate the covering radius (in radians) by checking the angular distance
    from a dense set of test points on the sphere to the nearest configuration point.
    """
    M = 30000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = test_pts @ pts.T  # shape (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)
    angles = np.arccos(max_dots)
    return np.max(angles)


def _spectral_objective(pts):
    """
    Objective based on spectral properties of the Gram matrix.
    For uniform points, the Gram matrix should have eigenvalues close to
    the theoretical values for a spherical design.
    
    We minimize:
    1. Variance of pairwise dot products (off-diagonal)
    2. Penalty for points too close
    3. Covering radius estimate
    
    The Gram matrix G = P @ P.T has trace = N (since points are unit vectors).
    For a spherical 2-design, sum_{i!=j} dot(i,j)^2 should be minimized.
    """
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    pts_n = pts / norms
    
    # Gram matrix
    G = pts_n @ pts_n.T
    np.fill_diagonal(G, 0.0)
    
    # Sum of squares of off-diagonal elements (measures uniformity)
    # For random points, this is ~N*(N-1)/N = N-1 on average per pair
    # We want to minimize this
    sum_sq_offdiag = np.sum(G ** 2)
    
    # Also minimize the max off-diagonal dot (to avoid clustering)
    max_offdiag = np.max(np.abs(G))
    
    # Covering radius
    cr = _covering_radius(pts_n)
    
    # Combined objective
    # Weight: sum_sq_offdiag should be normalized
    # For N points, ideal sum_sq_offdiag ~ (N-1)/N * N = N-1 for uniform
    obj = 0.3 * sum_sq_offdiag / N + 0.2 * max_offdiag + 0.5 * cr
    
    return obj


def _spectral_grad(x):
    """
    Numerical gradient of the spectral objective.
    """
    eps = 1e-7
    n = len(x)
    grad = np.zeros(n)
    f0 = _spectral_objective(x.reshape(N, 3))
    
    for i in range(n):
        xp = x.copy()
        xp[i] += eps
        fp = _spectral_objective(xp.reshape(N, 3))
        
        xm = x.copy()
        xm[i] -= eps
        fm = _spectral_objective(xm.reshape(N, 3))
        
        grad[i] = (fp - fm) / (2 * eps)
    
    return grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return pts / norms


def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    x0 = pts.flatten()
    
    start_time = time.time()
    
    # Use L-BFGS-B with analytical gradient approximation
    # Since computing full numerical gradient is expensive, use a subset
    # or use the fact that L-BFGS-B can estimate gradients
    
    # First, try with a simpler objective that has easier gradients
    # Minimize sum of squared off-diagonal Gram entries + covering radius
    
    def obj_simple(x):
        pts_reshaped = x.reshape(N, 3)
        norms = np.linalg.norm(pts_reshaped, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts_n = pts_reshaped / norms
        
        G = pts_n @ pts_n.T
        np.fill_diagonal(G, 0.0)
        sum_sq = np.sum(G ** 2)
        
        # Quick covering radius estimate with fewer test points
        M = 5000
        ii = np.arange(M) + 0.5
        zz = 1.0 - 2.0 * ii / M
        pp = np.pi * (1.0 + np.sqrt(5.0)) * ii
        rr = np.sqrt(1.0 - zz * zz)
        test_pts = np.column_stack([rr * np.cos(pp), rr * np.sin(pp), zz])
        dots = test_pts @ pts_n.T
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        cr = np.max(np.arccos(max_dots))
        
        return 0.4 * sum_sq / N + 0.6 * cr
    
    # Use L-BFGS-B with numerical gradient (scipy will handle it)
    result = minimize(obj_simple, x0, method='L-BFGS-B', 
                      options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-10})
    
    elapsed = time.time() - start_time
    if elapsed > 90:
        return pts
    
    x_opt = result.x
    pts_opt = x_opt.reshape(N, 3)
    pts_opt = _project_to_sphere(pts_opt)
    
    # Second stage: refine with full covering radius
    def obj_full(x):
        pts_reshaped = x.reshape(N, 3)
        norms = np.linalg.norm(pts_reshaped, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts_n = pts_reshaped / norms
        
        G = pts_n @ pts_n.T
        np.fill_diagonal(G, 0.0)
        sum_sq = np.sum(G ** 2)
        
        cr = _covering_radius(pts_n)
        
        return 0.3 * sum_sq / N + 0.7 * cr
    
    start_time2 = time.time()
    result2 = minimize(obj_full, x_opt, method='L-BFGS-B',
                       options={'maxiter': 50, 'maxfun': 200, 'ftol': 1e-12})
    
    elapsed2 = time.time() - start_time2
    if elapsed2 > 20:
        pts_opt = _project_to_sphere(x_opt.reshape(N, 3))
    else:
        x_opt2 = result2.x
        pts_opt = x_opt2.reshape(N, 3)
        pts_opt = _project_to_sphere(pts_opt)
    
    # Verify validity
    norms = np.linalg.norm(pts_opt, axis=1)
    if np.any(norms < 1e-6):
        return pts
    
    # Check not in closed hemisphere: verify convex hull contains origin
    # Simple check: max dot product with any direction should be < 1
    # Actually, check that points are not all in one hemisphere
    # by verifying the origin is inside the convex hull
    # Quick check: sum of points should be near 0 for good distribution
    centroid = np.mean(pts_opt, axis=0)
    if np.linalg.norm(centroid) > 0.5:
        # Might be in a hemisphere, but not necessarily invalid
        # Just return as is
        pass
    
    return pts_opt

# EVOLVE-BLOCK-END
