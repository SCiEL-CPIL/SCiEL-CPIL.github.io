# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _score(points):
    """Compute (d_min/d_max)^2"""
    d = pdist(points)
    d_min = d.min()
    d_max = d.max()
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _neg_score(points):
    return -_score(points)

def _refine(points, max_time=5.0):
    """Refine points using Nelder-Mead, respecting time budget"""
    t0 = time.time()
    best_points = points.copy()
    best_score = _score(points)
    
    # Flatten for optimization
    x0 = points.flatten()
    
    def obj(x):
        p = x.reshape(-1, 3)
        return -_score(p)
    
    result = minimize(obj, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
    
    if time.time() - t0 > max_time:
        return best_points, best_score
    
    if result.fun < -best_score:
        best_points = result.x.reshape(-1, 3)
        best_score = -result.fun
    
    return best_points, best_score

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    deadline = time.time() + 580  # Safety margin before 630s limit
    
    # Start with known good symmetric configurations
    candidates = []
    
    # Configuration 1: Based on cuboctahedron + 2 points
    # Cuboctahedron has 12 vertices, add 2 more
    # Try: 14 points as a known optimal packing
    # The optimal 14-point configuration in 3D is known to have specific structure
    
    # Let's use multiple diverse initializations
    
    # Init 1: Random with seed
    np.random.seed(42)
    candidates.append(np.random.randn(n, d))
    
    # Init 2: Spherical Fibonacci-like distribution
    golden_ratio = (1 + np.sqrt(5)) / 2
    theta = 2 * np.pi * np.arange(n) / golden_ratio
    z = 1 - 2 * np.arange(n) / n
    r = np.sqrt(1 - z**2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    candidates.append(np.column_stack([x, y, z]))
    
    # Init 3: Icosahedron (12) + 2 center-ish points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico[0])  # Normalize to unit sphere
    # Add 2 more points - try center and one more
    center = np.zeros((1, 3))
    extra = np.array([[0, 0, 1.5]])
    candidates.append(np.vstack([ico, center, extra]))
    
    # Init 4: Octahedron (6) + cube (8) = 14, scaled appropriately
    oct_pts = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ])
    cube_pts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ]) / np.sqrt(3)  # Normalize cube to unit sphere
    candidates.append(np.vstack([oct_pts, cube_pts]))
    
    # Init 5: Random spherical
    np.random.seed(123)
    vecs = np.random.randn(n, d)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    candidates.append(vecs)
    
    # Init 6: Another random
    np.random.seed(456)
    vecs = np.random.randn(n, d)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    candidates.append(vecs)
    
    # Evaluate all candidates and pick best
    best_points = None
    best_score = -1
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 60:
            break
        # Quick score
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()
    
    # Refine the best candidate
    if best_points is not None and time.time() < deadline - 30:
        refined, refined_score = _refine(best_points, max_time=min(30, deadline - time.time() - 10))
        if refined_score > best_score:
            best_points = refined
            best_score = refined_score
    
    # Try refining a few more candidates if time permits
    if time.time() < deadline - 60:
        for cand in candidates:
            if time.time() > deadline - 30:
                break
            if np.allclose(cand, best_points, atol=1e-8):
                continue
            refined, refined_score = _refine(cand, max_time=min(20, deadline - time.time() - 15))
            if refined_score > best_score:
                best_points = refined
                best_score = refined_score
    
    # Final verification
    if not np.all(np.isfinite(best_points)):
        # Fallback to a known valid configuration
        np.random.seed(42)
        best_points = np.random.randn(n, d)
    
    return best_points

# EVOLVE-BLOCK-END
