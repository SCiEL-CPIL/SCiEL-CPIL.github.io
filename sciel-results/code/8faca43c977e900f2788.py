"""Heilbronn triangles: place 11 points in unit-area equilateral triangle to maximize min triangle area."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4  # area of the triangle

def _bary_to_cart(bary):
    """Convert barycentric coords (n,3) to cartesian (n,2)."""
    bary = np.clip(bary, 0, 1)
    bary = bary / bary.sum(axis=1, keepdims=True)
    return bary @ np.array([V0, V1, V2])

def _cart_to_bary(pts):
    """Convert cartesian (n,2) to barycentric (n,3)."""
    # Solve: p = a*V0 + b*V1 + c*V2, a+b+c=1
    # Using the linear system
    A = np.array([
        [V1[0] - V0[0], V2[0] - V0[0]],
        [V1[1] - V0[1], V2[1] - V0[1]]
    ])
    invA = np.linalg.inv(A)
    rel = pts - V0
    bc = rel @ invA.T
    a = 1.0 - bc[:, 0] - bc[:, 1]
    return np.column_stack([a, bc[:, 0], bc[:, 1]])

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                cross = (pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) - \
                        (pts[j][1] - pts[i][1]) * (pts[k][0] - pts[i][0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _min_triangle_area_fast(pts):
    """Faster vectorized min triangle area computation."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vectorize over k
            diffs_j = pts[j+1:] - pts[i]
            diffs_i_to_j = pts[j] - pts[i]
            crosses = diffs_j[:, 0] * (pts[j+1:, 1] - pts[i, 1]) - \
                      diffs_j[:, 1] * (pts[j+1:, 0] - pts[i, 0])
            areas = 0.5 * np.abs(crosses)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _neg_min_area(bary):
    """Objective: negative of minimum triangle area (to minimize)."""
    pts = _bary_to_cart(bary)
    return -_min_triangle_area_fast(pts)

def _make_initial_config(seed):
    """Create a symmetric initial configuration using barycentric coords."""
    rng = np.random.RandomState(seed)
    # Start with a reasonable spread
    bary = np.zeros((N, 3))
    # Place some points near vertices and some in center
    # Symmetric pattern
    bary[0] = [0.85, 0.075, 0.075]  # near V0
    bary[1] = [0.075, 0.85, 0.075]  # near V1
    bary[2] = [0.075, 0.075, 0.85]  # near V2
    bary[3] = [0.333, 0.333, 0.334]  # center
    bary[4] = [0.5, 0.25, 0.25]
    bary[5] = [0.25, 0.5, 0.25]
    bary[6] = [0.25, 0.25, 0.5]
    bary[7] = [0.6, 0.2, 0.2]
    bary[8] = [0.2, 0.6, 0.2]
    bary[9] = [0.2, 0.2, 0.6]
    # Add small random perturbation
    bary += rng.uniform(-0.02, 0.02, size=bary.shape)
    bary = np.clip(bary, 0.01, 0.98)
    bary = bary / bary.sum(axis=1, keepdims=True)
    return bary

def construct():
    deadline = time.time() + 140  # safety margin
    
    best_bary = None
    best_score = -float('inf')
    
    # Try multiple seeds with Nelder-Mead
    for seed in range(20):
        if time.time() > deadline:
            break
        
        bary0 = _make_initial_config(seed)
        
        # Nelder-Mead optimization
        result = minimize(
            _neg_min_area,
            bary0,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'maxfev': 50000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if result.success or result.fun < -best_score:
            bary_opt = result.x
            bary_opt = np.clip(bary_opt, 0.001, 0.998)
            bary_opt = bary_opt / bary_opt.sum(axis=1, keepdims=True)
            
            pts = _bary_to_cart(bary_opt)
            score = _min_triangle_area_fast(pts)
            
            if score > best_score:
                best_score = score
                best_bary = bary_opt
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    if best_bary is None:
        # Fallback: return initial config
        best_bary = _make_initial_config(0)
    
    pts = _bary_to_cart(best_bary)
    return pts
