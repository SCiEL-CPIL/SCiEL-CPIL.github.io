"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross product|
                dx1 = pts[j][0] - pts[i][0]
                dy1 = pts[j][1] - pts[i][1]
                dx2 = pts[k][0] - pts[i][0]
                dy2 = pts[k][1] - pts[i][1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    return min_area

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to Cartesian."""
    # Point = u*(1,0) + v*(0.5, sqrt(3)/2) + (1-u-v)*(0,0)
    # = (u + 0.5*v, (sqrt(3)/2)*v)
    u = bary[0]
    v = bary[1]
    return np.array([u + 0.5 * v, (np.sqrt(3) / 2) * v])

def _cart_to_bary(pt):
    """Convert Cartesian to barycentric (u, v)."""
    x, y = pt
    v = y / (np.sqrt(3) / 2)
    u = x - 0.5 * v
    return np.array([u, v])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to valid region."""
    u = max(0.0, min(1.0, bary[0]))
    v = max(0.0, min(1.0, bary[1]))
    if u + v > 1.0:
        scale = 1.0 / (u + v)
        u *= scale
        v *= scale
    return np.array([u, v])

def _objective(bary_flat):
    """Objective function: negative of min triangle area (to minimize)."""
    bary = bary_flat.reshape(N, 2)
    # Clamp to valid region
    bary = np.array([_clamp_bary(b) for b in bary])
    pts = np.array([_bary_to_cart(b) for b in bary])
    return -_min_triangle_area(pts)

def _initial_config_symmetric():
    """Create a symmetric initial configuration."""
    bary = []
    # 3 vertices
    bary.append([0.0, 0.0])
    bary.append([1.0, 0.0])
    bary.append([0.0, 1.0])
    # 3 edge midpoints
    bary.append([0.5, 0.0])
    bary.append([0.0, 0.5])
    bary.append([0.5, 0.5])
    # Centroid
    bary.append([1/3, 1/3])
    # 4 more: at 1/4 and 3/4 along edges, or other symmetric positions
    # Let's try: points at (1/4, 1/4), (3/4, 1/4), (1/4, 3/4), (3/4, 3/4) in barycentric
    # But need u+v <= 1
    bary.append([0.25, 0.25])
    bary.append([0.5, 0.25])
    bary.append([0.25, 0.5])
    bary.append([0.5, 0.5])  # Already have this, let me adjust
    # Let me redo: 11 points
    # Vertices: (0,0), (1,0), (0,1) in bary
    # Edge midpoints: (0.5,0), (0,0.5), (0.5,0.5)
    # Centroid: (1/3, 1/3)
    # 4 more: let's use (1/4, 1/4), (3/4, 1/4), (1/4, 3/4), and (0.5, 0.25) or similar
    bary = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.0, 0.5],
        [0.5, 0.5],
        [1/3, 1/3],
        [0.25, 0.25],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.5, 0.25]
    ]
    return np.array(bary)

def _random_config(rng):
    """Generate a random valid configuration."""
    bary = []
    for _ in range(N):
        u = rng.random()
        v = rng.random() * (1.0 - u)
        bary.append([u, v])
    return np.array(bary)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    best_bary = None
    best_score = -float('inf')
    
    # Start with symmetric configuration
    bary0 = _initial_config_symmetric()
    bary0_flat = bary0.flatten()
    
    # Optimize from symmetric start
    result = minimize(
        _objective,
        bary0_flat,
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    
    if result.fun < best_score:
        best_score = result.fun
        best_bary = result.x.reshape(N, 2)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        # Not enough time for restarts, return best
        bary = np.array([_clamp_bary(b) for b in best_bary])
        pts = np.array([_bary_to_cart(b) for b in bary])
        return pts
    
    # Multiple random restarts
    rng = np.random.RandomState(42)
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        bary_rand = _random_config(rng)
        bary_rand_flat = bary_rand.flatten()
        
        try:
            result = minimize(
                _objective,
                bary_rand_flat,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_bary = result.x.reshape(N, 2)
        except Exception:
            continue
    
    # Final clamp and convert
    bary = np.array([_clamp_bary(b) for b in best_bary])
    pts = np.array([_bary_to_cart(b) for b in bary])
    
    return pts
# EVOLVE-BLOCK-END

