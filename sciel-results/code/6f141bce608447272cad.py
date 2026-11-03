import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Triangle vertices in Cartesian coordinates
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples (i, j, k) with i < j < k
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) to cartesian (x, y).
    bary shape: (N, 2)
    Returns: (N, 2)
    """
    u = bary[:, 0]
    v = bary[:, 1]
    # P = u * V1 + v * V2 (since V0 is origin)
    x = u * V1[0] + v * V2[0]
    y = u * V1[1] + v * V2[1]
    return np.column_stack([x, y])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid region u>=0, v>=0, u+v<=1."""
    bary = bary.copy()
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _min_triangle_area(points):
    """Compute the smallest area among all triangles formed by triples of points.
    points: (N, 2) array of Cartesian coordinates.
    """
    pi = points[_IDX_TRIPLES[:, 0]]
    pj = points[_IDX_TRIPLES[:, 1]]
    pk = points[_IDX_TRIPLES[:, 2]]
    
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _objective_bary(x):
    """Objective function for optimization. Returns negative min area (to maximize).
    x: flattened barycentric coordinates (2*N,)
    """
    bary = x.reshape(-1, 2)
    # Check validity quickly
    if np.any(bary < 0) or np.any(bary[:, 0] + bary[:, 1] > 1.0 + 1e-9):
        # If slightly out of bounds, clamp. If far out, penalize.
        # Nelder-Mead might step out. Clamping is safer for continuity.
        bary = _clamp_bary(bary)
    
    points = _bary_to_cart(bary)
    min_area = _min_triangle_area(points)
    return -min_area

def _generate_grid_initializations(n_grids=5):
    """Generate initial configurations based on a grid of points in the triangle."""
    inits = []
    # Create a grid of barycentric points
    # We want to distribute N points.
    # A simple strategy: take a coarse grid and select N points that are well-separated.
    
    # Grid size
    k = 5
    grid_points = []
    for i in range(k):
        for j in range(k - i):
            u = i / (k - 1) if k > 1 else 0
            v = j / (k - 1) if k > 1 else 0
            if u + v <= 1.0:
                grid_points.append([u, v])
    grid_points = np.array(grid_points)
    
    # We need to pick N points from this grid or interpolate.
    # Better: Use a specific structured pattern.
    # Pattern 1: Vertices + Midpoints + Centroid + Others
    # Let's just use random sampling from the grid with some jitter for diversity.
    
    rng = np.random.RandomState(42)
    for _ in range(n_grids):
        # Select N random points from the grid
        if len(grid_points) >= N:
            idx = rng.choice(len(grid_points), N, replace=False)
            bary = grid_points[idx].copy()
            # Add small jitter to break symmetry
            bary += rng.uniform(-0.05, 0.05, bary.shape)
            bary = _clamp_bary(bary)
            inits.append(bary)
        else:
            # Fallback to random
            bary = rng.uniform(0, 1, (N, 2))
            bary = _clamp_bary(bary)
            inits.append(bary)
            
    return inits

def _generate_random_initializations(n_random=10, seed=0):
    """Generate random initial configurations."""
    inits = []
    rng = np.random.RandomState(seed)
    for _ in range(n_random):
        bary = rng.uniform(0, 1, (N, 2))
        bary = _clamp_bary(bary)
        inits.append(bary)
    return inits

def _generate_structured_initialization():
    """A specific structured initialization based on known good distributions."""
    # Try to place points at vertices, edge midpoints, and centroid, then fill in.
    # Vertices: (0,0), (1,0), (0,1) in barycentric
    # Midpoints: (0.5, 0), (0, 0.5), (0.5, 0.5)
    # Centroid: (1/3, 1/3)
    # This gives 6 points. We need 5 more.
    # Add points at 1/4 and 3/4 along edges? Or inner grid?
    
    points = [
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], # Vertices
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], # Midpoints
        [1/3, 1/3] # Centroid
    ]
    
    # Add 5 more points. Let's try a finer grid layer.
    # Points at distance 1/4 from vertices?
    # (0.25, 0), (0, 0.25), (0.75, 0), (0, 0.75), (0.25, 0.25)
    points.extend([
        [0.25, 0.0],
        [0.0, 0.25],
        [0.75, 0.0],
        [0.0, 0.75],
        [0.25, 0.25]
    ])
    
    bary = np.array(points)
    bary = _clamp_bary(bary)
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Leave 5s margin for return and overhead
    
    def time_left():
        return deadline - time.time()
        
    best_score = -np.inf
    best_bary = None
    
    # Collect initializations
    inits = []
    inits.append(_generate_structured_initialization())
    inits.extend(_generate_grid_initializations(n_grids=5))
    inits.extend(_generate_random_initializations(n_random=10, seed=42))
    
    # Options for Nelder-Mead
    # maxiter: Max iterations per run
    # xatol, fatol: Tolerances
    nm_options = {
        'maxiter': 200,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True
    }
    
    for bary_init in inits:
        if time_left() < 2.0:
            break
            
        x0 = bary_init.flatten()
        
        try:
            res = minimize(
                _objective_bary, 
                x0, 
                method='Nelder-Mead', 
                options=nm_options
            )
            
            bary_res = _clamp_bary(res.x.reshape(-1, 2))
            points_res = _bary_to_cart(bary_res)
            score = _min_triangle_area(points_res)
            
            if score > best_score:
                best_score = score
                best_bary = bary_res.copy()
                
        except Exception:
            pass
            
    # If no valid result found (should not happen), use a fallback
    if best_bary is None:
        best_bary = _generate_structured_initialization()
        best_score = _min_triangle_area(_bary_to_cart(best_bary))
        
    # Final check: ensure points are strictly inside (or on boundary)
    best_bary = _clamp_bary(best_bary)
    best_points = _bary_to_cart(best_bary)
    
    return best_points
