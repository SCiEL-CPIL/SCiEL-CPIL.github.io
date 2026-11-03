"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Precompute all triplets
_TRIPLETS = list(combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

DEADLINE = 110  # Safety margin for 150s limit


def _bary_to_cart(bc):
    """Convert barycentric (b, c) to cartesian. a = 1-b-c.
    Vertices: A=(0,0), B=(1,0), C=(0.5, S3H)
    cart = b*B + c*C = (b + 0.5*c, S3H*c)
    """
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = S3H * c
    return np.column_stack([x, y])


def _project_bary(bc):
    """Project barycentric coords to valid simplex: b>=0, c>=0, b+c<=1."""
    b = bc[:, 0].copy()
    c = bc[:, 1].copy()
    # Clip non-negative
    b = np.clip(b, 0, None)
    c = np.clip(c, 0, None)
    # Clip b+c <= 1
    s = b + c
    mask = s > 1.0
    if np.any(mask):
        scale = 1.0 / s[mask]
        b[mask] *= scale
        c[mask] *= scale
    return np.column_stack([b, c])


def _min_area_bary(bc):
    """Min triangle area given barycentric coords."""
    cart = _bary_to_cart(bc)
    pi = cart[_T_I]
    pj = cart[_T_J]
    pk = cart[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def construct():
    start_time = time.time()
    
    def _elapsed_ok(margin=5):
        return (time.time() - start_time) < (DEADLINE - margin)

    best_bc = None
    best_area = 0.0

    # Generate multiple symmetric initializations
    inits = []
    
    # Init 1: Triangular grid (10 points) + Centroid
    grid_points = []
    # Bottom row (y=0)
    for i in range(4):
        grid_points.append([i/3.0, 0.0])
    # Second row (y=S3H/3)
    for i in range(3):
        grid_points.append([0.5/3.0 + i/3.0, S3H/3.0])
    # Third row (y=2*S3H/3)
    for i in range(2):
        grid_points.append([1.0/3.0 + i/3.0, 2*S3H/3.0])
    # Top point (y=S3H)
    grid_points.append([0.5, S3H])
    # Centroid
    grid_points.append([0.5, S3H/3.0])
    
    init_cart = np.array(grid_points)
    c_init = init_cart[:, 1] / S3H
    b_init = init_cart[:, 0] - 0.5 * c_init
    init_bc = np.column_stack([b_init, c_init])
    init_bc = _project_bary(init_bc)
    inits.append(init_bc)
    
    # Init 2-4: Random uniform in triangle
    np.random.seed(42)
    for _ in range(3):
        u = np.sqrt(np.random.rand(N))
        v = np.random.rand(N)
        b_rand = u * (1 - v)
        c_rand = u * v
        bc_init = np.column_stack([b_rand, c_rand])
        inits.append(bc_init)

    # Run local search on each init
    for i, init in enumerate(inits):
        if not _elapsed_ok(10):
            break
            
        def obj_local(flat_bc):
            bc_new = _project_bary(flat_bc.reshape(-1, 2))
            return -_min_area_bary(bc_new)
            
        try:
            # Use Nelder-Mead for fast local optimization
            res = minimize(
                obj_local,
                init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False}
            )
            
            new_bc = _project_bary(res.x.reshape(-1, 2))
            new_area = _min_area_bary(new_bc)
            
            if new_area > best_area:
                best_area = new_area
                best_bc = new_bc
                
        except Exception:
            pass

    # Fallback if no optimization worked
    if best_bc is None:
        best_bc = inits[0]
        best_area = _min_area_bary(best_bc)

    return _bary_to_cart(best_bc)
