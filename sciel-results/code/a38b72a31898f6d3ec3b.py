import numpy as np
import time
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = 140.0

def _hex_vertices(cx, cy, angle_deg):
    """Return vertices of a unit hexagon centered at (cx, cy) with rotation angle_deg."""
    angle_rad = np.deg2rad(angle_deg)
    # Vertices of unit hexagon at origin (flat-top)
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    base_v = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    verts = base_v @ R.T + np.array([cx, cy])
    return verts

def _hex_norm(x, y):
    """Hexagonal norm for flat-top hexagon. Max of |x|, |x/2 + sqrt(3)y/2|, |-x/2 + sqrt(3)y/2|"""
    return np.maximum(np.abs(x), np.maximum(np.abs(x/2 + np.sqrt(3)*y/2), np.abs(-x/2 + np.sqrt(3)*y/2)))

def _calculate_s_required(params):
    """Calculate the minimum side length S of the outer hexagon required to contain all inner hexagons."""
    n = len(params) // 3
    s_max = 0.0
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        verts = _hex_vertices(cx, cy, ang)
        # Compute norm for all vertices
        norms = _hex_norm(verts[:, 0], verts[:, 1])
        s_max = max(s_max, np.max(norms))
    return s_max

def _calculate_overlap_penalty(params):
    """Calculate penalty for overlaps between inner hexagons."""
    n = len(params) // 3
    penalty = 0.0
    min_dist = np.sqrt(3)
    for i in range(n):
        cx_i = params[3*i]
        cy_i = params[3*i+1]
        for j in range(i+1, n):
            cx_j = params[3*j]
            cy_j = params[3*j+1]
            d = np.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
            if d < min_dist:
                penalty += (min_dist - d)**2
    return penalty

def _objective(params):
    if time.time() > DEADLINE:
        return 1e10
    s_req = _calculate_s_required(params)
    penalty = _calculate_overlap_penalty(params)
    return s_req + 100.0 * penalty

def _check_validity(inner, center, side, angle):
    """Check if the packing is valid."""
    # Check overlaps
    n = len(inner)
    min_dist = np.sqrt(3)
    for i in range(n):
        x_i, y_i, _ = inner[i]
        for j in range(i+1, n):
            x_j, y_j, _ = inner[j]
            d = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
            if d < min_dist - 1e-9:
                return False
    
    # Check containment
    # Outer hexagon center (cx, cy), side S, angle 0 (assumed)
    cx, cy = center
    S = side
    # Shift inner coordinates relative to outer center
    for i in range(n):
        x_i, y_i, ang_i = inner[i]
        verts = _hex_vertices(x_i - cx, y_i - cy, ang_i)
        norms = _hex_norm(verts[:, 0], verts[:, 1])
        if np.max(norms) > S + 1e-9:
            return False
    return True

def _get_initial_guesses():
    """Generate several initial guesses for the optimization."""
    guesses = []
    
    # Guess 1: Compact cluster
    # Center
    # 6 around
    # 4 more
    base = []
    base.append([0.0, 0.0, 0.0])
    # 6 neighbors
    for k in range(6):
        ang = np.deg2rad(k * 60)
        x = np.sqrt(3) * np.cos(ang)
        y = np.sqrt(3) * np.sin(ang)
        base.append([x, y, 0.0])
    
    # Add 4 more.
    # Place them on the "corners" of the 7-cluster.
    # The 7-cluster has vertices at distance 2 from center?
    # Let's just push them out from 4 of the outer hexagons.
    outer_indices = [1, 2, 4, 5] # Indices in base (0 is center, 1-5 are outer)
    # Wait, base has 7 elements. Indices 1..6 are outer.
    # Let's pick 4 of them.
    for idx in [1, 3, 5, 2]:
        x, y, _ = base[idx]
        # Direction from origin
        r = np.sqrt(x*x + y*y)
        if r == 0: continue
        dx = x/r
        dy = y/r
        new_x = x + np.sqrt(3)*dx
        new_y = y + np.sqrt(3)*dy
        base.append([new_x, new_y, 0.0])
    
    guesses.append(base)
    
    # Guess 2: Linear arrangement (fallback, but bad)
    lin = []
    for k in range(N):
        lin.append([k*np.sqrt(3), 0.0, 0.0])
    guesses.append(lin)
    
    # Guess 3: Random
    np.random.seed(42)
    rand = []
    for _ in range(N):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        a = np.random.uniform(0, 180)
        rand.append([x, y, a])
    guesses.append(rand)
    
    return guesses

def construct():
    global DEADLINE
    DEADLINE = time.time() + 140.0
    
    best_score = float('inf')
    best_params = None
    
    guesses = _get_initial_guesses()
    
    for guess in guesses:
        if time.time() > DEADLINE - 5.0:
            break
        
        params0 = np.array(guess).flatten()
        
        # Use L-BFGS-B
        try:
            res = minimize(_objective, params0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
        except Exception:
            continue
            
        if res.success or res.fun < best_score:
            # Validate
            n = N
            inner = []
            for i in range(n):
                inner.append([res.x[3*i], res.x[3*i+1], res.x[3*i+2]])
            
            # Calculate S
            s_req = _calculate_s_required(res.x)
            
            # Check validity
            if _check_validity(inner, (0.0, 0.0), s_req, 0.0):
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
                    # We can also try to tighten S by re-calculating exactly
                    # The objective uses s_req, so best_params should be good.
    
    if best_params is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    n = N
    inner = []
    for i in range(n):
        inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    
    s_req = _calculate_s_required(best_params)
    
    return inner, (0.0, 0.0), s_req, 0.0

