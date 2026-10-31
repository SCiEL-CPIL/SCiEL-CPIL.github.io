"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import Delaunay
import itertools

N = 11
H = np.sqrt(3) / 2
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute all C(11,3)=165 triangle index triplets for final verification
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    # bary columns are weights for V0, V1, V2
    return bary[:, 0:1] * V0 + bary[:, 1:2] * V1 + bary[:, 2:3] * V2

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 3)."""
    # Inverse of the transformation
    # x = b1*1 + b2*0.5
    # y = b2*H
    b2 = cart[:, 1] / H
    b1 = cart[:, 0] - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack((b0, b1, b2))

def _clamp_bary(bary, eps=1e-8):
    """Clamp barycentric coordinates to be valid (non-negative, sum to 1)."""
    b = bary.copy()
    # Clip small negatives
    b[b < 0] = 0
    # Normalize
    s = np.sum(b, axis=1, keepdims=True)
    s[s == 0] = 1.0
    b /= s
    return b

def _all_areas_cart(pts):
    """Vectorized computation of all 165 triangle areas from Cartesian points."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area_cart(pts):
    """Minimum area among all 165 triangles."""
    return float(np.min(_all_areas_cart(pts)))

def _delaunay_min_area(pts):
    """
    Compute the minimum area triangle using Delaunay triangulation.
    This is a lower bound on the global min area, but usually the global min is in the Delaunay set.
    """
    try:
        tri = Delaunay(pts)
        areas = np.zeros(len(tri.simplices))
        for i, s in enumerate(tri.simplices):
            pi, pj, pk = pts[s[0]], pts[s[1]], pts[s[2]]
            areas[i] = 0.5 * np.abs(np.cross(pj - pi, pk - pi))
        return float(np.min(areas))
    except Exception:
        return 0.0

def _obj_smooth_bary(flat, beta=50.0):
    """
    Objective function for L-BFGS-B.
    Input: flat array of size 2*N (u, v for each point, w is implied).
    Returns: Negative of the soft-min area of Delaunay triangles.
    """
    # Reconstruct barycentric coordinates
    bary = np.zeros((N, 3))
    bary[:, 0] = 1.0 - flat[0:N] - flat[N:2*N]
    bary[:, 1] = flat[0:N]
    bary[:, 2] = flat[N:2*N]
    
    # Clamp to valid barycentric space
    bary = _clamp_bary(bary)
    
    # Convert to Cartesian
    pts = _bary_to_cart(bary)
    
    # Compute areas of Delaunay triangles
    try:
        tri = Delaunay(pts)
        areas = np.zeros(len(tri.simplices))
        for i, s in enumerate(tri.simplices):
            pi, pj, pk = pts[s[0]], pts[s[1]], pts[s[2]]
            areas[i] = 0.5 * np.abs(np.cross(pj - pi, pk - pi))
        
        # Soft min approximation
        m = np.min(areas)
        shifted = -beta * (areas - m)
        shifted = np.clip(shifted, -400, 400)
        log_sum = np.log(np.sum(np.exp(shifted)))
        soft_min_val = m - log_sum / beta
        
        return -soft_min_val
    except Exception:
        return 1.0 # Large penalty for invalid configurations

def _init_structured_bary():
    """Generate a structured initial set of points in barycentric coordinates."""
    # Start with a hexagonal-like packing
    pts_cart = []
    # Center
    pts_cart.append([1/3.0, H/3.0])
    
    # Ring 1
    r1 = 0.25
    for k in range(6):
        angle = k * np.pi / 3
        x = 1/3.0 + r1 * np.cos(angle)
        y = H/3.0 + r1 * np.sin(angle)
        # Check if inside, if not, project or skip. 
        # For simplicity, let's just place them and let the optimizer fix positions.
        pts_cart.append([x, y])
        
    # Ring 2 (approximate)
    r2 = 0.45
    for k in range(3):
        angle = k * 2 * np.pi / 3 + np.pi/6
        x = 1/3.0 + r2 * np.cos(angle)
        y = H/3.0 + r2 * np.sin(angle)
        pts_cart.append([x, y])
        
    # Pad if necessary
    while len(pts_cart) < N:
        pts_cart.append([1/3.0, H/3.0])
        
    pts_cart = np.array(pts_cart[:N])
    # Project inside triangle
    bary = _cart_to_bary(pts_cart)
    bary = _clamp_bary(bary)
    
    # Extract u, v (b1, b2)
    return np.concatenate([bary[:, 1], bary[:, 2]])

def _init_random_bary(seed):
    """Generate random points in barycentric coordinates."""
    rng = np.random.default_rng(seed)
    bary = np.zeros((N, 3))
    # Dirichlet distribution with alpha=1 gives uniform distribution in triangle
    bary = rng.dirichlet(alpha=np.ones(3), size=N)
    # Extract u, v
    return np.concatenate([bary[:, 1], bary[:, 2]])

def _local_polish_active_set(pts, deadline, max_iter=50):
    """
    Custom local search: identify the smallest triangles (active set) and move points to increase their area.
    """
    current = pts.copy()
    best_score = _min_area_cart(current)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        areas = _all_areas_cart(current)
        min_area = np.min(areas)
        
        # Identify triangles with area close to min (active constraints)
        threshold = min_area * 1.05
        active_idx = np.where(areas < threshold)[0]
        
        if len(active_idx) == 0:
            break
            
        # Calculate gradient of min area w.r.t points? 
        # Approximation: For each active triangle, try to move its vertices apart slightly.
        # This is a heuristic "push" step.
        
        # Instead of complex gradients, we can use a small Nelder-Mead on the active points only?
        # Or just a small random perturbation guided by the active triangles.
        
        # Let's try a simple coordinate ascent on the points involved in the smallest triangle.
        smallest_idx = np.argmin(areas)
        i, j, k = _TRI_IDX[smallest_idx]
        
        # Try small moves for points i, j, k
        improved = False
        step = 0.005
        for _ in range(20):
            if time.time() > deadline:
                break
            # Move i
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    trial = current.copy()
                    trial[i] = _clamp_bary(_cart_to_bary(trial[i] + np.array([dx, dy])))[0]
                    # Need to convert back to cart for area calc
                    trial_cart = _bary_to_cart(_clamp_bary(_cart_to_bary(trial)))
                    s = _min_area_cart(trial_cart)
                    if s > best_score + 1e-12:
                        current = trial
                        best_score = s
                        improved = True
            if not improved:
                break
        if not improved:
            break
        step *= 0.8
            
    return current, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 115.0 # Safety margin
    
    def tleft():
        return deadline - time.time()
        
    best_pts_bary = None
    best_score = -1.0
    
    # Initializations
    inits = [_init_structured_bary()]
    for s in [1, 42, 100, 200, 300, 500, 700, 900, 1200, 1500, 2000, 3000, 5000, 7000, 10000]:
        inits.append(_init_random_bary(s))
        
    # Phase 1: L-BFGS-B Optimization
    for init in inits:
        if tleft() < 10:
            break
        try:
            # Use L-BFGS-B
            res = minimize(
                _obj_smooth_bary, init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Reconstruct and clamp
            bary = np.zeros((N, 3))
            bary[:, 0] = 1.0 - res.x[0:N] - res.x[N:2*N]
            bary[:, 1] = res.x[0:N]
            bary[:, 2] = res.x[N:2*N]
            bary = _clamp_bary(bary)
            pts_cart = _bary_to_cart(bary)
            
            score = _min_area_cart(pts_cart)
            
            if score > best_score:
                best_score = score
                best_pts_bary = bary.copy()
                
        except Exception:
            continue
            
    if best_pts_bary is None:
        best_pts_bary = _init_structured_bary()
        best_score = _min_area_cart(_bary_to_cart(_clamp_bary(best_pts_bary)))
        
    # Phase 2: Nelder-Mead Polish on Hard Objective (Cartesian)
    # Convert best bary to cart for NM
    best_pts_cart = _bary_to_cart(best_pts_bary)
    
    def _obj_nm(flat):
        pts = _bary_to_cart(_clamp_bary(_cart_to_bary(flat.reshape(-1, 2))))
        return -_min_area_cart(pts)
        
    if tleft() > 20:
        try:
            res = minimize(
                _obj_nm, best_pts_cart.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            opt_cart = _bary_to_cart(_clamp_bary(_cart_to_bary(res.x.reshape(-1, 2))))
            score = _min_area_cart(opt_cart)
            if score > best_score:
                best_score = score
                best_pts_bary = _cart_to_bary(opt_cart)
        except Exception:
            pass
            
    # Phase 3: Active Set Local Polish
    if tleft() > 15:
        best_pts_cart = _bary_to_cart(best_pts_bary)
        pts_polished, score_polished = _local_polish_active_set(best_pts_cart, deadline)
        if score_polished > best_score:
            best_score = score_polished
            best_pts_bary = _cart_to_bary(pts_polished)
            
    # Phase 4: Final L-BFGS polish on the best point found
    if tleft() > 10:
        init_flat = np.concatenate([best_pts_bary[:, 1], best_pts_bary[:, 2]])
        try:
            res = minimize(
                _obj_smooth_bary, init_flat,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            bary = np.zeros((N, 3))
            bary[:, 0] = 1.0 - res.x[0:N] - res.x[N:2*N]
            bary[:, 1] = res.x[0:N]
            bary[:, 2] = res.x[N:2*N]
            bary = _clamp_bary(bary)
            pts_cart = _bary_to_cart(bary)
            score = _min_area_cart(pts_cart)
            if score > best_score:
                best_score = score
                best_pts_bary = bary.copy()
        except Exception:
            pass
            
    # Return the best points in Cartesian coordinates
    return _bary_to_cart(best_pts_bary)
