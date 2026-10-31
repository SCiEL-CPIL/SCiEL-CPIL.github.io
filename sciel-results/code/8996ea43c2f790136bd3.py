"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.spatial import Delaunay

H = np.sqrt(3) / 2
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute all C(11,3)=165 triangle index triplets
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    # bary[:, 0] * V0 + bary[:, 1] * V1 + bary[:, 2] * V2
    # V0 is (0,0), V1 is (1,0), V2 is (0.5, H)
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = H * bary[:, 2]
    return np.column_stack((x, y))

def _cart_to_bary(pts):
    """Convert Cartesian (N, 2) to barycentric (N, 3)."""
    # x = b1 + 0.5*b2
    # y = H*b2  => b2 = y/H
    # b1 = x - 0.5*b2
    # b0 = 1 - b1 - b2
    b2 = pts[:, 1] / H
    b1 = pts[:, 0] - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack((b0, b1, b2))

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid simplex u,v,w >= 0, sum=1."""
    # Simple projection: clip negatives to 0, then renormalize
    bary_clipped = np.clip(bary, 0.0, 1.0)
    sums = bary_clipped.sum(axis=1, keepdims=True)
    # Avoid division by zero
    sums[sums == 0] = 1.0
    bary_clipped /= sums
    return bary_clipped

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
    """Minimum area among all triangles."""
    return float(np.min(_all_areas_cart(pts)))

def _delaunay_indices(pts):
    """Get indices of Delaunay triangles for the points."""
    try:
        tri = Delaunay(pts)
        return tri.simplices
    except Exception:
        # Fallback to all triangles if Delaunay fails (e.g., collinear points)
        return _TRI_IDX

def _soft_min_delaunay(bary_flat, beta=50.0):
    """
    Smooth objective based on Delaunay triangles.
    bary_flat: (2N,) array of (u,v) for each point, w=1-u-v.
    """
    bary = bary_flat.reshape(N, 2)
    w = 1.0 - bary[:, 0] - bary[:, 1]
    bary_full = np.column_stack((bary[:, 0], bary[:, 1], w))
    
    # Clamp to valid simplex
    bary_clipped = _clamp_bary(bary_full)
    
    # Convert to Cartesian for Delaunay
    pts = _bary_to_cart(bary_clipped)
    
    # Get Delaunay triangles
    try:
        simplices = Delaunay(pts).simplices
    except Exception:
        simplices = _TRI_IDX
        
    # Calculate areas of these specific triangles
    if len(simplices) == 0:
        return 0.0
        
    pi = pts[simplices[:, 0]]
    pj = pts[simplices[:, 1]]
    pk = pts[simplices[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    
    if len(areas) == 0:
        return 0.0
        
    m = np.min(areas)
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -400, 400)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return -(m - log_sum / beta)

def _obj_smooth(bary_flat, beta=50.0):
    """Objective for L-BFGS-B: negative soft-min area of Delaunay triangles."""
    return _soft_min_delaunay(bary_flat, beta)

def _obj_hard(bary_flat):
    """Objective for Nelder-Mead: negative hard min area."""
    bary = bary_flat.reshape(N, 2)
    w = 1.0 - bary[:, 0] - bary[:, 1]
    bary_full = np.column_stack((bary[:, 0], bary[:, 1], w))
    bary_clipped = _clamp_bary(bary_full)
    pts = _bary_to_cart(bary_clipped)
    return -_min_area_cart(pts)

def _init_hexagonal():
    """Generate a hexagonal grid-like initialization."""
    # Approximate optimal packing for 11 points
    # Centers of hexagons or similar
    bary = np.zeros((N, 3))
    
    # 3 corners, 3 midpoints, 1 center, 4 inner
    # This is a heuristic start
    m = 0.15
    # Corners (inset)
    bary[0] = [1-m, m/2, m/2]
    bary[1] = [m/2, 1-m, m/2]
    bary[2] = [m/2, m/2, 1-m]
    
    # Midpoints (inset)
    m2 = 0.1
    bary[3] = [(1-m2)/2, (1-m2)/2, m2]
    bary[4] = [(1-m2)/2, m2, (1-m2)/2]
    bary[5] = [m2, (1-m2)/2, (1-m2)/2]
    
    # Center
    bary[6] = [1/3, 1/3, 1/3]
    
    # 4 inner points forming a smaller square/diamond
    r = 0.1
    angles = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
    for i, a in enumerate(angles):
        # Approximate circular offset in barycentric space
        # This is tricky, so we just place them symmetrically
        bary[7+i] = [1/3 + r*np.cos(a)*0.5, 1/3 + r*np.sin(a)*0.5, 1/3 - r*(np.cos(a)+np.sin(a))*0.5]
        
    bary_clipped = _clamp_bary(bary)
    # Return (u, v) for optimization
    return bary_clipped[:, :2].flatten()

def _init_random(seed):
    """Random initialization inside triangle."""
    rng = np.random.default_rng(seed)
    bary = rng.random((N, 3))
    bary /= bary.sum(axis=1, keepdims=True)
    bary_clipped = _clamp_bary(bary)
    return bary_clipped[:, :2].flatten()

def _lbfgs_polish(bary_flat, beta, maxiter, deadline):
    """L-BFGS-B polish with soft-min objective."""
    try:
        res = minimize(
            _obj_smooth, bary_flat, args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        bary_opt = res.x.reshape(N, 2)
        w = 1.0 - bary_opt[:, 0] - bary_opt[:, 1]
        bary_full = np.column_stack((bary_opt[:, 0], bary_opt[:, 1], w))
        bary_clipped = _clamp_bary(bary_full)
        pts = _bary_to_cart(bary_clipped)
        return pts, _min_area_cart(pts)
    except Exception:
        bary_clipped = _clamp_bary(bary_flat.reshape(N, 2) if bary_flat.size == 2*N else bary_flat)
        # Handle case where input was already (N,3) or similar, but we expect (2N,)
        # If exception, return original points if possible
        if bary_flat.size == 2*N:
             b = bary_flat.reshape(N,2)
             w = 1-b[:,0]-b[:,1]
             bf = np.column_stack((b[:,0], b[:,1], w))
             bf = _clamp_bary(bf)
             pts = _bary_to_cart(bf)
             return pts, _min_area_cart(pts)
        return None, 0.0

def _nm_polish(bary_flat, deadline):
    """Nelder-Mead polish with hard objective."""
    try:
        res = minimize(
            _obj_hard, bary_flat,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        bary_opt = res.x.reshape(N, 2)
        w = 1.0 - bary_opt[:, 0] - bary_opt[:, 1]
        bary_full = np.column_stack((bary_opt[:, 0], bary_opt[:, 1], w))
        bary_clipped = _clamp_bary(bary_full)
        pts = _bary_to_cart(bary_clipped)
        return pts, _min_area_cart(pts)
    except Exception:
        if bary_flat.size == 2*N:
             b = bary_flat.reshape(N,2)
             w = 1-b[:,0]-b[:,1]
             bf = np.column_stack((b[:,0], b[:,1], w))
             bf = _clamp_bary(bf)
             pts = _bary_to_cart(bf)
             return pts, _min_area_cart(pts)
        return None, 0.0

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    def tleft():
        return deadline - time.time()
    
    # Generate diverse initializations in barycentric (u,v) space
    inits = [_init_hexagonal()]
    for s in [1, 7, 13, 29, 42, 100, 200, 300, 500, 700, 900, 1200, 1500, 2000]:
        inits.append(_init_random(s))
    
    best_pts = None
    best_score = -1.0
    
    # Phase 1: Multi-start L-BFGS with multiple betas
    for init in inits:
        if tleft() < 12:
            break
        for beta in [30.0, 60.0, 120.0]:
            if tleft() < 5:
                break
            pts, sc = _lbfgs_polish(init, beta, 500, deadline)
            if pts is not None and sc > best_score:
                best_score = sc
                best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback
        b = _init_hexagonal().reshape(N, 2)
        w = 1-b[:,0]-b[:,1]
        bf = _clamp_bary(np.column_stack((b[:,0], b[:,1], w)))
        best_pts = _bary_to_cart(bf)
        best_score = _min_area_cart(best_pts)
        
    # Convert best_pts to barycentric for further polishing
    best_bary = _cart_to_bary(best_pts)
    best_bary_flat = best_bary[:, :2].flatten()
    
    # Phase 2: Nelder-Mead polish on best
    if tleft() > 15:
        pts, sc = _nm_polish(best_bary_flat, deadline)
        if pts is not None and sc > best_score:
            best_score = sc
            best_pts = pts.copy()
            best_bary_flat = _cart_to_bary(pts)[:, :2].flatten()
            
    # Phase 3: Differential Evolution / Random Restart with Polish
    # We use random perturbations in barycentric space
    rng = np.random.default_rng(99999)
    
    for trial in range(30):
        if tleft() < 15:
            break
            
        # Perturb the best solution
        scale = 0.05 + 0.01 * trial
        pert_bary = best_bary_flat.reshape(N, 2) + rng.normal(0, scale, size=(N, 2))
        
        # Also try some fresh random starts occasionally
        if trial % 5 == 0:
            pert_bary = _init_random(int(time.time()*1000) % 10000)
            
        pert_flat = pert_bary.flatten()
        
        # L-BFGS polish
        pts, sc = _lbfgs_polish(pert_flat, 60.0, 300, deadline)
        if pts is not None and sc > best_score:
            best_score = sc
            best_pts = pts.copy()
            best_bary_flat = _cart_to_bary(pts)[:, :2].flatten()
            
        # NM polish if time allows
        if tleft() > 10:
            pts, sc = _nm_polish(best_bary_flat, deadline)
            if pts is not None and sc > best_score:
                best_score = sc
                best_pts = pts.copy()
                best_bary_flat = _cart_to_bary(pts)[:, :2].flatten()
                
    # Phase 4: Final Coordinate Descent Refinement
    # Simple local search on Cartesian coordinates
    if tleft() > 10:
        current = best_pts.copy()
        score = _min_area_cart(current)
        steps = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        
        for step in steps:
            improved = True
            while improved and tleft() > 5:
                improved = False
                for i in range(N):
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            if dx == 0 and dy == 0:
                                continue
                            trial = current.copy()
                            trial[i] = current[i] + np.array([dx, dy])
                            # Clamp to triangle
                            bary_trial = _cart_to_bary(trial)
                            bary_clipped = _clamp_bary(bary_trial)
                            trial_clamped = _bary_to_cart(bary_clipped)
                            
                            s = _min_area_cart(trial_clamped)
                            if s > score + 1e-15:
                                current = trial_clamped
                                score = s
                                improved = True
        if score > best_score:
            best_score = score
            best_pts = current
            
    return best_pts
# EVOLVE-BLOCK-END
