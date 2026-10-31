import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
H = np.sqrt(3) / 2

# Pre-compute indices for combinations of 3 from 11
_tri_indices = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

# Triangle vertices in cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

def _bary_to_cart(bary):
    """Convert barycentric coords (a,b,c) to cartesian. bary shape (n,3) with a+b+c=1."""
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * H
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """Convert cartesian points to barycentric coords."""
    x = pts[:, 0]
    y = pts[:, 1]
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])

def _normalize_bary(bary):
    """Ensure barycentric coords are non-negative and sum to 1."""
    bary = np.maximum(bary, 1e-15)
    bary = bary / bary.sum(axis=1, keepdims=True)
    return bary

def _all_areas_cart(pts):
    """Vectorized computation of all 165 triangle areas from cartesian points."""
    pi = pts[_tri_indices[:, 0]]
    pj = pts[_tri_indices[:, 1]]
    pk = pts[_tri_indices[:, 2]]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area_cart(pts):
    """Exact minimum area from cartesian points."""
    return float(np.min(_all_areas_cart(pts)))

def _min_area_bary(bary):
    """Compute min area from barycentric coordinates."""
    cart = _bary_to_cart(_normalize_bary(bary))
    return _min_area_cart(cart)

def _objective_de(flat_bary):
    """
    Objective for Differential Evolution.
    We work in 2D barycentric coordinates (b, c) where a = 1 - b - c.
    Domain: b >= 0, c >= 0, b + c <= 1.
    We use the full 3D vector for DE bounds [0,1]x[0,1]x[0,1] but normalize inside.
    To help DE, we penalize points that are too close to each other or outside valid range slightly?
    No, normalization handles it. But DE needs a smooth landscape.
    Using -min_area is non-smooth.
    Using -soft_min is smoother.
    """
    bary = flat_bary.reshape(-1, 3)
    # Normalize to ensure valid barycentric coords
    bary = _normalize_bary(bary)
    
    cart = _bary_to_cart(bary)
    areas = _all_areas_cart(cart)
    
    # Soft min with high beta to approximate hard min but remain smooth enough for DE
    # DE doesn't need gradients, but smoothness helps it explore.
    m = np.min(areas)
    beta = 100.0
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = m - log_sum / beta
    
    return -soft_min

def _objective_nm(flat_bary):
    """Objective for Nelder-Mead (hard min)."""
    bary = flat_bary.reshape(-1, 3)
    bary = _normalize_bary(bary)
    return -_min_area_bary(bary)

def _polish(bary_init, time_limit):
    """
    Perform local refinement using Nelder-Mead on the hard objective.
    """
    start = time.time()
    try:
        res = minimize(
            _objective_nm,
            bary_init.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        opt_bary = _normalize_bary(res.x.reshape(-1, 3))
        score = _min_area_bary(opt_bary)
        
        if time.time() - start > time_limit:
             # If it took too long, we might have missed the global best in this call, 
             # but NM is usually fast enough if converged.
             pass
             
        return opt_bary, score
    except Exception:
        return bary_init, _min_area_bary(bary_init)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_bary = None
    best_score = -1.0
    
    # 1. Global Search with Differential Evolution
    # Bounds for barycentric coordinates (a, b, c) are [0, 1] each.
    # We normalize them, so this is a valid search space covering the triangle.
    bounds = [(0.0, 1.0)] * (N * 3)
    
    try:
        de_result = differential_evolution(
            _objective_de,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            polish=False, # We do our own polish
            init='sobol',
            updating='immediate',
            workers=-1
        )
        
        if de_result.success or de_result.x is not None:
            best_bary = _normalize_bary(de_result.x.reshape(-1, 3))
            best_score = _min_area_bary(best_bary)
    except Exception:
        # Fallback to structured init if DE fails
        bary_pts = []
        bary_pts.append(np.array([1.0, 0.0, 0.0]))
        bary_pts.append(np.array([0.0, 1.0, 0.0]))
        bary_pts.append(np.array([0.0, 0.0, 1.0]))
        bary_pts.append(np.array([0.5, 0.5, 0.0]))
        bary_pts.append(np.array([0.0, 0.5, 0.5]))
        bary_pts.append(np.array([0.5, 0.0, 0.5]))
        bary_pts.append(np.array([1/3, 1/3, 1/3]))
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            r = 0.15
            da = r * np.cos(angle)
            db = r * np.sin(angle)
            dc = -da - db
            pt = np.array([1/3 + da, 1/3 + db, 1/3 + dc])
            pt = _normalize_bary(pt.reshape(1, 3))[0]
            bary_pts.append(pt)
        best_bary = np.array(bary_pts)
        best_score = _min_area_bary(best_bary)
    
    elapsed = time.time() - start_time
    if elapsed < deadline:
        # 2. Local Polish on the best result from DE
        polished_bary, polished_score = _polish(best_bary, deadline - elapsed)
        if polished_score > best_score:
            best_bary = polished_bary
            best_score = polished_score
    
    # 3. Try a few random restarts with NM if time permits
    # This helps if DE got stuck in a bad basin
    rng = np.random.default_rng(123)
    attempts = 0
    while time.time() < deadline - 10.0:
        # Random initialization
        r1 = rng.random(N)
        r2 = rng.random(N)
        mask = r1 + r2 > 1.0
        r1[mask] = 1.0 - r1[mask]
        r2[mask] = 1.0 - r2[mask]
        a = 1.0 - r1 - r2
        bary_init = np.column_stack([a, r1, r2])
        bary_init = _normalize_bary(bary_init)
        
        # Quick NM polish (limited iterations to save time)
        try:
            res = minimize(
                _objective_nm,
                bary_init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            opt_bary = _normalize_bary(res.x.reshape(-1, 3))
            score = _min_area_bary(opt_bary)
            
            if score > best_score:
                best_score = score
                best_bary = opt_bary.copy()
                # If we improved, we might want to do a longer polish on this new best
                # But we must be careful with time.
        except Exception:
            pass
        
        attempts += 1
        if attempts > 50: # Cap on random restarts to avoid infinite loop if NM is slow
            break

    best_cart = _bary_to_cart(_normalize_bary(best_bary))
    return best_cart

if __name__ == "__main__":
    pts = construct()
    print(f"Best score: {_min_area_cart(pts)}")
