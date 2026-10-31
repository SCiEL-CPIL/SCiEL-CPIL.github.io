import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute indices for combinations of 3 from 11
_tri_indices = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _clamp_bary(bary):
    """
    Clamp 2D barycentric coordinates (u, v) to the valid triangle region.
    u >= 0, v >= 0, u + v <= 1.
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip to non-negative
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    # Check u + v <= 1
    sums = u + v
    invalid = sums > 1.0
    if np.any(invalid):
        # Scale down to boundary
        scale = 1.0 / sums[invalid]
        u[invalid] *= scale
        v[invalid] *= scale
        
    return np.column_stack([u, v])

def _calc_dets(bary):
    """
    Calculate absolute determinants of barycentric coordinates for all triangles.
    bary: (N, 2) array of (u, v).
    Returns: Array of shape (165,) with |det|.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Extract coordinates for all triangles
    ui = u[_tri_indices[:, 0]]
    vi = v[_tri_indices[:, 0]]
    wi = w[_tri_indices[:, 0]]
    
    uj = u[_tri_indices[:, 1]]
    vj = v[_tri_indices[:, 1]]
    wj = w[_tri_indices[:, 1]]
    
    uk = u[_tri_indices[:, 2]]
    vk = v[_tri_indices[:, 2]]
    wk = w[_tri_indices[:, 2]]
    
    # Determinant of [[ui, vi, wi], [uj, vj, wj], [uk, vk, wk]]
    # = ui(vj*wk - vk*wj) - vi(uj*wk - uk*wj) + wi(uj*vk - uk*vj)
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def _hard_obj(flat_bary):
    """
    Objective for hard minimization: -min(area_ratio).
    """
    bary = flat_bary.reshape(N, 2)
    bary = _clamp_bary(bary)
    dets = _calc_dets(bary)
    return -np.min(dets)

def _soft_obj(flat_bary, beta=100.0):
    """
    Objective for soft minimization: -soft_min(area_ratio).
    """
    bary = flat_bary.reshape(N, 2)
    bary = _clamp_bary(bary)
    dets = _calc_dets(bary)
    
    # Soft min: -1/beta * log(sum(exp(-beta * dets)))
    # To avoid overflow, subtract min
    m = np.min(dets)
    shifted = -beta * (dets - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = m - log_sum / beta
    
    return -soft_min

def _bary_to_cart(bary):
    """
    Convert 2D barycentric (u, v) to Cartesian.
    V0=(0,0), V1=(1,0), V2=(0.5, H)
    x = v + 0.5*w = v + 0.5(1-u-v) = 0.5 - 0.5u + 0.5v
    y = w*H = (1-u-v)H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = 0.5 - 0.5 * u + 0.5 * v
    y = w * H
    return np.column_stack([x, y])

def _init_structured():
    """
    Generate structured initial points in 2D barycentric coords.
    """
    pts = []
    # Vertices
    pts.append([1.0, 0.0]) # u=1, v=0 -> V0
    pts.append([0.0, 1.0]) # u=0, v=1 -> V1
    pts.append([0.0, 0.0]) # u=0, v=0 -> V2
    
    # Edge midpoints
    pts.append([0.5, 0.5]) # Mid V0-V1
    pts.append([0.5, 0.0]) # Mid V0-V2
    pts.append([0.0, 0.5]) # Mid V1-V2
    
    # Centroid
    pts.append([1/3, 1/3])
    
    # 4 more points around centroid
    cx, cy = 1/3, 1/3
    r = 0.15
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        du = r * np.cos(angle)
        dv = r * np.sin(angle)
        pt = [cx + du, cy + dv]
        # Clamp if necessary, though small r should keep it inside
        pt_clamped = _clamp_bary(np.array([pt]))[0]
        pts.append(pt_clamped)
        
    return np.array(pts)

def _init_random(rng):
    """
    Generate random points in 2D barycentric coords.
    """
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def _init_hexagonal():
    """
    Generate hexagonal lattice points.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try different scales
    for s in np.linspace(0.1, 0.4, 10):
        points = []
        for i in range(6):
            for j in range(6):
                u = i * s
                v = j * s * 0.866 # approx sqrt(3)/2
                if u + v <= 1 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            val = -_hard_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
                
    if best_bary is None:
        rng = np.random.default_rng(42)
        best_bary = _init_random(rng)
        
    return best_bary

def _optimize_local(bary_init, time_left_fn, rng):
    """
    Refine a candidate using L-BFGS-B (soft) then Nelder-Mead (hard).
    """
    best_bary = bary_init.copy()
    best_score = -_hard_obj(best_bary.flatten())
    
    # Stage 1: L-BFGS-B with soft objective
    if time_left_fn() > 10.0:
        try:
            # Use a few betas
            for beta in [50.0, 200.0]:
                if time_left_fn() < 5.0: break
                res = minimize(
                    _soft_obj,
                    best_bary.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * (N * 2),
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                opt_bary = _clamp_bary(res.x.reshape(N, 2))
                score = -_hard_obj(opt_bary.flatten())
                if score > best_score:
                    best_score = score
                    best_bary = opt_bary.copy()
        except Exception:
            pass
            
    # Stage 2: Nelder-Mead with hard objective
    if time_left_fn() > 5.0:
        try:
            res = minimize(
                _hard_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            nm_bary = _clamp_bary(res.x.reshape(N, 2))
            nm_score = -_hard_obj(nm_bary.flatten())
            if nm_score > best_score:
                best_score = nm_score
                best_bary = nm_bary.copy()
        except Exception:

            pass

    return best_bary, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
        
    rng = np.random.default_rng(12345)
    
    candidates = []
    
    # 1. Structured Init
    struct_bary = _init_structured()
    struct_score = -_hard_obj(struct_bary.flatten())
    candidates.append((struct_bary, struct_score))
    
    # 2. Hexagonal Init
    hex_bary = _init_hexagonal()
    hex_score = -_hard_obj(hex_bary.flatten())
    candidates.append((hex_bary, hex_score))
    
    # 3. Random Inits
    for _ in range(5):
        rand_bary = _init_random(rng)
        rand_score = -_hard_obj(rand_bary.flatten())
        candidates.append((rand_bary, rand_score))
        
    # 4. Differential Evolution (Global Search)
    if time_left() > 60.0:
        try:
            # DE works on box bounds. We map [0,1]^2 to triangle using u=x, v=y(1-x)
            # But our objective takes u,v directly.
            # So we define a wrapper that takes DE vars (x,y) in [0,1], maps to u,v, and calls objective.
            
            def de_obj(flat_xy):
                xy = flat_xy.reshape(N, 2)
                x = xy[:, 0]
                y = xy[:, 1]
                u = x
                v = y * (1 - x)
                bary = np.column_stack([u, v])
                # No need to clamp as mapping is inside triangle
                dets = _calc_dets(bary)
                return -np.min(dets)
                
            bounds = [(0, 1), (0, 1)] * N
            de_res = differential_evolution(
                de_obj, 
                bounds, 
                maxiter=30, 
                popsize=20, 
                tol=1e-5, 
                seed=42, 
                polish=False
            )
            
            if de_res.x is not None:
                xy = de_res.x.reshape(N, 2)
                x = xy[:, 0]
                y = xy[:, 1]
                u = x
                v = y * (1 - x)
                de_bary = np.column_stack([u, v])
                de_score = -_hard_obj(de_bary.flatten())
                candidates.append((de_bary, de_score))
                
        except Exception:
            pass

    # Sort candidates by score (descending)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_bary = candidates[0][0]
    best_score = candidates[0][1]
    
    # Refine top candidates
    # We have ~40-60s left if DE ran, or ~140s if not.
    # Let's refine top 3.
    for bary_init, score_init in candidates[:3]:
        if time_left() < 15.0:
            break
        try:
            opt_bary, opt_score = _optimize_local(bary_init, time_left, rng)
            if opt_score > best_score:
                best_score = opt_score
                best_bary = opt_bary.copy()
        except Exception:
            pass
            
    # Final Polish on the very best
    if time_left() > 5.0:
        try:
            res = minimize(
                _hard_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            final_bary = _clamp_bary(res.x.reshape(N, 2))
            final_score = -_hard_obj(final_bary.flatten())
            if final_score > best_score:
                best_score = final_score
                best_bary = final_bary.copy()
        except Exception:
            pass

    # Convert to Cartesian
    best_cart = _bary_to_cart(best_bary)
    
    return best_cart

if __name__ == "__main__":
    pts = construct()
    # Calculate score for debugging
    bary = np.zeros((11, 2))
    # Re-derive bary from cart for check? No, just trust the internal logic.
    # The score returned by evaluator is min_area / total_area.
    # Our internal score is min_det.
    # They should be identical.
    print(f"Internal Score: {-_hard_obj(_bary_to_cart(np.array([[1/3, 1/3]]*11)).flatten()) if False else 'N/A'}")
    # Actually, let's just print the min area of the result to be sure.
    # But construct() returns cart.
    # We can compute score from cart if we had the function, but we don't need to.
    pass
