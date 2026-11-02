"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 140  # safety margin from 150s limit

# Precompute all triplets for N=11 points (indices into the 11 points)
_TRIPLETS = list(itertools.combinations(range(N), 3))
_N_TRIPLETS = len(_TRIPLETS)  # 165

# Precompute index arrays for vectorized area computation
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

def _bary_to_cart(bary_pts):
    """Convert barycentric coordinates (u, v) to Cartesian (x, y) for triangle (0,0), (1,0), (0.5, S3H).
    bary_pts: array of shape (N, 2) where each row is (u, v). w = 1 - u - v.
    x = u * 1 + v * 0.5
    y = v * S3H
    """
    u = bary_pts[:, 0]
    v = bary_pts[:, 1]
    x = u + 0.5 * v
    y = v * S3H
    return np.column_stack((x, y))

def _cart_to_bary(cart_pts):
    """Convert Cartesian (x, y) to barycentric (u, v).
    x = u + 0.5 v
    y = v S3H  => v = y / S3H
    u = x - 0.5 v
    """
    x = cart_pts[:, 0]
    y = cart_pts[:, 1]
    v = y / S3H
    u = x - 0.5 * v
    return np.column_stack((u, v))

def _vectorized_min_area_bary(bary_pts):
    """Compute min area of triangles formed by points given in barycentric coords.
    Area = TRI_AREA * |det([[u_i, v_i, w_i], [u_j, v_j, w_j], [u_k, v_k, w_k]])|
    """
    pi = bary_pts[_T_I]
    pj = bary_pts[_T_J]
    pk = bary_pts[_T_K]
    
    # w = 1 - u - v
    wi = 1 - pi[:, 0] - pi[:, 1]
    wj = 1 - pj[:, 0] - pj[:, 1]
    wk = 1 - pk[:, 0] - pk[:, 1]
    
    # Determinant of 3x3 matrix
    # | ui vi wi |
    # | uj vj wj |
    # | uk vk wk |
    det = (pi[:, 0] * (pj[:, 1] * wk - wj * pk[:, 1]) -
           pi[:, 1] * (pj[:, 0] * wk - wj * pk[:, 0]) +
           wi     * (pj[:, 0] * pk[:, 1] - pj[:, 1] * pk[:, 0]))
    
    areas = np.abs(det) * TRI_AREA
    return np.min(areas)

def _objective_bary(flat_params):
    """Objective for optimization: minimize negative min area.
    Input is flat barycentric coordinates (u1, v1, u2, v2, ...).
    We assume points are in the simplex. DE will handle bounds.
    """
    bary_pts = flat_params.reshape(-1, 2)
    # Ensure points are in simplex by projecting if necessary?
    # DE keeps them in bounds (0,1) for u and v, but u+v can be > 1.
    # We must project to simplex: if u+v > 1, scale or reflect.
    # Standard reflection: if u+v > 1, swap u and 1-u-v? No, just clip?
    # Better: use a valid mapping.
    # Let's just compute area. If points are outside, area might be weird.
    # But DE bounds are (0,1) for u and (0,1) for v.
    # We need to enforce u+v <= 1.
    # Simple projection: if u+v > 1, set u = u/(u+v), v = v/(u+v)? No, that changes position.
    # Reflection across the line u+v=1:
    # u' = 1 - v, v' = 1 - u? No.
    # Let's use the standard rejection or projection.
    # For speed, we'll assume DE finds good points and we polish with NM which respects bounds?
    # NM does not respect bounds well.
    # Let's add a penalty for u+v > 1.
    penalty = 0.0
    sums = bary_pts[:, 0] + bary_pts[:, 1]
    viol = np.maximum(0, sums - 1)
    penalty = np.sum(viol * 100)
    
    ma = _vectorized_min_area_bary(bary_pts)
    return -ma + penalty

def _project_bary_to_simplex(bary_pts):
    """Project barycentric points to the simplex u>=0, v>=0, u+v<=1."""
    proj = bary_pts.copy()
    for i in range(len(proj)):
        u, v = proj[i]
        if u < 0: u = 0
        if v < 0: v = 0
        if u + v > 1:
            # Scale down to boundary
            s = u + v
            u = u / s
            v = v / s
        proj[i] = [u, v]
    return proj

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_bary = None
    best_area = 0.0
    
    # Initial random configuration in barycentric coords
    # Sample u, v such that u+v <= 1
    def random_bary():
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        return np.array([u, v])
        
    bary_pts = np.array([random_bary() for _ in range(N)])
    area = _vectorized_min_area_bary(bary_pts)
    best_bary = bary_pts.copy()
    best_area = area
    
    # Phase 1: Differential Evolution in Barycentric Space
    # Bounds: u in [0, 1], v in [0, 1]. Constraint u+v <= 1 handled by penalty.
    elapsed = time.time() - start_time
    if elapsed < 60:
        try:
            bounds = [(0, 1), (0, 1)] * N
            result = differential_evolution(
                _objective_bary,
                bounds,
                maxiter=50,
                popsize=20,
                seed=42,
                tol=1e-6,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.5),
                recombination=0.7
            )
            opt_bary = result.x.reshape(-1, 2)
            opt_bary = _project_bary_to_simplex(opt_bary)
            area = _vectorized_min_area_bary(opt_bary)
            if area > best_area:
                best_area = area
                best_bary = opt_bary.copy()
        except Exception:
            pass
    
    # Phase 2: Local Refinement with Nelder-Mead
    # We will run NM from multiple starts (best + random perturbations)
    elapsed = time.time() - start_time
    n_starts = 20
    for s in range(n_starts):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Start from best or random
        if s == 0:
            start_pts = best_bary.copy()
        elif s % 2 == 0 and best_bary is not None:
            # Perturb best
            start_pts = best_bary + rng.normal(0, 0.05, size=(N, 2))
            start_pts = _project_bary_to_simplex(start_pts)
        else:
            # Random start
            start_pts = np.array([random_bary() for _ in range(N)])
        
        try:
            # Nelder-Mead on the barycentric objective
            # We need to handle the simplex constraint. NM might step outside.
            # The penalty in _objective_bary helps.
            result = minimize(
                _objective_bary,
                start_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            opt_bary = result.x.reshape(-1, 2)
            opt_bary = _project_bary_to_simplex(opt_bary)
            
            area = _vectorized_min_area_bary(opt_bary)
            if area > best_area:
                best_area = area
                best_bary = opt_bary.copy()
                
        except Exception:
            continue

    # Phase 3: Polish best with high precision
    elapsed = time.time() - start_time
    if best_bary is not None and elapsed < DEADLINE - 5:
        try:
            result = minimize(
                _objective_bary,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            opt_bary = result.x.reshape(-1, 2)
            opt_bary = _project_bary_to_simplex(opt_bary)
            area = _vectorized_min_area_bary(opt_bary)
            if area > best_area:
                best_area = area
                best_bary = opt_bary.copy()
        except Exception:
            pass

    # Convert best barycentric to Cartesian
    if best_bary is None:
        # Fallback
        best_bary = np.array([random_bary() for _ in range(N)])
        
    best_cart = _bary_to_cart(best_bary)
    
    return best_cart
# EVOLVE-BLOCK-END
