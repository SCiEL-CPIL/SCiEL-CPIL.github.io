"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

# Vertices of the triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_N_TRIPLETS = len(_TRIPLETS)  # 165

# Precompute index arrays for vectorized area computation
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Vertices as a matrix for matrix multiplication
# bary (n, 3) -> cart (n, 2)
# cart = bary[:, 1:3] @ V12 where V12 is (2,2) matrix of V1, V2?
# p = a*V0 + b*V1 + c*V2. Since V0=0, p = b*V1 + c*V2.
# V1 = [1, 0], V2 = [0.5, S3H]
# p_x = b * 1 + c * 0.5
# p_y = b * 0 + c * S3H
# So cart = bary[:, 1:3] @ [[1, 0], [0.5, S3H]]
_V_MATRIX = np.array([[1.0, 0.0],
                      [0.5, S3H]])

DEADLINE = 140  # safety margin from 150s limit


def _bary_to_cart(bary_pts):
    """Convert barycentric coordinates (n, 3) to Cartesian (n, 2)."""
    # bary_pts: (n, 3), columns are a, b, c
    # We use b and c (indices 1 and 2)
    return bary_pts[:, 1:3] @ _V_MATRIX


def _cart_to_bary(cart_pts):
    """Convert Cartesian (n, 2) to barycentric coordinates (n, 3)."""
    # p = b*V1 + c*V2
    # x = b + 0.5c
    # y = S3H * c  => c = y / S3H
    # b = x - 0.5c
    # a = 1 - b - c
    c = cart_pts[:, 1] / S3H
    b = cart_pts[:, 0] - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack((a, b, c))


def _vectorized_min_area_cart(pts):
    """Vectorized computation of min triangle area using precomputed triplets."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _soft_min_area_cart(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _objective_soft_min(flat_bary, beta):
    """
    Objective function for L-BFGS-B.
    Input: flat array of 2*N barycentric coordinates (a, b for each point, c is implicit 1-a-b).
    We optimize a and b. c = 1 - a - b.
    Constraints: a>=0, b>=0, a+b<=1.
    We handle constraints via bounds in L-BFGS-B.
    """
    # Reshape to (N, 2)
    ab = flat_bary.reshape(N, 2)
    a = ab[:, 0]
    b = ab[:, 1]
    c = 1.0 - a - b
    
    # Construct full barycentric matrix
    bary = np.column_stack((a, b, c))
    
    # Convert to Cartesian
    pts = _bary_to_cart(bary)
    
    # Compute soft min area
    area = _soft_min_area_cart(pts, beta)
    
    # We want to maximize area, so minimize negative area
    return -area


def _objective_hard_min(flat_bary):
    """
    Objective function for Nelder-Mead (polish).
    Minimizes negative hard min area.
    """
    ab = flat_bary.reshape(N, 2)
    a = ab[:, 0]
    b = ab[:, 1]
    c = 1.0 - a - b
    bary = np.column_stack((a, b, c))
    pts = _bary_to_cart(bary)
    area = _vectorized_min_area_cart(pts)
    return -area


def _project_to_simplex(ab):
    """
    Project (a, b) pairs back to the valid simplex if they drift outside due to numerical errors.
    a>=0, b>=0, a+b<=1.
    """
    for i in range(N):
        a, b = ab[i]
        # Simple clipping is not enough for a+b<=1, but for L-BFGS-B with bounds it shouldn't happen.
        # For Nelder-Mead, it might.
        if a < 0: a = 0
        if b < 0: b = 0
        if a + b > 1:
            # Scale down
            s = 1.0 / (a + b)
            a *= s
            b *= s
        ab[i] = [a, b]
    return ab


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_ab = None
    best_area = 0.0
    
    # Bounds for L-BFGS-B: a in [0,1], b in [0,1]
    # Note: The constraint a+b<=1 is not directly handled by bounds, 
    # but the soft-min penalty or the nature of the problem usually keeps them valid.
    # However, if a+b > 1, c becomes negative, point is outside.
    # To be safe, we can add a penalty if c < 0 in the objective, 
    # or rely on the fact that the optimum is inside.
    # Let's add a small penalty for invalid barycentric coords in the soft-min objective.
    
    def _objective_soft_min_penalized(flat_bary, beta):
        ab = flat_bary.reshape(N, 2)
        a = ab[:, 0]
        b = ab[:, 1]
        c = 1.0 - a - b
        
        # Penalty for being outside the triangle (c < 0 or a < 0 or b < 0)
        # a, b are bounded by L-BFGS-B, but c can be negative.
        penalty = np.sum(np.maximum(0, -c) ** 2) * 100.0
        
        bary = np.column_stack((a, b, c))
        pts = _bary_to_cart(bary)
        area = _soft_min_area_cart(pts, beta)
        
        return -area + penalty

    # Phase 1: Multi-start Soft-Min Annealing
    # We will run several restarts. For each restart, we anneal beta.
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        # Random initial guess in barycentric coordinates
        # Generate random points in simplex
        # Method: generate 3 random numbers, normalize
        bary_init = rng.random((N, 3))
        bary_init /= bary_init.sum(axis=1, keepdims=True)
        
        # Extract a, b for optimization
        ab_init = bary_init[:, :2].flatten()
        
        # Annealing schedule
        # Start with low beta (smooth), increase to high beta (sharp)
        betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
        
        current_ab = ab_init.copy()
        
        for beta in betas:
            if time.time() - start_time > DEADLINE - 10:
                break
                
            try:
                res = minimize(
                    _objective_soft_min_penalized,
                    current_ab,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(0, 1), (0, 1)] * N,  # a in [0,1], b in [0,1]
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                current_ab = res.x
            except Exception:
                pass
        
        # Evaluate the result of this restart
        # Project to ensure validity (c >= 0)
        current_ab = _project_to_simplex(current_ab)
        
        a = current_ab[0::2]
        b = current_ab[1::2]
        c = 1.0 - a - b
        bary = np.column_stack((a, b, c))
        pts = _bary_to_cart(bary)
        
        # Check if all points are strictly inside or on boundary
        # c >= -eps
        if np.min(c) >= -1e-9:
            area = _vectorized_min_area_cart(pts)
            if area > best_area:
                best_area = area
                best_ab = current_ab.copy()
    
    # Phase 2: Local Refinement with Hard Min (Nelder-Mead)
    # Use the best solution found so far as a starting point
    if best_ab is not None:
        # We might also want to try refining from random points near the best one
        n_refine_starts = 5
        for s in range(n_refine_starts):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            if s == 0:
                start_ab = best_ab.copy()
            else:
                # Perturb best
                start_ab = best_ab + rng.normal(0, 0.01, size=(2*N,))
                start_ab = _project_to_simplex(start_ab)
            
            try:
                res = minimize(
                    _objective_hard_min,
                    start_ab,
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                opt_ab = res.x
                opt_ab = _project_to_simplex(opt_ab)
                
                a = opt_ab[0::2]
                b = opt_ab[1::2]
                c = 1.0 - a - b
                if np.min(c) >= -1e-9:
                    bary = np.column_stack((a, b, c))
                    pts = _bary_to_cart(bary)
                    area = _vectorized_min_area_cart(pts)
                    if area > best_area:
                        best_area = area
                        best_ab = opt_ab.copy()
            except Exception:
                pass

    # Finalize
    if best_ab is None:
        # Fallback: return random points
        bary = rng.random((N, 3))
        bary /= bary.sum(axis=1, keepdims=True)
        return _bary_to_cart(bary)
    
    a = best_ab[0::2]
    b = best_ab[1::2]
    c = 1.0 - a - b
    bary = np.column_stack((a, b, c))
    pts = _bary_to_cart(bary)
    
    return pts
# EVOLVE-BLOCK-END
