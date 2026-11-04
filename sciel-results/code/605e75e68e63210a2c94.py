"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices in Cartesian for potential fallback or checking
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples (i, j, k) with i < j < k
# There are C(11, 3) = 165 triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    x = v*1 + w*0.5 = v + (1-u-v)*0.5 = 0.5 - 0.5*u + 0.5*v
    y = w*sqrt(3)/2 = (1-u-v)*sqrt(3)/2
    """
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = (np.sqrt(3) / 2) * w
    return np.column_stack((x, y))

def _compute_dets(bary):
    """
    Compute the determinants for all triples of points given barycentric coords.
    The area of a triangle formed by points with barycentric coords P_i, P_j, P_k 
    is proportional to the absolute value of the determinant of the matrix 
    [[u_i, v_i, w_i], [u_j, v_j, w_j], [u_k, v_k, w_k]].
    Specifically, Area_cart = |det| * Area_triangle.
    Since we want to maximize the minimum area, we can just maximize the minimum |det|.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Extract coordinates for triples
    ui = u[_IDX_TRIPLES[:, 0]]
    vi = v[_IDX_TRIPLES[:, 0]]
    wi = w[_IDX_TRIPLES[:, 0]]
    
    uj = u[_IDX_TRIPLES[:, 1]]
    vj = v[_IDX_TRIPLES[:, 1]]
    wj = w[_IDX_TRIPLES[:, 1]]
    
    uk = u[_IDX_TRIPLES[:, 2]]
    vk = v[_IDX_TRIPLES[:, 2]]
    wk = w[_IDX_TRIPLES[:, 2]]
    
    # Determinant of 3x3 matrix:
    # | ui vi wi |
    # | uj vj wj |
    # | uk vk wk |
    # = ui(vj*wk - wj*vk) - vi(uj*wk - wj*uk) + wi(uj*vk - vj*uk)
    
    det = (ui * (vj * wk - wj * vk) 
           - vi * (uj * wk - wj * uk) 
           + wi * (uj * vk - vj * uk))
           
    return np.abs(det)

def _objective_smooth(x, beta, penalty_lambda=100.0):
    """
    Objective function for L-BFGS-B.
    Minimizes the negative smooth minimum of the determinants.
    x: (2N,) array of u, v coords.
    beta: smoothing parameter for log-sum-exp.
    penalty_lambda: penalty for points outside the simplex (u+v > 1).
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Penalty for invalid barycentric coordinates (u+v > 1)
    # We want u+v <= 1. Excess = max(0, u+v - 1)
    excess = np.maximum(0.0, u + v - 1.0)
    penalty = penalty_lambda * np.sum(excess**2)
    
    # Also penalize u < 0 or v < 0, though bounds should handle it, 
    # L-BFGS-B respects bounds strictly, so u,v >= 0 is guaranteed.
    
    dets = _compute_dets(bary)
    
    # Smooth minimum: -1/beta * log(sum(exp(-beta * dets)))
    # As beta -> inf, this approaches -min(dets)
    # We want to MAXIMIZE min(dets), so we MINIMIZE -min(dets).
    # So objective = penalty - smooth_min(dets)
    # smooth_min(dets) = -1/beta * log(sum(exp(-beta * dets)))
    # So -smooth_min(dets) = 1/beta * log(sum(exp(-beta * dets)))
    
    # To avoid overflow/underflow in exp, shift by min
    min_det = np.min(dets)
    shifted_dets = -beta * (dets - min_det)
    # exp(shifted_dets) is in range [0, 1] roughly, max is 1
    log_sum_exp = min_det * beta + np.log(np.sum(np.exp(shifted_dets)))
    smooth_min_val = log_sum_exp / beta
    
    # We want to maximize smooth_min_val, so minimize -smooth_min_val + penalty
    return -smooth_min_val + penalty

def _generate_initialization(rng, n_points=N):
    """
    Generate random points inside the triangle.
    """
    points = np.zeros((n_points, 2))
    for i in range(n_points):
        # Random point in simplex
        r1 = rng.uniform(0, 1)
        r2 = rng.uniform(0, 1)
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        points[i, 0] = r1
        points[i, 1] = r2
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    def time_left():
        return deadline - time.time()

    best_x = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Strategy: Multi-start with annealing of beta
    # 1. Generate a few diverse initializations
    n_starts = 10
    initializations = []
    for _ in range(n_starts):
        if time_left() < 20.0:
            break
        init = _generate_initialization(rng)
        initializations.append(init.flatten())
        
    # If we have time, also add a "grid-like" or "FPS-like" initialization
    # FPS is good for spreading points
    if time_left() > 30.0:
        fps_points = np.zeros((N, 2))
        # Start with centroid
        fps_points[0] = [1/3, 1/3]
        for k in range(1, N):
            # Sample many candidates
            candidates = _generate_initialization(rng, 500)
            # Calculate min distance to existing points
            diffs = candidates[:, np.newaxis, :] - fps_points[np.newaxis, :k, :]
            dists = np.linalg.norm(diffs, axis=2)
            min_dists = np.min(dists, axis=1)
            best_idx = np.argmax(min_dists)
            fps_points[k] = candidates[best_idx]
        initializations.append(fps_points.flatten())

    # Process each initialization
    for x0 in initializations:
        if time_left() < 10.0:
            break
            
        # Annealing schedule for beta
        # Start with low beta (smooth) to find basin, increase to sharp
        betas = [5.0, 20.0, 100.0, 500.0, 2000.0]
        
        current_x = x0.copy()
        
        for beta in betas:
            if time_left() < 5.0:
                break
                
            try:
                res = minimize(
                    _objective_smooth,
                    current_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * (2 * N),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                current_x = res.x
                
                # Evaluate true score (min det) to track best
                bary = current_x.reshape(N, 2)
                dets = _compute_dets(bary)
                # Check validity: u+v <= 1
                if np.any(bary[:, 0] + bary[:, 1] > 1.0 + 1e-6):
                    # If invalid, clip and re-evaluate, but score is effectively lower
                    # For tracking, we use the penalized value or just skip if badly invalid
                    pass 
                
                true_score = np.min(dets)
                if true_score > best_score:
                    best_score = true_score
                    best_x = current_x.copy()
                    
            except Exception as e:
                pass

    # Final Polish with very high beta on the best solution found
    if best_x is not None and time_left() > 5.0:
        try:
            res = minimize(
                _objective_smooth,
                best_x,
                args=(10000.0,),
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            final_x = res.x
            bary = final_x.reshape(N, 2)
            dets = _compute_dets(bary)
            final_score = np.min(dets)
            
            # Verify validity
            if np.all(bary[:, 0] >= -1e-9) and np.all(bary[:, 1] >= -1e-9) and np.all(bary[:, 0] + bary[:, 1] <= 1.0 + 1e-9):
                if final_score > best_score:
                    best_score = final_score
                    best_x = final_x
        except Exception:
            pass

    if best_x is None:
        # Fallback: Regular grid or random
        best_x = _generate_initialization(rng).flatten()

    # Convert best barycentric coordinates to Cartesian
    bary_best = best_x.reshape(N, 2)
    # Clip to ensure strictly inside/valid for safety
    bary_best[:, 0] = np.clip(bary_best[:, 0], 0, 1)
    bary_best[:, 1] = np.clip(bary_best[:, 1], 0, 1)
    # Project to simplex if u+v > 1
    s = bary_best[:, 0] + bary_best[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary_best[mask, 0] /= s[mask]
        bary_best[mask, 1] /= s[mask]
        
    points = _bary_to_cart(bary_best[:, 0], bary_best[:, 1])
    
    # Final safety check: ensure points are inside the triangle
    # y >= 0
    # y <= sqrt(3)/2
    # x >= y/sqrt(3)
    # x <= 1 - y/sqrt(3)
    # The barycentric conversion guarantees this if inputs are valid.
    
    return np.array(points)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
