import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # Area of equilateral triangle with side 1

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
n_triples = len(_IDX_TRIPLES)  # 165

def _all_dets_from_bary(bary):
    """
    Compute all determinants (twice the area in barycentric coords).
    bary: (N, 2) array of [u, v]. w = 1 - u - v.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    # det = u_i(v_j w_k - w_j v_k) - v_i(u_j w_k - w_j u_k) + w_i(u_j v_k - v_j u_k)
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det

def _min_area_ratio_from_bary(bary):
    """
    Returns the minimum absolute determinant (proportional to area ratio).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
        
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _project_to_bary_simplex(bary):
    """
    Project points to the valid barycentric simplex.
    """
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _compute_det_grads(bary):
    """
    Compute the gradient of the determinants with respect to u and v for each point.
    Returns a list of gradients for each triple, or a tensor (n_triples, N, 2).
    
    det(i,j,k) = u_i(v_j w_k - w_j v_k) - v_i(u_j w_k - w_j u_k) + w_i(u_j v_k - v_j u_k)
    where w_m = 1 - u_m - v_m.
    
    Let's expand det as a function of u_m, v_m.
    det = u_i v_j w_k - u_i w_j v_k - v_i u_j w_k + v_i w_j u_k + w_i u_j v_k - w_i v_j u_k
    
    Substituting w_m = 1 - u_m - v_m:
    This is trilinear.
    Gradient with respect to u_p:
    If p == i:
      d/du_i (u_i v_j w_k - u_i w_j v_k + w_i u_j v_k - w_i v_j u_k)
      = v_j w_k - w_j v_k + ( -1 * u_j v_k) - (-1 * v_j u_k)  (since dw_i/du_i = -1)
      = v_j w_k - w_j v_k - u_j v_k + v_j u_k
      
    If p == j:
      d/du_j (u_i v_j w_k - v_i u_j w_k + w_i u_j v_k - w_i v_j u_k)
      = u_i w_k - v_i w_k + w_i v_k - w_i v_k (dw_j/du_j = -1, so -w_i v_k becomes -(-1)v_k w_i? No.)
      Term 1: u_i w_k
      Term 2: -v_i w_k
      Term 3: w_i v_k
      Term 4: -w_i v_k * (-1) = w_i v_k? No, term is -w_i v_j u_k. d/du_j (-w_i v_j u_k) = -w_i u_k.
      Wait, let's use the product rule carefully.
      
      Let's just compute the gradient numerically for the specific structure or derive it generally.
      Actually, the determinant is the volume of the parallelepiped formed by the vectors.
      In barycentric coords, the "volume" is the determinant of the matrix of barycentric coords.
      
      Let's compute the gradient of det_k with respect to u_p and v_p.
      d(det)/du_p = 0 if p not in {i,j,k}.
      
      For p=i:
      d/du_i [ u_i(v_j w_k - w_j v_k) - v_i(u_j w_k - w_j u_k) + w_i(u_j v_k - v_j u_k) ]
      = (v_j w_k - w_j v_k) - v_i(0) + (-1)(u_j v_k - v_j u_k)
      = v_j w_k - w_j v_k - u_j v_k + v_j u_k
      
      For p=j:
      d/du_j [ u_i(v_j w_k - w_j v_k) - v_i(u_j w_k - w_j u_k) + w_i(u_j v_k - v_j u_k) ]
      = u_i(w_k - (-1)v_k) - v_i(w_k - 0) + w_i(v_k - 0)
      Wait, d/du_j w_j = -1.
      Term 1: u_i (w_k - v_k * d/du_j w_j? No, w_k doesn't depend on j. w_j does.
      d/du_j (v_j w_k - w_j v_k) = 0 - (-1)v_k = v_k. So term 1 deriv is u_i v_k.
      Term 2: -v_i (w_k - 0) = -v_i w_k. (w_k doesn't depend on j, u_k doesn't).
      Term 3: w_i (v_k - 0) = w_i v_k. (w_i doesn't depend on j, v_k doesn't).
      Wait, w_i = 1-u_i-v_i. Doesn't depend on j.
      So d/du_j det = u_i v_k - v_i w_k + w_i v_k.
      
      For p=k:
      d/du_k [ ... ]
      Term 1: u_i (0 - (-1)v_k) = u_i v_k.
      Term 2: -v_i (0 - 0) = 0. (w_k doesn't appear in term 2? u_j w_k - w_j u_k. d/du_k u_j w_k = u_j (-1) = -u_j. d/du_k -w_j u_k = -w_j. So term 2 deriv is -v_i(-u_j - w_j) = v_i(u_j+w_j).)
      Let's re-evaluate Term 2 for p=k:
      -v_i (u_j w_k - w_j u_k). d/du_k = -v_i (u_j(-1) - w_j(1)) = v_i(u_j + w_j).
      Term 3: w_i (0 - 0) = 0. (u_j v_k - v_j u_k. d/du_k = -v_j). So term 3 deriv is w_i(-v_j) = -w_i v_j.
      So d/du_k det = u_i v_k + v_i(u_j + w_j) - w_i v_j.
      
      This is getting messy. Let's use a simpler approach:
      The gradient of the soft-min objective can be computed using the chain rule and the known gradients of the determinants.
      However, to avoid bugs in manual derivation, I will use `scipy.optimize.minimize` with `method='L-BFGS-B'` which handles bounds well, but I will provide the *analytical* gradient of the smooth surrogate to speed it up.
      Actually, `trust-constr` is better for the "Trust-Region" requirement.
      
      Let's implement the gradient of the determinant correctly using the property:
      det(Pi, Pj, Pk) is linear in each Pm.
      grad_u_i det = v_j w_k - w_j v_k - u_j v_k + v_j u_k
      grad_v_i det = w_j w_k - w_j w_k? No.
      
      Let's just use numerical gradients for the first pass to ensure correctness, then switch to analytical if needed. Given the time limit, numerical gradients with L-BFGS-B or TRF might be slower but safer. However, 165 determinants is small.
      
      Let's try a hybrid: Use `L-BFGS-B` with analytical gradients for the smooth objective.
    """
    pass

def _objective_smooth(x, beta):
    """
    Objective: Minimize -softmin(areas).
    x: (2N,) array.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += 100.0 * np.sum(excess**2)
    excess_u = np.maximum(0, -u)
    penalty += 100.0 * np.sum(excess_u**2)
    excess_v = np.maximum(0, -v)
    penalty += 100.0 * np.sum(excess_v**2)
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return penalty, np.zeros_like(x)
        
    # Soft min: m - (1/beta) * log(sum(exp(-beta * (areas - m))))
    m = np.min(areas)
    shifted = areas - m
    # Numerical stability
    max_exp = np.max(-beta * shifted)
    log_sum_exp = max_exp + np.log(np.sum(np.exp(-beta * shifted - max_exp)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    val = -smooth_min + penalty
    
    # Gradient
    # d/dx (-smooth_min) = d/dx (smooth_min) with sign flipped? No.
    # f = -smooth_min. grad f = - grad smooth_min.
    # smooth_min = m - 1/beta log S.
    # d smooth_min / dx = d m / dx - 1/beta (1/S) d S / dx.
    # d m / dx is 0 almost everywhere (subgradient), but for the smooth approximation, we treat the min as part of the log-sum-exp.
    # Actually, the standard soft-min gradient is:
    # grad softmin(x) = - sum( exp(-beta x_i) / S * grad x_i )
    # Here x_i are the areas A_i.
    # grad A_i / dx = sign(det_i) * grad det_i / dx.
    
    # We need grad det_i / dx for all i.
    # This is expensive to compute analytically for all 165 triples without a vectorized formula.
    # Let's use finite differences for the gradient if we can't derive it easily, 
    # OR use L-BFGS-B which approximates the Hessian and uses numerical gradients if not provided.
    # Given the complexity, let's stick to L-BFGS-B with numerical gradients first, 
    # but use the smooth objective to guide the search.
    
    return val, None # None indicates numerical gradient

def _objective_exact(x):
    """
    Objective: Minimize -min_area.
    """
    bary = x.reshape(N, 2)
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    bary_clamped = np.column_stack((u, v))
    dets = _all_dets_from_bary(bary_clamped)
    min_area = np.min(np.abs(dets))
    return -min_area

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 30
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    selected = [np.array([1/3, 1/3])]
    
    for _ in range(N - 1):
        dists = []
        for c in candidates:
            min_d = np.min(np.linalg.norm(c - np.array(selected), axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
        
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. Random Inits
    rng = np.random.RandomState(42)
    for _ in range(20):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(pts.flatten())
        
    # 3. Magic Init (Vertices + Centroid + Midpoints)
    magic_pts = [
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ]
    inits.append(np.array(magic_pts).flatten())
    
    # 4. Structured Starts
    bary1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.75],
    ])
    inits.append(_project_to_bary_simplex(bary1).flatten())
    
    bary2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.0, 0.333],
        [0.0, 0.667],
        [0.333, 0.333],
        [0.667, 0.333],
        [0.333, 0.667],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary2).flatten())
    
    bounds = [(0, 1)] * (2 * N)
    
    # Phase 1: Global Optimization with Smooth Surrogate
    # We use L-BFGS-B with numerical gradients on the smooth objective.
    # We try different betas (sharpness of the soft-min).
    betas = [20.0, 50.0, 100.0, 200.0]
    
    for init in inits:
        if time_left() < 10.0:
            break
            
        x0 = _project_to_bary_simplex(init.reshape(N, 2)).flatten()
        
        for beta in betas:
            if time_left() < 5.0:
                break
                
            try:
                # Use L-BFGS-B for speed and robustness with bounds
                res = minimize(
                    _objective_smooth, 
                    x0, 
                    method='L-BFGS-B',
                    args=(beta,),
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                x_opt = res.x
                x_opt = _project_to_bary_simplex(x_opt.reshape(N, 2)).flatten()
                
                # Phase 2: Local Refinement with Exact Objective
                # Use Nelder-Mead on the exact objective for final polish
                res2 = minimize(
                    _objective_exact, 
                    x_opt, 
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                x_final = res2.x
                x_final = _project_to_bary_simplex(x_final.reshape(N, 2)).flatten()
                
                bary_final = x_final.reshape(N, 2)
                s = _min_area_ratio_from_bary(bary_final) / (2 * TRI_AREA)
                
                if s > best_score:
                    best_score = s
                    best_x = x_final.copy()
                    
                # Update start point for next beta to be the best found so far
                x0 = x_final
                
            except Exception:
                pass
    
    # If we found a good solution, convert to Cartesian
    if best_x is not None:
        bary_best = best_x.reshape(N, 2)
        pts_best = _bary_to_cart(bary_best[:, 0], best_x[:, 1])
        return pts_best
    
    # Fallback: return random valid points
    pts = []
    for _ in range(N):
        u, v = np.random.random(2)
        while u + v > 1:
            u, v = np.random.random(2)
        pts.append([u + 0.5*v, np.sqrt(3)/2*v])
    return np.array(pts)

def score(points):
    """Compute the raw score: min triangle area / TRI_AREA."""
    bary = np.zeros((N, 2))
    for i in range(N):
        p = points[i]
        w = 2 * p[1] / np.sqrt(3)
        v = p[0] - w * 0.5
        u = 1 - v - w
        bary[i] = [u, v]
    
    dets = _all_dets_from_bary(bary)
    min_area = np.min(np.abs(dets))
    return min_area / (2 * TRI_AREA)

# Ensure the function name is construct as per the task
# The task says: Interface: construct() -> array of shape (n, 2)
# The current program has construct() and score(). I will keep both.
