import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 115.0 # Safety margin

def _score(points):
    """Largest common radius: min of half pairwise distances and wall clearance."""
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu].min()
    clearance = (0.5 - np.abs(points)).min()
    return min(min_pair * 0.5, clearance)

def _soft_min_obj_and_grad(centers, T=0.01):
    """
    Objective: -soft_min(distances)
    We want to maximize the minimum distance.
    soft_min(x) = -T * log(sum(exp(-x_i / T)))
    Max soft_min(x) is approximated by min(x_i) as T -> 0.
    So we minimize -soft_min(x) = T * log(sum(exp(-x_i / T))).
    
    Here x_i are the pairwise distances (and wall clearances).
    Actually, it's easier to work with squared distances or just distances.
    Let's use distances.
    
    Let d_ij be distances.
    We want to maximize min(d_ij, wall_clearance).
    Let v be the vector of all constraints (distances and wall clearances).
    We minimize F = T * log( sum( exp(-v_i / T) ) ).
    
    Gradient of F w.r.t position p_k:
    F = T * log( S ), where S = sum_j exp(-v_j / T)
    dF/dp_k = T * (1/S) * dS/dp_k
    dS/dp_k = sum_j exp(-v_j / T) * (-1/T) * dv_j/dp_k
    dF/dp_k = - (1/S) * sum_j exp(-v_j / T) * dv_j/dp_k
    dF/dp_k = - sum_j (exp(-v_j / T) / S) * dv_j/dp_k
    Let w_j = exp(-v_j / T) / S. Note sum(w_j) = 1.
    dF/dp_k = - sum_j w_j * dv_j/dp_k
    
    v_j are:
    1. Pairwise distances d_ij = ||p_i - p_j||
       dv_ij/dp_k = 0 if k != i, j
       dv_ij/dp_i = (p_i - p_j) / d_ij
       dv_ij/dp_j = (p_j - p_i) / d_ij
       
    2. Wall clearances: c_ik = 0.5 - |p_ik|
       dv_ik/dp_k = -sign(p_ik)
       
    We need to handle d_ij = 0 carefully (set grad to 0 or random).
    """
    c = centers.reshape(-1, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf) # Ignore self
    
    # Wall clearances
    # We treat wall clearance as a "distance" to the boundary.
    # For each point i and dim k, clearance = 0.5 - |c_ik|.
    # We need to accumulate gradients from these too.
    
    # Collect all "values" v for the soft-min
    # This is memory intensive if N is large, but N=100 is small.
    # Pairs: N*(N-1)/2 = 4950
    # Walls: N*3 = 300
    # Total ~5250 values.
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Wall values
    # c_ik > 0 -> 0.5 - c_ik
    # c_ik < 0 -> 0.5 + c_ik
    abs_c = np.abs(c)
    wall_vals = 0.5 - abs_c
    # Flatten wall_vals to align with indices
    # We need to map wall indices back to (point, dim) for gradients
    
    all_vals = np.concatenate([pair_dists, wall_vals.flatten()])
    
    # Weights for soft-min
    # To avoid underflow/overflow in exp, shift by max val?
    # exp(-v/T). If v is large, exp is small. If v is small, exp is large.
    # Standard trick: m = max(v). exp(-(v-m)/T).
    # But we want weights w_j proportional to exp(-v_j/T).
    # Let's just compute log-sum-exp carefully.
    
    # v_min = min(all_vals)
    # log(S) = log(sum(exp(-v_j/T))) = -v_min/T + log(sum(exp(-(v_j-v_min)/T)))
    # This is stable.
    
    v_min = np.min(all_vals)
    shifted = (all_vals - v_min) / T
    # exp(-shifted)
    exp_terms = np.exp(-shifted)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp # w_j
    
    # Objective value: T * log(S)
    # log(S) = -v_min/T + log(sum_exp)
    obj_val = T * (-v_min / T + np.log(sum_exp))
    obj_val = -obj_val # We want to minimize -soft_min, so return negative?
    # Wait.
    # Maximize soft_min(v) = -T * log(S).
    # Minimize -soft_min(v) = T * log(S).
    # So obj_val is T * log(S). This is the value to minimize.
    
    # Gradient
    grad = np.zeros_like(c)
    
    # 1. Pairwise contribution
    # w_j for pairs are weights[:len(pair_dists)]
    pair_weights = weights[:len(pair_dists)]
    
    # We need to sum w_j * dv_ij/dp_k
    # For each pair (i, j) with weight w:
    # grad_i += -w * (p_i - p_j) / d_ij
    # grad_j += -w * (p_j - p_i) / d_ij
    
    # Vectorized update:
    # diff[i,j] = p_i - p_j
    # We have dists matrix.
    # We only have upper triangle weights.
    
    # Create a matrix of weights for pairs (N x N)
    pair_w_matrix = np.zeros((N, N))
    pair_w_matrix[iu] = pair_weights
    pair_w_matrix = pair_w_matrix + pair_w_matrix.T
    
    # Safe dists for division
    safe_dists = np.where(dists > 1e-8, dists, 1.0)
    # Factor = w_ij / d_ij
    factors = pair_w_matrix / safe_dists
    np.fill_diagonal(factors, 0.0)
    
    # grad_pair_i = - sum_j factors[i,j] * (p_i - p_j)
    # = - p_i * sum_j factors[i,j] + sum_j factors[i,j] * p_j
    sum_factors = np.sum(factors, axis=1, keepdims=True)
    weighted_p = np.sum(factors[:, :, np.newaxis] * c[np.newaxis, :, :], axis=1)
    grad_pairs = -c * sum_factors + weighted_p
    
    # 2. Wall contribution
    # Wall weights are weights[len(pair_dists):]
    wall_weights = weights[len(pair_dists):]
    # wall_vals shape (N, 3). weights shape (N*3,).
    wall_w_matrix = wall_weights.reshape(N, 3)
    
    # dv_ik/dp_k = -sign(c_ik)
    # grad_wall_i = - sum_k w_ik * (-sign(c_ik)) = sum_k w_ik * sign(c_ik)
    signs = np.sign(c)
    grad_walls = wall_w_matrix * signs
    
    total_grad = (grad_pairs + grad_walls).flatten()
    
    return obj_val, total_grad

def construct():
    t0 = time.time()
    
    # Initial grid
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].copy()
    
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.RandomState(42)
    
    # Annealing temperature
    # Start with a higher T to explore, decrease to focus on min distance
    T_start = 0.05
    T_end = 0.001
    
    # Number of optimization steps
    max_steps = 30
    
    current_T = T_start
    
    for step in range(max_steps):
        if time.time() - t0 > DEADLINE - 5:
            break
            
        # Anneal T
        # Exponential decay
        current_T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Add small noise to break symmetry
        noise = rng.normal(0, 0.002, size=(N, 3))
        init_pts = np.clip(best_pts + noise, -0.49, 0.49)
        
        def objective(x):
            e, g = _soft_min_obj_and_grad(x, current_T)
            return e, g
            
        try:
            result = minimize(
                objective,
                init_pts.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'gtol': 1e-5, 'disp': False}
            )
            opt_pts = result.x.reshape(-1, 3)
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5, 0.5)
                r_opt = _score(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()
                else:
                    # If no improvement, accept anyway to help escape local minima?
                    # Or reject? Usually accept if it's close or better in soft-min metric.
                    # Let's check soft-min value.
                    e_new, _ = _soft_min_obj_and_grad(opt_pts, current_T)
                    e_old, _ = _soft_min_obj_and_grad(best_pts, current_T)
                    if e_new < e_old:
                        best_pts = opt_pts.copy()
                        # Even if r didn't improve immediately, the configuration might be better
                        # for the next step with lower T.
                        # But we must return the best R found so far.
                        
        except Exception:
            pass
            
    # Final Polish
    # Use a very sharp soft-min or direct local search to maximize r
    if time.time() - t0 < DEADLINE - 2:
        T_polish = 0.0005
        def obj_polish(x):
            e, g = _soft_min_obj_and_grad(x, T_polish)
            return e, g
            
        try:
            result = minimize(
                obj_polish,
                best_pts.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'gtol': 1e-7, 'disp': False}
            )
            opt_pts = result.x.reshape(-1, 3)
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5, 0.5)
                r_opt = _score(opt_pts)
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()
        except Exception:
            pass
            
    return best_pts
