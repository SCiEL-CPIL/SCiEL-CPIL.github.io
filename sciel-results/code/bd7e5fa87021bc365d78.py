import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def _pairwise_sq_dists(P):
    """Compute squared distances between all pairs of points."""
    # P is (N, D)
    # dist^2[i, j] = ||P[i] - P[j]||^2
    # Using broadcasting
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=-1)
    return sq_dists

def _objective_and_grad(x, beta):
    """
    x: flattened array of shape (N*D,)
    beta: temperature for soft-min/max
    Returns: (objective_value, gradient)
    """
    P = x.reshape(N, D)
    
    # Compute pairwise squared distances
    sq_dists = _pairwise_sq_dists(P)
    
    # Get upper triangle indices (i < j)
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_vals = sq_dists[i_idx, j_idx]
    
    # Soft max and min
    # To avoid overflow, subtract max for exp
    max_d = np.max(d_vals)
    min_d = np.min(d_vals)
    
    # Soft max: (1/beta) * log(sum(exp(beta * d)))
    # exp(beta * d - beta * max_d) to stabilize
    exp_pos = np.exp(beta * (d_vals - max_d))
    sum_pos = np.sum(exp_pos)
    M_soft = (1.0 / beta) * (np.log(sum_pos) + beta * max_d)
    
    # Soft min: -(1/beta) * log(sum(exp(-beta * d)))
    # exp(-beta * d + beta * min_d) to stabilize? 
    # exp(-beta * d) is small for large d. 
    # Let's use: -(1/beta) * log(sum(exp(-beta * (d - min_d)))) - min_d
    # Actually, softmin(x) = -softmax(-x)
    # softmax(-d) uses max(-d) = -min_d
    exp_neg = np.exp(-beta * (d_vals - min_d))
    sum_neg = np.sum(exp_neg)
    m_soft = -(1.0 / beta) * (np.log(sum_neg) - beta * min_d)
    # Wait, let's re-verify softmin formula.
    # m = -1/beta * log(sum(exp(-beta d)))
    # Let d' = d - min_d. min(d') = 0.
    # sum(exp(-beta (d' + min_d))) = exp(-beta min_d) * sum(exp(-beta d'))
    # log(sum) = -beta min_d + log(sum(exp(-beta d')))
    # m = -1/beta * (-beta min_d + log(sum_neg_stable)) = min_d - 1/beta * log(sum_neg_stable)
    # This is correct.
    
    obj = M_soft / m_soft
    
    # Gradient
    # d M_soft / d d_k = (exp(beta d_k) / sum_pos) * beta / beta = exp_pos_k / sum_pos
    # d m_soft / d d_k = 
    # m_soft = min_d - 1/beta log(sum_neg)
    # dm/dd_k = -1/beta * (1/sum_neg) * (-beta exp_neg_k) = exp_neg_k / sum_neg
    
    w_pos = exp_pos / sum_pos
    w_neg = exp_neg / sum_neg
    
    # grad_obj wrt d_vals
    # obj = M / m
    # d(obj)/d(d_k) = (1/m) * dM/dd_k - (M/m^2) * dm/dd_k
    # dM/dd_k = w_pos_k
    # dm/dd_k = w_neg_k
    grad_d = (w_pos / m_soft) - (M_soft / (m_soft * m_soft)) * w_neg
    
    # grad_d is shape (N*(N-1)/2,)
    # Now map back to P
    grad_P = np.zeros_like(P)
    
    # For each pair (i, j), d_k = ||P_i - P_j||^2
    # d(d_k)/d(P_i) = 2(P_i - P_j)
    # d(d_k)/d(P_j) = 2(P_j - P_i)
    
    # Loop is fine for N=14 (91 pairs)
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        coeff = 2.0 * grad_d[k]
        vec = P[i] - P[j]
        grad_P[i] += coeff * vec
        grad_P[j] -= coeff * vec
        
    return obj, grad_P.flatten()

def _generate_initial_points():
    """Generate initial points on a Fibonacci sphere."""
    # Fibonacci sphere
    n = N
    offset = 2 / n
    increment = np.pi * (3 - np.sqrt(5))
    
    y = np.arange(n) - n / 2 + 0.5
    y /= (n / 2 - 0.5)
    
    r = np.sqrt(1 - y**2)
    phi = np.arange(n) * increment
    
    x = np.cos(phi) * r
    z = np.sin(phi) * r
    
    P = np.column_stack((x, y, z))
    
    # Center at origin
    P -= np.mean(P, axis=0)
    
    # Scale to unit norm roughly? 
    # The objective is scale invariant. 
    # Let's scale so that mean squared distance is 1.
    sq_dists = _pairwise_sq_dists(P)
    i_idx, j_idx = np.triu_indices(N, k=1)
    mean_d2 = np.mean(sq_dists[i_idx, j_idx])
    if mean_d2 > 0:
        P /= np.sqrt(mean_d2)
        
    return P

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_score = np.inf
    best_P = None
    
    # Try a few different initializations
    # 1. Fibonacci sphere
    # 2. Random points on sphere
    # 3. Grid (fallback)
    
    inits = []
    
    # Fibonacci
    P0 = _generate_initial_points()
    inits.append(P0)
    
    # Rotated Fibonacci
    for _ in range(5):
        # Random rotation
        theta = np.random.uniform(0, 2*np.pi, 3)
        # Simple rotation around axes
        Rx = np.array([[1, 0, 0], [0, np.cos(theta[0]), -np.sin(theta[0])], [0, np.sin(theta[0]), np.cos(theta[0])]])
        Ry = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])], [0, 1, 0], [-np.sin(theta[1]), 0, np.cos(theta[1])]])
        Rz = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0], [np.sin(theta[2]), np.cos(theta[2]), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        P_rot = P0 @ R.T
        inits.append(P_rot)
        
    # Add some noise to Fibonacci
    P_noise = P0 + 0.1 * np.random.randn(N, D)
    inits.append(P_noise)

    beta = 50.0  # Temperature for soft min/max
    
    for P_init in inits:
        if time.time() > deadline:
            break
            
        x0 = P_init.flatten()
        
        try:
            res = minimize(
                _objective_and_grad, 
                x0, 
                args=(beta,), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if res.success or res.fun < best_score:
                P_opt = res.x.reshape(N, D)
                # Calculate exact score
                sq_dists = _pairwise_sq_dists(P_opt)
                i_idx, j_idx = np.triu_indices(N, k=1)
                d_vals = sq_dists[i_idx, j_idx]
                min_d = np.min(d_vals)
                max_d = np.max(d_vals)
                
                if min_d > 0:
                    score = (max_d / min_d) ** 2
                    if score < best_score:
                        best_score = score
                        best_P = P_opt
        except Exception:
            continue
            
    if best_P is None:
        # Fallback to initial
        return P0
    
    return best_P
