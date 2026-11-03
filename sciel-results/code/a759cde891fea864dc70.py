import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    
    # Estimate spacing
    r_est = np.sqrt(np.pi / (n_points * np.sqrt(3)))
    s_est = 2 * r_est
    
    for attempt in range(10):
        s = s_est * (0.95 + 0.02 * attempt)
        k = int(1.0 / s) + 5
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    points.append([x, y])
                    
        if len(points) < n_points:
            continue
            
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
        P = points[idx[:n_points]]
        
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)
        min_dist = np.min(dist_mat[np.triu_indices(n_points, k=1)])
        min_bound = np.min(1 - np.linalg.norm(P, axis=1))
        r_init = min(min_dist / 2.0, min_bound)
        
        if r_init > best_r:
            best_r = r_init
            best_pts = P
            
    if best_pts is None:
        P = np.zeros((n_points, 2))
        return P
        
    return best_pts

def _objective_and_grad(P_flat, tau):
    P = P_flat.reshape(-1, 2)
    N = P.shape[0]
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=2)
    
    iu = np.triu_indices(N, k=1)
    d_pair = dist_mat[iu] / 2.0
    
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    
    d_all = np.concatenate([d_pair, d_bound])
    
    d_min = np.min(d_all)
    exp_terms = np.exp((d_min - d_all) / tau)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    soft_min_val = d_min - tau * np.log(sum_exp)
    
    grad_P = np.zeros_like(P)
    
    # Pairwise
    row_idx, col_idx = iu
    diffs = P[row_idx] - P[col_idx]
    dists = np.linalg.norm(diffs, axis=1)
    dists[dists == 0] = 1e-12
    
    grad_vecs = 0.5 * diffs / dists[:, np.newaxis]
    
    np.add.at(grad_P, row_idx, weights[:len(d_pair)] * grad_vecs)
    np.add.at(grad_P, col_idx, -weights[:len(d_pair)] * grad_vecs)
    
    # Boundary
    norms = np.linalg.norm(P, axis=1)
    norms[norms == 0] = 1e-12
    w_bound = weights[len(d_pair):]
    grad_bound_vecs = -P / norms[:, np.newaxis]
    grad_P += w_bound[:, np.newaxis] * grad_bound_vecs
    
    # Minimize -soft_min
    return -soft_min_val, -grad_P.flatten()

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    P = _hex_init(N, seed=42)
    P_flat = P.flatten()
    
    tau_start = 0.05
    tau_end = 0.002
    n_steps = 30
    
    for i in range(n_steps):
        if time.time() - start_time > time_limit - 10.0:
            break
        t = i / (n_steps - 1)
        tau = tau_start * (tau_end / tau_start) ** t
        
        try:
            res = minimize(_objective_and_grad, P_flat, jac=True, method='L-BFGS-B', 
                           args=(tau,), options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            P_flat = res.x
        except Exception:
            pass
            
    P = P_flat.reshape(-1, 2)
    
    # Polish
    def calc_radius(P):
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)
        min_dist = np.min(dist_mat[np.triu_indices(N, k=1)])
        min_bound = np.min(1 - np.linalg.norm(P, axis=1))
        return min(min_dist / 2.0, min_bound)
        
    current_radius = calc_radius(P)
    best_radius = current_radius
    best_P = P.copy()
    
    step_size = 0.002
    for _ in range(5000):
        if time.time() - start_time > time_limit - 2.0:
            break
        i = np.random.randint(N)
        dir = np.random.normal(0, 1, size=2)
        dir /= np.linalg.norm(dir)
        
        P_try = P.copy()
        P_try[i] += dir * step_size
        
        if np.linalg.norm(P_try[i]) > 1.0:
            continue
            
        r_try = calc_radius(P_try)
        if r_try > current_radius:
            P = P_try
            current_radius = r_try
            if current_radius > best_radius:
                best_radius = current_radius
                best_P = P.copy()
            
    return best_P

# EVOLVE-BLOCK-END
