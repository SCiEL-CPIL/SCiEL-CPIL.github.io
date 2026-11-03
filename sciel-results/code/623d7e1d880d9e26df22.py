"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    
    # Distance to boundary
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    
    # Pairwise distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return min(r_pair, r_bound)

def _get_initial_ring():
    """Generate a single ring of N points with equal angular spacing."""
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Estimate radius: for N points on a ring, chord length = 2*R*sin(pi/N)
    # We want chord/2 = 1 - R  =>  R*sin(pi/N) = 1 - R  =>  R(1+sin(pi/N)) = 1
    s = np.sin(np.pi / N)
    R = 1.0 / (1.0 + s)
    # But this assumes all points are on the ring at radius R.
    # Actually, for a ring packing, the radius of small circles r satisfies:
    # 2*r = 2*R*sin(pi/N)  =>  r = R*sin(pi/N)
    # and  r = 1 - R
    # So R*sin(pi/N) = 1 - R  =>  R = 1/(1+sin(pi/N))
    # r = sin(pi/N)/(1+sin(pi/N))
    # For N=50: sin(pi/50) ~ 0.06279, R ~ 0.9410, r ~ 0.0590
    # This is much smaller than the optimal 0.1258.
    # So a single ring is NOT the optimal structure for N=50.
    # The optimal packing has multiple rings.
    
    # Let's use a better initialization: hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0 
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.98

def _neg_score(P_flat):
    """Objective function for minimization: negative of the score."""
    P = P_flat.reshape(N, 2)
    score = _compute_score(P)
    return -score

def _soft_min_score(P, beta=100.0):
    """Smooth surrogate for the min score using log-sum-exp."""
    P = P.reshape(N, 2)
    
    # Boundary distances
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    
    # Pairwise distances / 2
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist2, np.inf)
    # We want min of dist/2, which is 0.5 * min(dist)
    # Use soft min: -1/beta * log(sum(exp(-beta * x)))
    # For pairwise: x_ij = dist_ij / 2
    # For boundary: x_i = d_bound_i
    
    # Combine all constraints into one soft min
    # r = min(min_ij dist_ij/2, min_i d_bound_i)
    # -beta * r = max(max_ij -beta*dist_ij/2, max_i -beta*d_bound_i)
    # Use log-sum-exp for max
    
    # Pairwise terms: -beta * dist_ij / 2 for i < j
    # Boundary terms: -beta * d_bound_i
    
    # To avoid overflow, use the trick: log(sum(exp(a_i))) = max(a) + log(sum(exp(a_i - max(a))))
    
    # Get all the negative values
    # Pairwise: -0.5 * beta * sqrt(dist2) for i<j
    # This is expensive to compute fully, so let's compute pairwise distances first
    dist = np.sqrt(dist2)
    np.fill_diagonal(dist, np.inf)
    
    # Get upper triangle
    iu = np.triu_indices(N, k=1)
    pair_vals = -0.5 * beta * dist[iu]
    
    bound_vals = -beta * d_bound
    
    all_vals = np.concatenate([pair_vals, bound_vals])
    
    m = np.max(all_vals)
    if not np.isfinite(m):
        return -np.inf
    s = np.sum(np.exp(all_vals - m))
    soft_max = m + np.log(s)
    
    # soft_max = -beta * r_soft
    # r_soft = -soft_max / beta
    r_soft = -soft_max / beta
    
    return r_soft

def _neg_soft_score(P_flat, beta=100.0):
    P = P_flat.reshape(N, 2)
    r = _soft_min_score(P, beta)
    return -r

def _local_refine(P, deadline):
    """Local refinement using Nelder-Mead with tight tolerances."""
    P_curr = P.copy()
    score_curr = _compute_score(P_curr)
    
    try:
        res = minimize(_neg_score, P_curr.flatten(), method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                'adaptive': True})
        P_opt = res.x.reshape(N, 2)
        score_opt = _compute_score(P_opt)
        if score_opt > score_curr:
            P_curr = P_opt
            score_curr = score_opt
    except Exception:
        pass
    
    return P_curr, score_curr

def _force_directed_refine(P, r, deadline, max_iters=5000):
    """Force-directed refinement to push points apart and keep them inside."""
    P_curr = P.copy()
    r_curr = r
    alpha = 0.005
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        forces = np.zeros_like(P_curr)
        
        # Pairwise repulsion
        diff = P_curr[:, np.newaxis, :] - P_curr[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        target = 2.0 * r_curr
        mask = dist < target * 1.5  # Consider pairs that are close
        
        if np.any(mask):
            overlap = np.zeros_like(dist)
            overlap[mask] = target - dist[mask]
            overlap[overlap < 0] = 0
            
            inv_dist = np.zeros_like(dist)
            np.divide(1.0, dist, out=inv_dist, where=dist > 0)
            
            weight = overlap * inv_dist
            weight[~mask] = 0
            
            sum_w = np.sum(weight, axis=1)
            term1 = P_curr * sum_w[:, np.newaxis]
            term2 = (P_curr.T @ weight).T
            
            forces += term1 - term2
        
        # Boundary repulsion
        dist_origin = np.linalg.norm(P_curr, axis=1)
        limit = 1.0 - r_curr
        outside = dist_origin > limit
        if np.any(outside):
            viol = dist_origin[outside] - limit
            unit_vec = P_curr[outside] / dist_origin[outside, np.newaxis]
            forces[outside] -= unit_vec * viol * 1.0
        
        # Center attraction to prevent drift
        forces -= 0.01 * P_curr
        
        P_new = P_curr + alpha * forces
        new_score = _compute_score(P_new)
        
        if new_score > r_curr:
            P_curr = P_new
            r_curr = new_score
            alpha = min(alpha * 1.1, 0.05)
        else:
            alpha *= 0.95
        
        if alpha < 1e-7:
            break
    
    return P_curr, r_curr

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Initialization
    P_init = _get_initial_ring()
    P_best = P_init.copy()
    score_best = _compute_score(P_best)
    
    # 2. Powell optimization
    try:
        res1 = minimize(_neg_score, P_best.flatten(), method='Powell',
                        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        P_opt1 = res1.x.reshape(N, 2)
        score_opt1 = _compute_score(P_opt1)
        
        if score_opt1 > score_best:
            P_best = P_opt1
            score_best = score_opt1
    except Exception:
        pass
    
    if time.time() > deadline - 30.0:
        return P_best
    
    # 3. Soft-min optimization with increasing beta
    try:
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() > deadline - 20.0:
                break
            res_soft = minimize(lambda x: _neg_soft_score(x, beta), P_best.flatten(),
                               method='Powell',
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
            P_soft = res_soft.x.reshape(N, 2)
            score_soft = _compute_score(P_soft)
            
            if score_soft > score_best:
                P_best = P_soft
                score_best = score_soft
    except Exception:
        pass
    
    # 4. Force-directed refinement
    if time.time() < deadline - 15.0:
        P_fd, r_fd = _force_directed_refine(P_best, score_best, deadline)
        if r_fd > score_best:
            P_best = P_fd
            score_best = r_fd
    
    # 5. Final high-precision polish
    if time.time() < deadline - 5.0:
        P_best, score_best = _local_refine(P_best, deadline)
    
    return P_best
# EVOLVE-BLOCK-END
