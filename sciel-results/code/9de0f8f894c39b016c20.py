import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145

    # Generate a triangular lattice
    R = 10
    pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            pts.append((x, y))
    
    pts = np.array(pts)
    num_pts = len(pts)
    
    dists_from_origin = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    
    pool_size = 300
    candidate_indices = sorted_indices[:pool_size]
    
    def score_subset(indices):
        sub_pts = pts[indices]
        diff = sub_pts[:, np.newaxis, :] - sub_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    # Multiple initial configurations via different strategies
    initial_configs = []
    
    # Config 1: 16 closest to origin
    initial_configs.append(sorted_indices[:N].copy())
    
    # Config 2: 16 closest to a slightly offset center
    offset_pts = pts.copy()
    offset_pts[:, 0] -= 0.3
    offset_dists = np.sum(offset_pts**2, axis=1)
    offset_sorted = np.argsort(offset_dists)
    initial_configs.append(offset_sorted[:N].copy())
    
    # Config 3: Random subset from candidate pool
    np.random.seed(42)
    random_subset = np.random.choice(candidate_indices, size=N, replace=False)
    initial_configs.append(random_subset.copy())
    
    # Config 4: Another random with different seed
    np.random.seed(123)
    random_subset2 = np.random.choice(candidate_indices, size=N, replace=False)
    initial_configs.append(random_subset2.copy())
    
    # Config 5: Symmetric ring pattern - 1 center + 5 inner + 10 outer
    ring_pts = np.zeros((N, 2))
    ring_pts[0] = [0, 0]
    for i in range(1, 6):
        angle = 2 * np.pi * i / 5
        ring_pts[i] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
    for i in range(6, N):
        angle = 2 * np.pi * (i - 6) / 10
        ring_pts[i] = [1.2 * np.cos(angle), 1.2 * np.sin(angle)]
    
    # Find closest lattice points to ring positions
    ring_config = []
    used = set()
    for rp in ring_pts:
        dists_to_ring = np.sum((pts - rp)**2, axis=1)
        dists_to_ring[list(used)] = 1e10
        best_idx = np.argmin(dists_to_ring)
        ring_config.append(best_idx)
        used.add(best_idx)
    initial_configs.append(np.array(ring_config))
    
    # Greedy improvement for each config
    def greedy_improve(initial_indices, time_budget=10):
        current_indices = initial_indices.copy()
        current_score = score_subset(current_indices)
        
        in_subset = set(current_indices)
        outside_candidates = [i for i in candidate_indices if i not in in_subset]
        
        improved = True
        while improved and time.time() < start_time + time_budget:
            improved = False
            for i_idx, i in enumerate(current_indices):
                if time.time() > start_time + time_budget:
                    break
                for j in outside_candidates:
                    if time.time() > start_time + time_budget:
                        break
                    new_indices = current_indices.copy()
                    new_indices[i_idx] = j
                    new_score = score_subset(new_indices)
                    if new_score < current_score:
                        current_score = new_score
                        current_indices = new_indices
                        in_subset = set(current_indices)
                        outside_candidates = [k for k in candidate_indices if k not in in_subset]
                        improved = True
                        break
                if improved:
                    break
                    
        return current_indices, current_score

    best_indices = None
    best_lattice_score = 1e10
    
    for config in initial_configs:
        if time.time() > deadline - 60:
            break
        improved_indices, sc = greedy_improve(config, time_budget=8)
        if sc < best_lattice_score:
            best_lattice_score = sc
            best_indices = improved_indices
    
    best_pts = pts[best_indices]
    
    # Normalize: fix closest pair to (0,0) and (1,0)
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    d = dists[iu]
    min_idx = np.argmin(d)
    pair = np.unravel_index(min_idx, (N, N))
    i0, i1 = pair
    if i0 > i1:
        i0, i1 = i1, i0
    
    p0 = best_pts[i0]
    p1 = best_pts[i1]
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        return best_pts
    
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    transformed = (best_pts - p0) @ R_mat.T / norm_v
    
    perm = list(range(N))
    perm[0] = i0
    perm[1] = i1
    remaining = [i for i in range(N) if i != i0 and i != i1]
    perm[2:] = remaining
    transformed = transformed[perm]
    
    init_params = transformed[2:].flatten()
    
    def exact_objective(params):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_opt = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d_opt = dists_opt[iu]
        
        min_d = np.min(d_opt)
        max_d = np.max(d_opt)
        
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    def exact_gradient(params):
        """Analytical gradient of (max_d/min_d)^2"""
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-8:
            return np.zeros_like(params)
        
        min_d = np.sqrt(min_d_sq)
        max_d = np.sqrt(max_d_sq)
        ratio_sq = (max_d / min_d)**2
        
        # Gradient w.r.t. d_sq for each pair
        # f = (max_d/min_d)^2
        # df/dmax_d = 2*max_d/min_d^2
        # df/dmin_d = -2*max_d^2/min_d^3
        # dmax_d/ddmax_d_sq = 1/(2*max_d)
        # dmin_d/ddmin_d_sq = 1/(2*min_d)
        
        grad_d_sq = np.zeros_like(d_sq)
        max_idx = np.argmax(d_sq)
        min_idx = np.argmin(d_sq)
        
        # Only the max and min pairs contribute to gradient
        grad_d_sq[max_idx] = 1.0 / max_d  # df/dmax_d * dmax_d/ddmax_d_sq
        grad_d_sq[min_idx] = -1.0 / min_d  # df/dmin_d * dmin_d/ddmin_d_sq
        
        # Scale by ratio
        # Actually: f = max_d^2 / min_d^2
        # df/dmax_d_sq = 2*max_d^2/min_d^2 * (1/max_d_sq) = 2/max_d_sq * max_d^2/min_d^2 = 2*f/max_d_sq
        # df/dmin_d_sq = -2*max_d^2/min_d^2 * (1/min_d_sq) = -2*f/min_d_sq
        
        grad_d_sq = np.zeros_like(d_sq)
        grad_d_sq[max_idx] = 2.0 / max_d_sq
        grad_d_sq[min_idx] = -2.0 / min_d_sq
        grad_d_sq *= ratio_sq
        
        # Now compute gradient w.r.t. params
        # d_sq[i,j] = |p_i - p_j|^2
        # d(d_sq[i,j])/dp_k = 2*(p_k - p_i) if k==i, 2*(p_k - p_j) if k==j, 0 otherwise
        
        grad_params = np.zeros((N, 2))
        for k in range(len(d_sq)):
            i, j = iu[0][k], iu[1][k]
            if grad_d_sq[k] != 0:
                grad_params[i] += 2 * grad_d_sq[k] * (pts_opt[i] - pts_opt[j])
                grad_params[j] += 2 * grad_d_sq[k] * (pts_opt[j] - pts_opt[i])
        
        # Only params[2:] are variables
        return grad_params[2:].flatten()

    # Smooth surrogate with annealed beta
    def soft_objective(params, beta=20.0):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Numerically stable log-sum-exp
        max_d_sq = np.max(d_sq)
        min_d_sq = np.min(d_sq)
        
        # Soft max of d_sq
        shifted_max = d_sq - max_d_sq
        soft_max_d_sq = max_d_sq + np.log(np.sum(np.exp(beta * shifted_max))) / beta
        
        # Soft min of d_sq
        shifted_min = d_sq - min_d_sq
        soft_min_d_sq = min_d_sq - np.log(np.sum(np.exp(-beta * shifted_min))) / beta
        
        if soft_min_d_sq < 1e-8:
            return 1e10
        
        max_d_soft = np.sqrt(soft_max_d_sq)
        min_d_soft = np.sqrt(soft_min_d_sq)
        
        return (max_d_soft / min_d_soft)**2

    bnds = [(-15, 15)] * (2 * (N - 2))
    
    best_params = init_params
    best_score = exact_objective(init_params)
    
    # Phase 1: Soft surrogate with annealing
    if time.time() < deadline - 40:
        for beta in [5.0, 10.0, 20.0, 40.0]:
            if time.time() > deadline - 15:
                break
            try:
                def soft_obj_fixed(params, b=beta):
                    return soft_objective(params, b)
                
                res = minimize(soft_obj_fixed, best_params, method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < 1e9:
                    exact_score = exact_objective(res.x)
                    if exact_score < best_score:
                        best_params = res.x
                        best_score = exact_score
            except Exception:
                pass
    
    # Phase 2: Exact objective with analytical gradient
    if time.time() < deadline - 20:
        try:
            res = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                           jac=exact_gradient,
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish (derivative-free)
    if time.time() < deadline - 12:
        try:
            res = minimize(exact_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Phase 4: High-precision L-BFGS-B polish with gradient
    if time.time() < deadline - 5:
        try:
            res = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                           jac=exact_gradient,
                           options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12})
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Phase 5: Final ultra-precise polish
    if time.time() < deadline - 2:
        try:
            res = minimize(exact_objective, best_params, method='L-BFGS-B', bounds=bnds,
                           jac=exact_gradient,
                           options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50})
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass
    
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    return final_pts

# EVOLVE-BLOCK-END
