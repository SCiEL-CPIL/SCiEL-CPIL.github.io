import numpy as np
import time

N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 148  # 150s limit, leave small margin

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

    def soft_objective(params, beta=50.0):
        pts_opt = np.zeros((N, 2))
        pts_opt[0] = [0.0, 0.0]
        pts_opt[1] = [1.0, 0.0]
        pts_opt[2:] = params.reshape(-1, 2)
        
        diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Soft max: log(sum(exp(beta * d_sq))) / beta
        max_d_sq_soft = np.log(np.sum(np.exp(beta * d_sq))) / beta
        # Soft min: -log(sum(exp(-beta * d_sq))) / beta
        min_d_sq_soft = -np.log(np.sum(np.exp(-beta * d_sq))) / beta
        
        if min_d_sq_soft < 1e-8:
            return 1e10
        
        max_d_soft = np.sqrt(max_d_sq_soft)
        min_d_soft = np.sqrt(min_d_sq_soft)
        
        return (max_d_soft / min_d_soft)**2

    def normalize_and_parametrize(pts):
        """Normalize points so min distance = 1, align closest pair to (0,0) and (1,0)"""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        if min_d < 1e-8:
            return None
        min_idx = np.argmin(d)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair
        if i0 > i1: i0, i1 = i1, i0
        
        p0 = pts[i0]
        p1 = pts[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-8:
            return None
        
        cos_t = v[0] / norm_v
        sin_t = v[1] / norm_v
        R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        transformed = (pts - p0) @ R_mat.T / norm_v
        
        perm = list(range(N))
        perm[0] = i0
        perm[1] = i1
        remaining = [i for i in range(N) if i != i0 and i != i1]
        perm[2:] = remaining
        transformed = transformed[perm]
        
        return transformed[2:].flatten()

    bnds = [(-5, 5)] * (2 * (N - 2))
    
    # Generate diverse initial configurations
    rng = np.random.RandomState(42)
    initial_configs = []
    
    # Config 1: Hexagonal lattice subset
    R = 6
    pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            pts.append((x, y))
    pts = np.array(pts)
    dists_from_origin = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    lattice_pts = pts[sorted_indices[:N]]
    params = normalize_and_parametrize(lattice_pts)
    if params is not None:
        initial_configs.append(params)
    
    # Config 2: Square grid
    grid_pts = []
    for i in range(4):
        for j in range(4):
            grid_pts.append([i, j])
    grid_pts = np.array(grid_pts, dtype=float)
    params = normalize_and_parametrize(grid_pts)
    if params is not None:
        initial_configs.append(params)
    
    # Config 3: Circular arrangement with perturbation
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = np.ones(N)
    circ_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    circ_pts += rng.normal(0, 0.1, size=(N, 2))
    params = normalize_and_parametrize(circ_pts)
    if params is not None:
        initial_configs.append(params)
    
    # Config 4: Concentric rings (4 points each on 4 rings)
    ring_pts = []
    for ring_idx in range(4):
        radius = ring_idx + 1
        ring_angles = np.linspace(0, 2*np.pi, 4, endpoint=False) + ring_idx * np.pi/4
        ring_pts.extend([[radius * np.cos(a), radius * np.sin(a)] for a in ring_angles])
    ring_pts = np.array(ring_pts)
    params = normalize_and_parametrize(ring_pts)
    if params is not None:
        initial_configs.append(params)
    
    # Config 5: Random with minimum distance constraint
    for attempt in range(100):
        rand_pts = rng.uniform(-3, 3, size=(N, 2))
        diff = rand_pts[:, np.newaxis, :] - rand_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        if np.min(dists[iu]) > 0.5:
            params = normalize_and_parametrize(rand_pts)
            if params is not None:
                initial_configs.append(params)
                break
    
    # Optimize each initial config
    best_params = None
    best_score = 1e10
    
    for cfg_idx, init_params in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
        
        current_params = init_params.copy()
        current_score = exact_objective(current_params)
        
        # Phase 1: Soft surrogate with adaptive beta
        for beta in [15.0, 30.0, 60.0]:
            if time.time() > deadline - 10:
                break
            try:
                def soft_obj_adaptive(params, b=beta):
                    return soft_objective(params, beta=b)
                res = minimize(soft_obj_adaptive, current_params, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6})
                if res.fun < 1e9:
                    exact_score = exact_objective(res.x)
                    if exact_score < current_score:
                        current_params = res.x.copy()
                        current_score = exact_score
            except Exception:
                pass
        
        # Phase 2: Exact L-BFGS-B
        if time.time() < deadline - 8:
            try:
                res = minimize(exact_objective, current_params, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < current_score:
                    current_params = res.x.copy()
                    current_score = res.fun
            except Exception:
                pass
        
        # Phase 3: Powell for non-smoothness
        if time.time() < deadline - 6:
            try:
                res = minimize(exact_objective, current_params, method='Powell', bounds=bnds,
                               options={'maxiter': 300, 'ftol': 1e-10, 'xtol': 1e-8})
                if res.fun < current_score:
                    current_params = res.x.copy()
                    current_score = res.fun
            except Exception:
                pass
        
        # Phase 4: Final polish
        if time.time() < deadline - 3:
            try:
                res = minimize(exact_objective, current_params, method='L-BFGS-B', bounds=bnds, 
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.fun < current_score:
                    current_params = res.x.copy()
                    current_score = res.fun
            except Exception:
                pass
        
        if current_score < best_score:
            best_score = current_score
            best_params = current_params.copy()
    
    # Global perturbation search to escape local minima
    if best_params is not None and time.time() < deadline - 5:
        for perturb_iter in range(50):
            if time.time() > deadline - 5:
                break
            # Perturb best solution
            perturbation_scale = 0.3 / (1 + perturb_iter * 0.1)
            perturbed = best_params + rng.normal(0, perturbation_scale, size=best_params.shape)
            # Clip to bounds
            for i in range(len(perturbed)):
                perturbed[i] = np.clip(perturbed[i], bnds[i][0], bnds[i][1])
            
            perturbed_score = exact_objective(perturbed)
            
            # Quick local optimization from perturbed point
            try:
                res = minimize(exact_objective, perturbed, method='Powell', bounds=bnds,
                               options={'maxiter': 100, 'ftol': 1e-10, 'xtol': 1e-8})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    final_pts = np.zeros((N, 2))
    final_pts[0] = [0.0, 0.0]
    final_pts[1] = [1.0, 0.0]
    final_pts[2:] = best_params.reshape(-1, 2)
    
    # Verify no duplicate points
    diff = final_pts[:, np.newaxis, :] - final_pts[np.newaxis, :, :]
    dists_final = np.sqrt(np.sum(diff**2, axis=2))
    iu_final = np.triu_indices(N, 1)
    d_final = dists_final[iu_final]
    if np.min(d_final) < 1e-10:
        # Fallback
        return lattice_pts
    
    return final_pts

# EVOLVE-BLOCK-END
