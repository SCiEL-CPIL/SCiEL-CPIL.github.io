import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _min_area_bary_fast(points_bary):
    """Vectorized minimum triangle area in barycentric coordinates."""
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    n = len(u)
    min_a = float('inf')
    
    # Compute all determinants in a vectorized manner
    # For each triple (i,j,k): det = u_i*(v_j*w_k - w_j*v_k) - v_i*(u_j*w_k - w_j*u_k) + w_i*(u_j*v_k - v_j*u_k)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Vectorized over k
            idx = np.arange(n)
            mask = (idx > j)
            if not np.any(mask):
                continue
            ks = idx[mask]
            
            u_i, v_i, w_i = u[i], v[i], w[i]
            u_j, v_j, w_j = u[j], v[j], w[j]
            u_k, v_k, w_k = u[ks], v[ks], w[ks]
            
            dets = (u_i * (v_j * w_k - w_j * v_k) 
                    - v_i * (u_j * w_k - w_j * u_k) 
                    + w_i * (u_j * v_k - v_j * u_k))
            
            a = np.abs(dets)
            min_a = min(min_a, np.min(a))
    
    return min_a

def _soft_min_area(points_bary, beta=50.0):
    """Soft minimum using log-sum-exp for smooth optimization."""
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    n = len(u)
    all_areas = []
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                det = (u[i] * (v[j] * w[k] - w[j] * v[k]) 
                       - v[i] * (u[j] * w[k] - w[j] * u[k]) 
                       + w[i] * (u[j] * v[k] - v[j] * u[k]))
                all_areas.append(abs(det))
    
    all_areas = np.array(all_areas)
    # Log-sum-exp for soft minimum
    max_a = np.max(all_areas)
    soft_min = max_a + np.log(np.mean(np.exp(all_areas - max_a)))
    
    # Actually we want the minimum, so use negative log-sum-exp
    min_a = np.min(all_areas)
    # Soft minimum: -log(mean(exp(-beta * a))) / beta
    neg_beta_areas = -beta * all_areas
    max_neg = np.max(neg_beta_areas)
    log_mean_exp = max_neg + np.log(np.mean(np.exp(neg_beta_areas - max_neg)))
    soft_min = -log_mean_exp / beta
    
    return soft_min

def _bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian."""
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3) / 2])
    
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    pts = u[:, np.newaxis] * A + v[:, np.newaxis] * B + w[:, np.newaxis] * C
    return pts

def _clamp_to_triangle(bary):
    """Clamp barycentric coordinates to valid region."""
    bary = bary.copy()
    bary[:, 0] = np.clip(bary[:, 0], 0, 1)
    bary[:, 1] = np.clip(bary[:, 1], 0, 1)
    sum_uv = bary[:, 0] + bary[:, 1]
    mask = sum_uv > 1
    if np.any(mask):
        factor = 1.0 / sum_uv[mask]
        bary[mask, 0] *= factor
        bary[mask, 1] *= factor
    return bary

def _polish(bary, deadline):
    """High-precision polish pass with tighter tolerances."""
    def objective(params):
        uv = params.reshape(N, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1 - u - v
        
        penalty = 0
        if np.any(w < 0):
            penalty = 100 * np.sum(w[w < 0]**2)
        if np.any(u < 0) or np.any(v < 0):
            penalty += 100 * (np.sum(u[u < 0]**2) + np.sum(v[v < 0]**2))
        
        min_a = _min_area_bary_fast(uv)
        return -min_a + penalty
    
    try:
        res = minimize(
            objective,
            bary.flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1)] * (2 * N),
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        
        uv_opt = res.x.reshape(N, 2)
        w_opt = 1 - uv_opt[:, 0] - uv_opt[:, 1]
        if np.any(w_opt < -1e-8) or np.any(uv_opt < -1e-8):
            return bary
        
        return uv_opt
    except Exception:
        return bary

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = -float('inf')
    best_points = None
    
    # Initial configurations
    inits = []
    
    # 1. 10-point grid + center
    grid_10 = [
        (1.0, 0.0), (2/3, 1/3), (1/3, 2/3), (0.0, 1.0),
        (2/3, 0.0), (1/3, 1/3), (0.0, 2/3),
        (1/3, 0.0), (0.0, 1/3),
        (0.0, 0.0)
    ]
    inits.append(np.array(grid_10 + [(0.4, 0.4)]))
    
    # 2. Random perturbations
    rng = np.random.RandomState(42)
    
    for i in range(30):
        base = np.array(grid_10 + [(0.4, 0.4)])
        noise = rng.uniform(-0.1, 0.1, size=base.shape)
        perturbed = base + noise
        perturbed = _clamp_to_triangle(perturbed)
        inits.append(perturbed)
        
        # Random Dirichlet
        bary = rng.dirichlet([1, 1, 1], size=N)
        inits.append(bary[:, :2])
        
        if time.time() > deadline - 15:
            break
    
    # Optimization loop with soft surrogate
    for init in inits:
        if time.time() > deadline - 10:
            break
        
        # First pass: soft surrogate to escape local minima
        def soft_objective(params):
            uv = params.reshape(N, 2)
            u = uv[:, 0]
            v = uv[:, 1]
            w = 1 - u - v
            
            penalty = 0
            if np.any(w < -1e-4):
                penalty = 1000 * np.sum(w[w < -1e-4]**2)
            if np.any(u < -1e-4) or np.any(v < -1e-4):
                penalty += 1000 * (np.sum(u[u < -1e-4]**2) + np.sum(v[v < -1e-4]**2))
            
            soft_a = _soft_min_area(uv, beta=30.0)
            return soft_a + penalty
        
        try:
            res_soft = minimize(
                soft_objective,
                init.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            uv_soft = res_soft.x.reshape(N, 2)
            uv_soft = _clamp_to_triangle(uv_soft)
            
            # Second pass: high-precision polish
            if time.time() < deadline - 5:
                uv_polished = _polish(uv_soft, deadline)
            else:
                uv_polished = uv_soft
            
            # Evaluate
            min_a = _min_area_bary_fast(uv_polished)
            if min_a > best_score:
                best_score = min_a
                best_points = uv_polished.copy()
                
        except Exception:
            pass
    
    # Additional random restarts if time permits
    extra_restarts = 5
    for i in range(extra_restarts):
        if time.time() > deadline - 5:
            break
        
        bary = rng.dirichlet([1, 1, 1], size=N)[:, :2]
        
        def soft_objective(params):
            uv = params.reshape(N, 2)
            u = uv[:, 0]
            v = uv[:, 1]
            w = 1 - u - v
            
            penalty = 0
            if np.any(w < -1e-4):
                penalty = 1000 * np.sum(w[w < -1e-4]**2)
            if np.any(u < -1e-4) or np.any(v < -1e-4):
                penalty += 1000 * (np.sum(u[u < -1e-4]**2) + np.sum(v[v < -1e-4]**2))
            
            soft_a = _soft_min_area(uv, beta=30.0)
            return soft_a + penalty
        
        try:
            res_soft = minimize(
                soft_objective,
                bary.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            uv_soft = res_soft.x.reshape(N, 2)
            uv_soft = _clamp_to_triangle(uv_soft)
            
            if time.time() < deadline - 3:
                uv_polished = _polish(uv_soft, deadline)
            else:
                uv_polished = uv_soft
            
            min_a = _min_area_bary_fast(uv_polished)
            if min_a > best_score:
                best_score = min_a
                best_points = uv_polished.copy()
                
        except Exception:
            pass
    
    if best_points is None:
        best_points = np.array(grid_10 + [(0.4, 0.4)])
    
    # Convert to Cartesian
    pts = _bary_to_cart(best_points)
    
    return pts
