# EVOLVE-BLOCK-START

import numpy as np
import time

N = 26

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist[mask] < (min_dist[mask] - tol)):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(200):
        valid = True
        
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _init_grid():
    """Grid-based initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    k = 0
    for i in range(5):
        for j in range(5):
            if k >= N:
                break
            pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
            radii[k] = 0.095
            k += 1
    if k < N:
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _init_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _solve_cvxpy(deadline):
    """Attempt to solve using CVXPY SOCP formulation (maximize sum of radii)."""
    try:
        import cvxpy as cp
    except ImportError:
        return None
        
    x = cp.Variable((N, 2))
    r = cp.Variable(N)
    
    objective = cp.Maximize(cp.sum(r))
    
    constraints = [
        r >= 0,
        x[:, 0] - r >= 0,
        1 - x[:, 0] - r >= 0,
        x[:, 1] - r >= 0,
        1 - x[:, 1] - r >= 0,
    ]
    
    # For each pair, we need ||x_i - x_j|| >= r_i + r_j
    # This is a non-convex constraint. Instead, we can use the dual:
    # Introduce auxiliary variables and use the fact that for fixed positions,
    # we want to maximize radii. But this is still non-convex.
    # 
    # Alternative: Use a heuristic approach - solve a relaxed problem where
    # we enforce ||x_i - x_j|| >= r_i + r_j + eps via SOCP by reformulating.
    # Actually, ||x_i - x_j|| >= r_i + r_j is equivalent to:
    # there exists a unit vector u such that (x_i - x_j)·u >= r_i + r_j
    # But this is still non-convex.
    #
    # Let's skip CVXPY for this non-convex problem and rely on local methods.
    return None

def _solve_slsqp_vectorized(pos, radii, deadline):
    """Solve using SLSQP with vectorized constraints."""
    from scipy.optimize import minimize
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    def obj(x):
        return -np.sum(x[2*N:])
        
    def cons_fun(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        # Boundary constraints
        c_bound = np.concatenate([
            p[:, 0] - r,
            1.0 - p[:, 0] - r,
            p[:, 1] - r,
            1.0 - p[:, 1] - r
        ])
        
        # Overlap constraints
        dx = p[:, 0, np.newaxis] - p[np.newaxis, :, 0]
        dy = p[:, 1, np.newaxis] - p[np.newaxis, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        c_overlap = (dist - min_dist)[mask]
        
        return np.concatenate([c_bound, c_overlap])
        
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, 
                       constraints=[{'type': 'ineq', 'fun': cons_fun}],
                       options={'maxiter': 500, 'ftol': 1e-12})
        if res.success:
            p = res.x[:2*N].reshape(N, 2)
            r = res.x[2*N:]
            p, r = _post_process(p, r)
            if _is_valid(p, r):
                return p, r
    except Exception:
        pass
    return None

def _gradient_refine(pos, radii, deadline):
    """Gradient-based refinement using projected gradient ascent on radii."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(123)
    lr = 0.01
    n_iter = 0
    
    while time.time() < deadline - 3 and n_iter < 2000:
        n_iter += 1
        
        # Compute gradients of objective w.r.t. positions (push circles apart)
        grad_pos = np.zeros((N, 2))
        grad_r = np.zeros(N)
        
        for i in range(N):
            # Boundary forces
            if best_pos[i, 0] - best_radii[i] < 0.01:
                grad_pos[i, 0] -= 1.0
            if 1.0 - best_pos[i, 0] - best_radii[i] < 0.01:
                grad_pos[i, 0] += 1.0
            if best_pos[i, 1] - best_radii[i] < 0.01:
                grad_pos[i, 1] -= 1.0
            if 1.0 - best_pos[i, 1] - best_radii[i] < 0.01:
                grad_pos[i, 1] += 1.0
            
            # Pairwise repulsion
            for j in range(N):
                if i == j:
                    continue
                dx = best_pos[i, 0] - best_pos[j, 0]
                dy = best_pos[i, 1] - best_pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-12
                sum_r = best_radii[i] + best_radii[j]
                if dist < sum_r:
                    # Push apart
                    force = (sum_r - dist) / dist
                    grad_pos[i, 0] += force * dx
                    grad_pos[i, 1] += force * dy
                    grad_r[i] -= force * 0.5
                    grad_r[j] -= force * 0.5
        
        # Update positions (projected gradient)
        best_pos += lr * grad_pos
        best_pos = np.clip(best_pos, 0.01, 0.99)
        
        # Update radii
        best_radii += lr * grad_r
        best_radii = np.clip(best_radii, 0.001, 0.3)
        
        # Post-process to ensure validity
        best_pos, best_radii = _post_process(best_pos, best_radii)
        
        score = np.sum(best_radii)
        if score > best_score:
            best_score = score
    
    return best_pos, best_radii

def _local_refine(pos, radii, deadline):
    """Iterative local refinement with random perturbations and gradient steps."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(42)
    
    while time.time() < deadline - 3:
        # Gradient refinement
        if time.time() < deadline - 5:
            gpos, gradii = _gradient_refine(best_pos, best_radii, deadline - 5)
            s = np.sum(gradii)
            if s > best_score:
                best_score = s
                best_pos = gpos.copy()
                best_radii = gradii.copy()
        
        # Try SLSQP from current best
        if time.time() < deadline - 10:
            res = _solve_slsqp_vectorized(best_pos, best_radii, deadline)
            if res:
                p, r = res
                s = np.sum(r)
                if s > best_score:
                    best_score = s
                    best_pos = p.copy()
                    best_radii = r.copy()
        
        # Random perturbation
        perturbed_pos = best_pos + rng.normal(0, 0.005, (N, 2))
        perturbed_pos = np.clip(perturbed_pos, 0.05, 0.95)
        perturbed_radii = best_radii + rng.normal(0, 0.002, N)
        perturbed_radii = np.clip(perturbed_radii, 0.005, 0.3)
        
        perturbed_pos, perturbed_radii = _post_process(perturbed_pos, perturbed_radii)
        
        res = _solve_slsqp_vectorized(perturbed_pos, perturbed_radii, deadline)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p.copy()
                best_radii = r.copy()
    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    best_pos = None
    best_radii = None
    best_score = -1

    # 1. Grid Init + SLSQP
    if time.time() < deadline - 15:
        pos, radii = _init_grid()
        res = _solve_slsqp_vectorized(pos, radii, deadline)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 2. Multiple Random Inits
    for seed in range(15):
        if time.time() > deadline - 15:
            break
        pos, radii = _init_random(seed)
        res = _solve_slsqp_vectorized(pos, radii, deadline)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 3. Local refinement on best solution
    if best_pos is not None and time.time() < deadline - 5:
        refined_pos, refined_radii = _local_refine(best_pos, best_radii, deadline)
        s = np.sum(refined_radii)
        if s > best_score:
            best_score = s
            best_pos = refined_pos
            best_radii = refined_radii

    if best_pos is None:
        pos, radii = _init_grid()
        pos, radii = _post_process(pos, radii)
        return np.column_stack([pos, radii])

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
