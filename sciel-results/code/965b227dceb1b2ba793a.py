# EVOLVE-BLOCK-START

import numpy as np
import time
from scipy.optimize import minimize

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
    
    for _ in range(300):
        valid = True
        
        # Boundary constraints
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.99999, 1e-6)
                valid = False
        
        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    if sum_r > 1e-10:
                        k = dist / sum_r * 0.99999
                        radii[i] = max(radii[i] * k, 1e-6)
                        radii[j] = max(radii[j] * k, 1e-6)
                        valid = False
                    else:
                        radii[i] = max(radii[i] * 0.99, 1e-6)
                        radii[j] = max(radii[j] * 0.99, 1e-6)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _compute_overlap_gradients(pos, radii):
    """Compute gradients of overlap violations for gradient-based optimization."""
    n = len(radii)
    grad_pos = np.zeros_like(pos)
    grad_radii = np.zeros_like(radii)
    
    # Boundary gradients
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if x - r < 0:
            grad_pos[i, 0] += 1
            grad_radii[i] -= 1
        if 1 - x - r < 0:
            grad_pos[i, 0] -= 1
            grad_radii[i] -= 1
        if y - r < 0:
            grad_pos[i, 1] += 1
            grad_radii[i] -= 1
        if 1 - y - r < 0:
            grad_pos[i, 1] -= 1
            grad_radii[i] -= 1
    
    # Overlap gradients
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            sum_r = radii[i] + radii[j]
            if dist < sum_r and dist > 1e-12:
                gx = dx / dist
                gy = dy / dist
                grad_pos[i, 0] += gx
                grad_pos[i, 1] += gy
                grad_pos[j, 0] -= gx
                grad_pos[j, 1] -= gy
                grad_radii[i] -= 1
                grad_radii[j] -= 1
    
    return grad_pos, grad_radii

def _gradient_descent(pos, radii, deadline, lr=0.01, n_iter=2000):
    """Gradient-based optimization with adaptive learning rate."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
            
        grad_pos, grad_radii = _compute_overlap_gradients(pos, radii)
        
        pos_new = pos + lr * grad_pos
        radii_new = radii + lr * grad_radii
        
        pos_new = np.clip(pos_new, 0.01, 0.99)
        radii_new = np.clip(radii_new, 0.01, 0.25)
        
        pos_new, radii_new = _post_process(pos_new, radii_new)
        score = np.sum(radii_new)
        
        if score > best_score:
            best_score = score
            best_pos = pos_new.copy()
            best_radii = radii_new.copy()
            pos = pos_new
            radii = radii_new
        else:
            lr *= 0.95
    
    return best_pos, best_radii

def _solve_slsqp(pos, radii, deadline):
    """Solve using SLSQP."""
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = [(0.01, 0.99)] * (2 * N) + [(0.01, 0.25)] * N
    
    def obj(x):
        return -np.sum(x[2*N:])
    
    def cons_fun(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        c_bound = np.concatenate([
            p[:, 0] - r,
            1.0 - p[:, 0] - r,
            p[:, 1] - r,
            1.0 - p[:, 1] - r
        ])
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        i_idx, j_idx = np.where(mask)
        dx = p[i_idx, 0] - p[j_idx, 0]
        dy = p[i_idx, 1] - p[j_idx, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        c_overlap = dist - (r[i_idx] + r[j_idx])
        
        return np.concatenate([c_bound, c_overlap])
    
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, 
                       constraints=[{'type': 'ineq', 'fun': cons_fun}],
                       options={'maxiter': 200, 'ftol': 1e-12})
        if res.success or res.fun < 0:
            p = res.x[:2*N].reshape(N, 2)
            r = res.x[2*N:]
            r = np.maximum(r, 1e-6)
            p, r = _post_process(p, r)
            if _is_valid(p, r):
                return p, r
    except Exception:
        pass
    return None

def _init_grid():
    """Grid-based initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx >= N:
                break
            x = (j + 0.5) / 5.0
            y = (i + 0.5) / 5.0
            pos[idx] = [x, y]
            radii[idx] = 0.09
    
    rng = np.random.RandomState(42)
    pos += rng.normal(0, 0.01, (N, 2))
    pos = np.clip(pos, 0.05, 0.95)
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _init_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.07, 0.11, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _local_refine(pos, radii, deadline):
    """Iterative local refinement with multiple strategies."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(123)
    
    # Strategy 1: SLSQP from perturbed starts
    for _ in range(20):
        if time.time() > deadline - 3:
            break
        
        perturbed_pos = best_pos + rng.normal(0, 0.015, (N, 2))
        perturbed_pos = np.clip(perturbed_pos, 0.03, 0.97)
        perturbed_radii = best_radii * (1 + rng.normal(0, 0.05, N))
        perturbed_radii = np.clip(perturbed_radii, 0.02, 0.20)
        
        perturbed_pos, perturbed_radii = _post_process(perturbed_pos, perturbed_radii)
        
        res = _solve_slsqp(perturbed_pos, perturbed_radii, deadline)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p.copy()
                best_radii = r.copy()
    
    # Strategy 2: Gradient descent refinement
    if time.time() < deadline - 5:
        gd_pos, gd_radii = _gradient_descent(best_pos, best_radii, deadline)
        s = np.sum(gd_radii)
        if s > best_score:
            best_score = s
            best_pos = gd_pos.copy()
            best_radii = gd_radii.copy()
    
    # Strategy 3: Larger perturbations
    for _ in range(10):
        if time.time() > deadline - 2:
            break
        
        perturbed_pos = best_pos + rng.normal(0, 0.03, (N, 2))
        perturbed_pos = np.clip(perturbed_pos, 0.05, 0.95)
        perturbed_radii = best_radii * (1 + rng.normal(0, 0.1, N))
        perturbed_radii = np.clip(perturbed_radii, 0.03, 0.18)
        
        perturbed_pos, perturbed_radii = _post_process(perturbed_pos, perturbed_radii)
        
        res = _solve_slsqp(perturbed_pos, perturbed_radii, deadline)
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
    deadline = start_time + 115.0

    best_pos = None
    best_radii = None
    best_score = -1

    # 1. Try structured initialization with SLSQP
    pos, radii = _init_grid()
    res = _solve_slsqp(pos, radii, deadline)
    if res:
        p, r = res
        s = np.sum(r)
        if s > best_score:
            best_score = s
            best_pos = p
            best_radii = r

    # 2. Try multiple random initializations
    for seed in range(15):
        if time.time() > deadline - 20:
            break
        pos, radii = _init_random(seed)
        res = _solve_slsqp(pos, radii, deadline)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 3. Local refinement on best solution
    if best_pos is not None and time.time() < deadline - 10:
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
