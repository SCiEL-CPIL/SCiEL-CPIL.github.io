# EVOLVE-BLOCK-START

import numpy as np
import time

N = 26

def _compute_score(pos, radii):
    """Compute the sum of radii, the objective to maximize."""
    return np.sum(radii)

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
    
    for _ in range(500):
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
                    # Scale both radii proportionally
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

def _init_grid():
    """Grid-based initialization with better spacing."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    k = 0
    for i in range(5):
        for j in range(5):
            if k >= N:
                break
            pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
            radii[k] = 0.09
            k += 1
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _init_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.05, 0.11, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _init_hexagonal():
    """Hexagonal packing initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    dx = 0.2
    dy = dx * np.sqrt(3) / 2
    
    k = 0
    for row in range(5):
        for col in range(5):
            if k >= N:
                break
            x = 0.1 + col * dx + (row % 2) * dx / 2
            y = 0.1 + row * dy
            if x < 0.9 and y < 0.9:
                pos[k] = [x, y]
                radii[k] = 0.09
                k += 1
    
    rng = np.random.RandomState(123)
    while k < N:
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        too_close = False
        for i in range(k):
            d = np.sqrt((x - pos[i, 0])**2 + (y - pos[i, 1])**2)
            if d < 0.18:
                too_close = True
                break
        if not too_close:
            pos[k] = [x, y]
            radii[k] = 0.08
            k += 1
        else:
            pos[k] = [x, y]
            radii[k] = 0.06
            k += 1
    
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _solve_cvxpy(pos_init, radii_init, deadline):
    """Solve using CVXPY SOCP formulation - primary solver for global optimality."""
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
    
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append(cp.norm(x[i] - x[j], 2) >= r[i] + r[j])
            
    prob = cp.Problem(objective, constraints)
    
    try:
        x.value = pos_init
        r.value = radii_init
    except:
        pass
    
    solvers_to_try = []
    try:
        solvers_to_try.append(cp.CLARABEL)
    except:
        pass
    try:
        solvers_to_try.append(cp.SCS)
    except:
        pass
    try:
        solvers_to_try.append(cp.ECOS)
    except:
        pass
    try:
        solvers_to_try.append(cp.MOSEK)
    except:
        pass
        
    success = False
    for solver in solvers_to_try:
        try:
            remaining_time = deadline - time.time()
            if remaining_time < 5:
                break
            prob.solve(solver=solver, warm_start=True, verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                success = True
                break
        except Exception:
            continue
        
    if not success:
        return None
        
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return None
        
    pos = np.array(x.value)
    radii = np.array(r.value)
    radii = np.maximum(radii, 1e-6)
    
    pos, radii = _post_process(pos, radii)
    if not _is_valid(pos, radii):
        return None
        
    return pos, radii

def _solve_slsqp_vectorized(pos, radii, deadline):
    """Solve using SLSQP with vectorized constraints as fallback."""
    from scipy.optimize import minimize
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = [(0.01, 0.99)] * (2 * N) + [(0.01, 0.25)] * N
    
    def obj(x):
        return -np.sum(x[2*N:])
        
    def cons_fun(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        c_bound = np.concatenate([
            p[:, 0] - r - 1e-8,
            1.0 - p[:, 0] - r - 1e-8,
            p[:, 1] - r - 1e-8,
            1.0 - p[:, 1] - r - 1e-8
        ])
        
        dx = p[:, 0, np.newaxis] - p[np.newaxis, :, 0]
        dy = p[:, 1, np.newaxis] - p[np.newaxis, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        c_overlap = (dist - min_dist - 1e-8)[mask]
        
        return np.concatenate([c_bound, c_overlap])
        
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, 
                       constraints=[{'type': 'ineq', 'fun': cons_fun}],
                       options={'maxiter': 300, 'ftol': 1e-12})
        p = res.x[:2*N].reshape(N, 2)
        r = res.x[2*N:]
        r = np.maximum(r, 1e-6)
        p, r = _post_process(p, r)
        if _is_valid(p, r):
            return p, r
    except Exception:
        pass
    return None

def _local_refine(pos, radii, deadline):
    """Iterative local refinement with random perturbations."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(42)
    scales = [0.015, 0.01, 0.005, 0.002]
    
    while time.time() < deadline - 5:
        for scale in scales:
            if time.time() > deadline - 5:
                break
            
            perturbed_pos = best_pos + rng.normal(0, scale, (N, 2))
            perturbed_pos = np.clip(perturbed_pos, 0.02, 0.98)
            perturbed_radii = best_radii.copy()
            
            perturbed_pos, perturbed_radii = _post_process(perturbed_pos, perturbed_radii)
            
            # Try CVXPY first for better local refinement
            res = _solve_cvxpy(perturbed_pos, perturbed_radii, deadline)
            if res is None:
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
    deadline = start_time + 115.0

    best_pos = None
    best_radii = None
    best_score = -1

    # 1. Generate diverse initializations
    inits = []
    
    pos, radii = _init_grid()
    inits.append((pos, radii))
    
    pos, radii = _init_hexagonal()
    inits.append((pos, radii))
    
    for seed in range(8):
        pos, radii = _init_random(seed)
        inits.append((pos, radii))
    
    # 2. Solve each initialization with CVXPY (primary) or SLSQP (fallback)
    for pos, radii in inits:
        if time.time() > deadline - 15:
            break
        
        # Try CVXPY first
        res = _solve_cvxpy(pos, radii, deadline)
        if res is None:
            res = _solve_slsqp_vectorized(pos, radii, deadline)
        
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 3. Additional random inits if time permits
    for seed in range(8, 20):
        if time.time() > deadline - 10:
            break
        pos, radii = _init_random(seed)
        res = _solve_cvxpy(pos, radii, deadline)
        if res is None:
            res = _solve_slsqp_vectorized(pos, radii, deadline)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r

    # 4. Local refinement on best solution
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
