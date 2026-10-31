# EVOLVE-BLOCK-START

import numpy as np
import time
from scipy.optimize import minimize

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
    """Grid-based initialization."""
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

def _solve_socp(pos_init, radii_init, deadline):
    """
    Solve the circle packing problem using SOCP formulation.
    
    The problem is:
    maximize sum(r_i)
    subject to:
      ||p_i - p_j||_2 >= r_i + r_j for all i < j
      p_i in [r_i, 1-r_i]^2 for all i
      r_i >= 0 for all i
    
    This is a convex optimization problem that can be solved with SOCP solvers.
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None
    
    # Variables
    x = cp.Variable((N, 2))  # positions
    r = cp.Variable(N)       # radii
    
    # Objective: maximize sum of radii
    objective = cp.Maximize(cp.sum(r))
    
    # Constraints
    constraints = []
    
    # Non-negativity of radii
    constraints.append(r >= 0)
    
    # Boundary constraints: each circle must be inside [0,1]^2
    for i in range(N):
        constraints.append(x[i, 0] - r[i] >= 0)
        constraints.append(1 - x[i, 0] - r[i] >= 0)
        constraints.append(x[i, 1] - r[i] >= 0)
        constraints.append(1 - x[i, 1] - r[i] >= 0)
    
    # Overlap constraints: ||p_i - p_j||_2 >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append(cp.norm(x[i] - x[j], 2) >= r[i] + r[j])
    
    prob = cp.Problem(objective, constraints)
    
    # Set initial values for warm start
    try:
        x.value = pos_init
        r.value = radii_init
    except:
        pass
    
    # Try multiple solvers
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
    
    # Ensure radii are non-negative and reasonable
    radii = np.maximum(radii, 1e-6)
    radii = np.minimum(radii, 0.5)
    
    # Post-process to ensure validity
    pos, radii = _post_process(pos, radii)
    
    if not _is_valid(pos, radii):
        return None
    
    return pos, radii

def _local_refine_socp(pos, radii, deadline):
    """
    Local refinement using multiple SOCP solves with perturbed initializations.
    """
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    rng = np.random.RandomState(42)
    
    # Try different perturbation scales
    scales = [0.03, 0.02, 0.015, 0.01, 0.005]
    
    while time.time() < deadline - 5:
        for scale in scales:
            if time.time() > deadline - 5:
                break
            
            # Generate perturbed initialization
            perturbed_pos = best_pos + rng.normal(0, scale, (N, 2))
            perturbed_pos = np.clip(perturbed_pos, 0.02, 0.98)
            perturbed_radii = best_radii.copy()
            
            # Try SOCP solve
            res = _solve_socp(perturbed_pos, perturbed_radii, deadline)
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

    # Generate multiple initializations
    inits = []
    
    # Grid init
    pos, radii = _init_grid()
    inits.append((pos, radii))
    
    # Random inits
    for seed in range(10):
        pos, radii = _init_random(seed)
        inits.append((pos, radii))
    
    # Try SOCP with each initialization
    for pos, radii in inits:
        if time.time() > deadline - 15:
            break
        
        res = _solve_socp(pos, radii, deadline)
        if res:
            p, r = res
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_pos = p
                best_radii = r
    
    # If no SOCP solution found, fall back to SLSQP
    if best_pos is None:
        for pos, radii in inits[:3]:
            if time.time() > deadline - 10:
                break
            # Simple gradient-based refinement
            try:
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
                
                res = minimize(obj, x0, method='SLSQP', bounds=bounds, 
                               constraints=[{'type': 'ineq', 'fun': cons_fun}],
                               options={'maxiter': 300, 'ftol': 1e-12})
                p = res.x[:2*N].reshape(N, 2)
                r = res.x[2*N:]
                r = np.maximum(r, 1e-6)
                p, r = _post_process(p, r)
                if _is_valid(p, r):
                    s = np.sum(r)
                    if s > best_score:
                        best_score = s
                        best_pos = p
                        best_radii = r
            except Exception:
                pass
    
    # Local refinement on best solution
    if best_pos is not None and time.time() < deadline - 5:
        refined_pos, refined_radii = _local_refine_socp(best_pos, best_radii, deadline)
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
