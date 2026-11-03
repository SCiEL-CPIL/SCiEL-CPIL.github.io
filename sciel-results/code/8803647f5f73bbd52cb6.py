import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4.0

# Precompute indices for all triangles
_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)
N_TRI = len(_indices)

def _params_to_points(params):
    """
    Convert 22 parameters (r1, r2 for each point) to Cartesian coordinates.
    Mapping:
    lambda_2 = r1
    lambda_1 = r2 * (1 - r1)
    lambda_0 = (1 - r1) * (1 - r2)
    x = lambda_1 + 0.5 * lambda_2
    y = lambda_2 * S3H
    """
    params = np.clip(params, 0.0, 1.0)
    r1 = params[0::2]
    r2 = params[1::2]
    
    lam2 = r1
    lam1 = r2 * (1.0 - r1)
    # lam0 = (1 - r1) * (1 - r2) is implicit
    
    x = lam1 + 0.5 * lam2
    y = lam2 * S3H
    
    return np.column_stack([x, y])

def _compute_areas(points):
    """
    Compute all 165 triangle areas from 11 points using vectorized operations.
    """
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    
    xi, yi = points[i, 0], points[i, 1]
    xj, yj = points[j, 0], points[j, 1]
    xk, yk = points[k, 0], points[k, 1]
    
    # 2 * Area = | (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi) |
    # Or using determinant formula:
    # 2A = | xi(yj-yk) + xj(yk-yi) + xk(yi-yj) |
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    areas = 0.5 * np.abs(det)
    return areas

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _soft_min_area(points, beta):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    """
    areas = _compute_areas(points)
    # Numerical stability: shift by max area
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # Use log-sum-exp trick
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

def _objective(params, beta):
    """
    Objective function for optimization: negative of soft min area.
    """
    points = _params_to_points(params)
    return -_soft_min_area(points, beta)

def _objective_hard(params):
    """
    Objective function for final evaluation: negative of hard min area.
    """
    points = _params_to_points(params)
    return -_min_area(points)

def _active_set_search(params, time_limit=10.0, max_iter=200):
    """
    Local search that only moves points involved in the smallest triangles.
    Operates in parameter space to ensure validity.
    """
    start_t = time.time()
    current_params = params.copy()
    current_points = _params_to_points(current_params)
    current_min = _min_area(current_points)
    best_params = current_params.copy()
    best_min = current_min
    
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for step_size in step_sizes:
        if time.time() - start_t > time_limit:
            break
        
        improved_in_pass = True
        while improved_in_pass:
            if time.time() - start_t > time_limit:
                break
            
            improved_in_pass = False
            current_points = _params_to_points(current_params)
            areas = _compute_areas(current_points)
            min_area = float(np.min(areas))
            
            if min_area > best_min:
                best_min = min_area
                best_params = current_params.copy()
            
            # Identify indices of points involved in min triangles
            threshold = min_area + 1e-10
            min_tri_idx = np.where(areas <= threshold)[0]
            involved = set()
            for idx in min_tri_idx:
                involved.add(int(_idx_arr[idx, 0]))
                involved.add(int(_idx_arr[idx, 1]))
                involved.add(int(_idx_arr[idx, 2]))
            
            if not involved:
                break
                
            # Try to move each involved point
            for p_idx in involved:
                if time.time() - start_t > time_limit:
                    break
                
                # Try random perturbations in parameter space for this point
                for _ in range(20):
                    # Perturb r1 and r2 for point p_idx
                    # Indices in flat array: 2*p_idx and 2*p_idx+1
                    d_r1 = np.random.uniform(-step_size, step_size)
                    d_r2 = np.random.uniform(-step_size, step_size)
                    
                    new_params = current_params.copy()
                    new_params[2*p_idx] += d_r1
                    new_params[2*p_idx+1] += d_r2
                    
                    # Clip to [0,1]
                    new_params = np.clip(new_params, 0.0, 1.0)
                    
                    new_points = _params_to_points(new_params)
                    new_min = _min_area(new_points)
                    
                    if new_min > min_area + 1e-12:
                        current_params = new_params
                        improved_in_pass = True
                        break # Move to next point or restart loop
                        
                if improved_in_pass:
                    break # Restart while loop to re-evaluate min triangles

    return best_params, best_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    np.random.seed(42)
    
    best_params = None
    best_score = -np.inf
    
    def _update_best(params):
        nonlocal best_params, best_score
        points = _params_to_points(params)
        score = _min_area(points)
        if score > best_score:
            best_score = score
            best_params = params.copy()
            return True
        return False

    # Phase 1: Differential Evolution
    # Bounds for 22 parameters (r1, r2 for 11 points), each in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Use a moderate beta for DE to smooth the landscape
    beta_de = 10.0
    
    try:
        if time.time() < deadline - 60:
            de_result = differential_evolution(
                _objective,
                bounds,
                args=(beta_de,),
                maxiter=30,
                popsize=15,
                seed=42,
                tol=1e-4,
                polish=False, # We will do our own polishing
                init='latinhypercube'
            )
            _update_best(de_result.x)
    except Exception:
        pass

    # If DE failed or didn't find anything good, try a few random starts
    if best_params is None or best_score < 0.01:
        for _ in range(5):
            random_params = np.random.rand(2 * N)
            _update_best(random_params)

    # Phase 2: Local Optimization with Soft-Min Annealing
    if best_params is not None and time.time() < deadline - 30:
        current_params = best_params.copy()
        
        # Anneal beta from low to high
        beta_schedule = [10.0, 50.0, 100.0, 500.0, 1000.0]
        
        for beta in beta_schedule:
            if time.time() > deadline - 15:
                break
            
            # Use L-BFGS-B for local refinement
            try:
                res = minimize(
                    _objective,
                    current_params,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < _objective(current_params, beta):
                    current_params = res.x
                    _update_best(current_params)
            except Exception:
                pass

    # Phase 3: Active Set Search (High Precision Polish)
    if best_params is not None and time.time() < deadline - 5:
        # Run active set search from the best found so far
        # Also try from a few slightly perturbed versions of the best
        start_params_list = [best_params.copy()]
        
        # Add a few perturbations
        for _ in range(3):
            perturbed = best_params + np.random.normal(0, 0.01, size=2*N)
            perturbed = np.clip(perturbed, 0, 1)
            start_params_list.append(perturbed)
        
        for sp in start_params_list:
            if time.time() > deadline - 2:
                break
            
            try:
                refined_params, refined_score = _active_set_search(
                    sp, 
                    time_limit=min(5.0, deadline - time.time() - 1)
                )
                _update_best(refined_params)
            except Exception:
                pass

    # Fallback
    if best_params is None:
        best_params = np.random.rand(2 * N)
        
    return _params_to_points(best_params)
