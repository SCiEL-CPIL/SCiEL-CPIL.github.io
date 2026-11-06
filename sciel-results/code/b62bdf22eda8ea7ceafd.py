"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _mean_reflectance_vec(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band using vectorized numpy."""
    if len(materials) == 0:
        n0 = problem["ambient_index"]
        ns = problem["substrate_index"]
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r
    
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    mat_dict = problem["materials"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    
    # Initialize total matrix for all wavelengths
    # M is shape (401, 2, 2)
    M = np.zeros((len(wavelengths), 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for mat, thick in zip(materials, thicknesses):
        n = mat_dict[mat]
        d = thick
        if d == 0:
            continue
        beta = 2.0 * np.pi * n * d / wavelengths
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Layer matrix
        m00 = cos_b
        m01 = 1.0j * sin_b / n
        m10 = n * 1.0j * sin_b
        m11 = cos_b
        
        # Multiply M @ m
        # M_new[i, 0, 0] = M[i, 0, 0]*m00 + M[i, 0, 1]*m10
        # M_new[i, 0, 1] = M[i, 0, 0]*m01 + M[i, 0, 1]*m11
        # M_new[i, 1, 0] = M[i, 1, 0]*m00 + M[i, 1, 1]*m10
        # M_new[i, 1, 1] = M[i, 1, 0]*m01 + M[i, 1, 1]*m11
        
        M_new = np.empty_like(M)
        M_new[:, 0, 0] = M[:, 0, 0] * m00 + M[:, 0, 1] * m10
        M_new[:, 0, 1] = M[:, 0, 0] * m01 + M[:, 0, 1] * m11
        M_new[:, 1, 0] = M[:, 1, 0] * m00 + M[:, 1, 1] * m10
        M_new[:, 1, 1] = M[:, 1, 0] * m01 + M[:, 1, 1] * m11
        M = M_new
        
    # Substrate admittance
    Ys = ns
    
    # Total admittance Y = M[1, 1] / M[0, 1]
    Y = M[:, 1, 1] / M[:, 0, 1]
    
    # Reflectance
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r) ** 2 * 100.0
    
    return np.mean(R)

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    if sum(thicknesses) > max_total_nm:
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    return True

def _nelder_mead_optimize(materials, initial_thicks, problem, time_limit):
    """
    Perform Nelder-Mead optimization on layer thicknesses.
    Returns (best_thicks, best_score).
    """
    n_params = len(initial_thicks)
    if n_params == 0:
        return initial_thicks, _mean_reflectance_vec(materials, initial_thicks, problem)
        
    # Initial simplex
    simplex = []
    x0 = np.array(initial_thicks, dtype=float)
    simplex.append(x0)
    
    # Add perturbed points
    for i in range(n_params):
        xi = x0.copy()
        # Initial step size
        step = 10.0 
        # Ensure step doesn't violate bounds immediately if possible, 
        # but NM handles it via reflection/contraction. 
        # We will project back to valid range after each step.
        xi[i] += step
        simplex.append(xi)
        
    simplex = np.array(simplex)
    
    # Evaluate initial simplex
    f_values = np.zeros(n_params + 1)
    for i in range(n_params + 1):
        # Project to valid range
        x = np.clip(simplex[i], 0.0, problem["max_layer_nm"])
        # Check total thickness constraint
        if np.sum(x) > problem["max_total_nm"]:
            # Scale down
            x = x * (problem["max_total_nm"] / np.sum(x))
        simplex[i] = x
        f_values[i] = _mean_reflectance_vec(materials, simplex[i].tolist(), problem)
        
    best_score = np.min(f_values)
    best_thicks = simplex[np.argmin(f_values)].tolist()
    
    alpha = 1.0  # Reflection
    gamma = 2.0  # Expansion
    rho = 0.5   # Contraction
    sigma = 0.5 # Shrinkage
    
    max_iter = 200 # Safety cap on iterations
    
    for _ in range(max_iter):
        if time.time() > time_limit:
            break
            
        # Sort simplex
        idx = np.argsort(f_values)
        simplex = simplex[idx]
        f_values = f_values[idx]
        
        # Check convergence
        if np.abs(f_values[0] - f_values[-1]) < 1e-6:
            break
            
        x_worst = simplex[-1]
        x_best = simplex[0]
        
        # Centroid of all but worst
        x_c = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = x_c + alpha * (x_c - x_worst)
        x_r = np.clip(x_r, 0.0, problem["max_layer_nm"])
        if np.sum(x_r) > problem["max_total_nm"]:
            x_r = x_r * (problem["max_total_nm"] / np.sum(x_r))
        f_r = _mean_reflectance_vec(materials, x_r.tolist(), problem)
        
        if f_r < f_values[0]:
            # Expansion
            x_e = x_c + gamma * (x_r - x_c)
            x_e = np.clip(x_e, 0.0, problem["max_layer_nm"])
            if np.sum(x_e) > problem["max_total_nm"]:
                x_e = x_e * (problem["max_total_nm"] / np.sum(x_e))
            f_e = _mean_reflectance_vec(materials, x_e.tolist(), problem)
            
            if f_e < f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
        elif f_r < f_values[-2]:
            # Accept reflection
            simplex[-1] = x_r
            f_values[-1] = f_r
        else:
            # Contraction
            if f_r < f_values[-1]:
                # Outside contraction
                x_c_new = x_c + rho * (x_r - x_c)
            else:
                # Inside contraction
                x_c_new = x_c - rho * (x_c - x_r)
                
            x_c_new = np.clip(x_c_new, 0.0, problem["max_layer_nm"])
            if np.sum(x_c_new) > problem["max_total_nm"]:
                x_c_new = x_c_new * (problem["max_total_nm"] / np.sum(x_c_new))
            f_c = _mean_reflectance_vec(materials, x_c_new.tolist(), problem)
            
            if f_c < min(f_r, f_values[-1]):
                simplex[-1] = x_c_new
                f_values[-1] = f_c
            else:
                # Shrinkage
                for i in range(1, n_params + 1):
                    simplex[i] = x_best + sigma * (simplex[i] - x_best)
                    simplex[i] = np.clip(simplex[i], 0.0, problem["max_layer_nm"])
                    if np.sum(simplex[i]) > problem["max_total_nm"]:
                        simplex[i] = simplex[i] * (problem["max_total_nm"] / np.sum(simplex[i]))
                    f_values[i] = _mean_reflectance_vec(materials, simplex[i].tolist(), problem)
                    
        # Update best
        current_best_idx = np.argmin(f_values)
        if f_values[current_best_idx] < best_score:
            best_score = f_values[current_best_idx]
            best_thicks = simplex[current_best_idx].tolist()
            
    return best_thicks, best_score

def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    
    dL = centre / (4.0 * nL)
    dH = centre / (4.0 * nH)
    
    # Generate candidate structures
    candidates = []
    
    # 1-layer L
    candidates.append((["L"], [dL]))
    
    # 3-layer L-H-L
    candidates.append((["L", "H", "L"], [dL, dH, dL]))
    
    # 5-layer L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L"], [dL, dH, dL, dH, dL]))
    
    # 7-layer L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L"], [dL, dH, dL, dH, dL, dH, dL]))
    
    # 9-layer L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 11-layer L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 13-layer L-H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 15-layer L-H-L-H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 17-layer L-H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # 19-layer L-H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 
                       [dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL, dH, dL]))
    
    # Evaluate initial candidates
    scored_candidates = []
    for mats, thicks in candidates:
        if not _is_valid(mats, thicks, problem):
            continue
        score = _mean_reflectance_vec(mats, thicks, problem)
        scored_candidates.append((score, mats, thicks))
        
    if not scored_candidates:
        # Fallback: empty stack or single layer
        return (["L"], [dL])
        
    # Sort by score
    scored_candidates.sort(key=lambda x: x[0])
    
    best_score = scored_candidates[0][0]
    best_result = (scored_candidates[0][1], scored_candidates[0][2])
    
    # Optimize the top 3 candidates
    # Allocate time for optimization
    time_per_opt = 15.0 # 15s each for top 3 = 45s total. Start time + 45 < 55 deadline.
    
    for i in range(min(3, len(scored_candidates))):
        if time.time() > deadline - time_per_opt:
            break
        score, mats, thicks = scored_candidates[i]
        if score > best_score * 1.1: # Heuristic: skip if much worse than current best
             continue
             
        opt_thicks, opt_score = _nelder_mead_optimize(mats, thicks, problem, time.time() + time_per_opt)
        
        if opt_score < best_score:
            best_score = opt_score
            best_result = (mats, opt_thicks)
            
    return best_result
# EVOLVE-BLOCK-END
