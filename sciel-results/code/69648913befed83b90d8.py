# EVOLVE-BLOCK-START
import time
import numpy as np

def _reflectance_spectrum(n_vals, d_vals, lambdas, n_ambient, n_substrate):
    """
    Vectorized reflectance computation for a stack of layers.
    n_vals: array of refractive indices (length N)
    d_vals: array of thicknesses in nm (length N)
    lambdas: array of wavelengths in nm (length W)
    Returns: array of reflectance (0 to 1) for each wavelength.
    """
    N = len(n_vals)
    W = len(lambdas)
    
    # Initialize total matrix as identity for each wavelength
    # M_total[w] = [[1, 0], [0, 1]]
    M11 = np.ones(W, dtype=complex)
    M12 = np.zeros(W, dtype=complex)
    M21 = np.zeros(W, dtype=complex)
    M22 = np.ones(W, dtype=complex)
    
    for j in range(N):
        n = n_vals[j]
        d = d_vals[j]
        # beta = 2*pi*n*d / lambda for each wavelength
        beta = 2.0 * np.pi * n * d / lambdas
        c = np.cos(beta)
        s = np.sin(beta)
        
        # Layer matrix: [[c, i*s/n], [i*n*s, c]]
        L11 = c
        L12 = 1j * s / n
        L21 = 1j * n * s
        L22 = c
        
        # M_total = M_total @ M_layer
        # New M11 = M11*L11 + M12*L21
        # New M12 = M11*L12 + M12*L22
        # New M21 = M21*L11 + M22*L21
        # New M22 = M21*L12 + M22*L22
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
    
    # Admittance Y = (M11 + M12 * n_sub) / (M21 + M22 * n_sub)
    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    
    # Reflection coefficient r = (n_ambient - Y) / (n_ambient + Y)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    
    return R

def _evaluate(n_vals, d_vals, lambdas, n_ambient, n_substrate):
    """Compute mean reflectance in percent."""
    R = _reflectance_spectrum(n_vals, d_vals, lambdas, n_ambient, n_substrate)
    return 100.0 * np.mean(R)

def _is_valid(d_vals, max_total_nm, max_layer_nm):
    """Check thickness constraints."""
    if len(d_vals) == 0:
        return True
    total = sum(d_vals)
    if total > max_total_nm + 1e-6:
        return False
    for t in d_vals:
        if t < -1e-6 or t > max_layer_nm + 1e-6:
            return False
    return True

def _n_from_materials(materials, n_L, n_H):
    """Convert material labels to refractive index array."""
    return np.array([n_L if m == 'L' else n_H for m in materials])

def _nelder_mead_optimize(materials, initial_thicks, problem, time_limit):
    """
    Perform Nelder-Mead optimization on layer thicknesses.
    Returns (best_thicks, best_score).
    """
    n_params = len(initial_thicks)
    if n_params == 0:
        return initial_thicks, _evaluate(_n_from_materials(materials, problem["materials"]["L"], problem["materials"]["H"]),
                                         np.array(initial_thicks, dtype=float),
                                         np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401),
                                         problem["ambient_index"], problem["substrate_index"])
        
    # Initial simplex
    simplex = []
    x0 = np.array(initial_thicks, dtype=float)
    # Project to feasible region
    x0 = np.clip(x0, 0.0, problem["max_layer_nm"])
    total = np.sum(x0)
    if total > problem["max_total_nm"]:
        x0 = x0 * (problem["max_total_nm"] / total)
    simplex.append(x0)
    
    # Add perturbed points
    step = 5.0 
    for i in range(n_params):
        xi = x0.copy()
        xi[i] += step
        # Project
        xi = np.clip(xi, 0.0, problem["max_layer_nm"])
        total = np.sum(xi)
        if total > problem["max_total_nm"]:
            xi = xi * (problem["max_total_nm"] / total)
        simplex.append(xi)
        
    simplex = np.array(simplex)
    
    # Pre-compute for evaluation
    n_vals = _n_from_materials(materials, problem["materials"]["L"], problem["materials"]["H"])
    lambdas = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    
    def eval_vec(x):
        # x is 1D array of thicknesses
        if not _is_valid(x, problem["max_total_nm"], problem["max_layer_nm"]):
            return float('inf')
        return _evaluate(n_vals, x, lambdas, n_ambient, n_substrate)

    # Evaluate initial simplex
    f_values = np.zeros(n_params + 1)
    for i in range(n_params + 1):
        f_values[i] = eval_vec(simplex[i])
        
    best_score = np.min(f_values)
    best_thicks = simplex[np.argmin(f_values)].tolist()
    
    alpha = 1.0  # Reflection
    gamma = 2.0  # Expansion
    rho = 0.5   # Contraction
    sigma = 0.5 # Shrinkage
    
    max_iter = 500
    
    for _ in range(max_iter):
        if time.time() > time_limit:
            break
            
        # Sort simplex
        idx = np.argsort(f_values)
        simplex = simplex[idx]
        f_values = f_values[idx]
        
        # Check convergence
        if np.abs(f_values[0] - f_values[-1]) < 1e-8:
            break
        # Also check if the simplex has collapsed in size
        if np.max(np.std(simplex, axis=0)) < 1e-6:
             break

        x_worst = simplex[-1]
        x_best = simplex[0]
        
        # Centroid of all but worst
        x_c = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = x_c + alpha * (x_c - x_worst)
        x_r = np.clip(x_r, 0.0, problem["max_layer_nm"])
        total = np.sum(x_r)
        if total > problem["max_total_nm"]:
            x_r = x_r * (problem["max_total_nm"] / total)
        f_r = eval_vec(x_r)
        
        if f_r < f_values[0]:
            # Expansion
            x_e = x_c + gamma * (x_r - x_c)
            x_e = np.clip(x_e, 0.0, problem["max_layer_nm"])
            total = np.sum(x_e)
            if total > problem["max_total_nm"]:
                x_e = x_e * (problem["max_total_nm"] / total)
            f_e = eval_vec(x_e)
            
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
            total = np.sum(x_c_new)
            if total > problem["max_total_nm"]:
                x_c_new = x_c_new * (problem["max_total_nm"] / total)
            f_c = eval_vec(x_c_new)
            
            if f_c < min(f_r, f_values[-1]):
                simplex[-1] = x_c_new
                f_values[-1] = f_c
            else:
                # Shrinkage
                for i in range(1, n_params + 1):
                    simplex[i] = x_best + sigma * (simplex[i] - x_best)
                    simplex[i] = np.clip(simplex[i], 0.0, problem["max_layer_nm"])
                    total = np.sum(simplex[i])
                    if total > problem["max_total_nm"]:
                        simplex[i] = simplex[i] * (problem["max_total_nm"] / total)
                    f_values[i] = eval_vec(simplex[i])
                    
        # Update best
        current_best_idx = np.argmin(f_values)
        if f_values[current_best_idx] < best_score:
            best_score = f_values[current_best_idx]
            best_thicks = simplex[current_best_idx].tolist()
            
    return best_thicks, best_score

def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lambdas = np.linspace(band[0], band[1], 401)
    center_lam = 0.5 * (band[0] + band[1])
    
    def time_ok():
        return time.time() < deadline - 2.0
    
    def eval_design(materials, thicknesses):
        """Evaluate a design, return mean reflectance in percent or inf if invalid."""
        if len(materials) > max_layers:
            return float('inf')
        d = np.array(thicknesses, dtype=float)
        if not _is_valid(d, max_total_nm, max_layer_nm):
            return float('inf')
        n_vals = _n_from_materials(materials, n_L, n_H)
        return _evaluate(n_vals, d, lambdas, n_ambient, n_substrate)
    
    # Generate initial candidate designs
    candidates = []
    
    # Quarter-wave stacks at center wavelength
    # Focus on odd number of layers starting and ending with L (standard for AR on glass)
    for n_layers in range(1, 12, 2):
        if n_layers > max_layers:
            break
        materials = []
        thicknesses = []
        for i in range(n_layers):
            if (i % 2 == 0):
                mat = 'L'
            else:
                mat = 'H'
            materials.append(mat)
            n = n_L if mat == 'L' else n_H
            thicknesses.append(center_lam / (4.0 * n))
        candidates.append((materials[:], thicknesses[:]))
        
        # Also try starting with H? Less common for AR but possible
        if n_layers > 1:
            materials_h = []
            thicknesses_h = []
            for i in range(n_layers):
                if (i % 2 == 0):
                    mat = 'H'
                else:
                    mat = 'L'
                materials_h.append(mat)
                n = n_L if mat == 'L' else n_H
                thicknesses_h.append(center_lam / (4.0 * n))
            candidates.append((materials_h[:], thicknesses_h[:]))
            
    # Evaluate all candidates and find the best
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    scored_candidates = []
    for mats, thicks in candidates:
        if not time_ok():
            break
        score = eval_design(mats, thicks)
        if score < best_score:
            best_score = score
            best_materials = mats[:]
            best_thicknesses = thicks[:]
        scored_candidates.append((score, mats, thicks))
        
    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [center_lam / (4.0 * n_L)]
        best_score = eval_design(best_materials, best_thicknesses)
        scored_candidates = [(best_score, best_materials[:], best_thicknesses[:])]
        
    # Sort candidates by score
    scored_candidates.sort(key=lambda x: x[0])
    
    # Optimize the top candidates
    # We will try to optimize the top 5 if time permits, giving more time to better candidates.
    # Total budget is ~140s. Evaluation of initial candidates is fast.
    # Let's allocate time based on rank.
    
    num_to_optimize = min(5, len(scored_candidates))
    for i in range(num_to_optimize):
        if time.time() > deadline - 5.0: # Need at least 5s left for final check
            break
            
        score, mats, thicks = scored_candidates[i]
        
        # Skip if much worse than current best
        if score > best_score * 1.5:
            continue
            
        # Allocate time: Top candidate gets more time.
        # If 5 candidates, maybe 30, 20, 15, 10, 5 seconds?
        # Let's just use remaining time / remaining candidates, with a min of 2s.
        remaining_candidates = num_to_optimize - i
        time_alloc = max(2.0, (deadline - time.time()) / remaining_candidates)
        
        opt_thicks, opt_score = _nelder_mead_optimize(mats, thicks, problem, time.time() + time_alloc)
        
        if opt_score < best_score:
            best_score = opt_score
            best_materials = mats
            best_thicknesses = opt_thicks
            
    # Final local search refinement on the best result found
    if time.time() < deadline - 10.0 and best_materials is not None:
        # Try a few random perturbations around the best
        current_thicks = np.array(best_thicknesses, dtype=float)
        current_score = best_score
        n_layers = len(best_materials)
        
        # Simple coordinate descent with small steps
        for step in [10.0, 5.0, 1.0]:
            if time.time() > deadline - 5.0:
                break
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for i in range(n_layers):
                    for delta in [-step, step]:
                        new_thicks = current_thicks.copy()
                        new_thicks[i] += delta
                        if new_thicks[i] < 0: new_thicks[i] = 0
                        if new_thicks[i] > max_layer_nm: new_thicks[i] = max_layer_nm
                        
                        if not _is_valid(new_thicks, max_total_nm, max_layer_nm):
                            continue
                        
                        n_vals = _n_from_materials(best_materials, n_L, n_H)
                        score = _evaluate(n_vals, new_thicks, lambdas, n_ambient, n_substrate)
                        
                        if score < current_score - 1e-10:
                            current_thicks = new_thicks
                            current_score = score
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_thicknesses = current_thicks.tolist()
                                
    return (best_materials, best_thicknesses)
# EVOLVE-BLOCK-END
