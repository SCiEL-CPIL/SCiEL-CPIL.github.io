"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band using characteristic matrix method (vectorized)."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wls = np.linspace(band[0], band[1], 401)
    
    n_layers = len(materials)
    if n_layers == 0:
        # Bare substrate reflectance
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return abs(r)**2 * 100.0
    
    n_mat = np.array([n_L if m == 'L' else n_H for m in materials])
    d = np.array(thicknesses, dtype=float)
    
    # Vectorized matrix multiplication
    # M is 2x2, stored as 4 arrays of shape (401,)
    M00 = np.ones(401, dtype=complex)
    M01 = np.zeros(401, dtype=complex)
    M10 = np.zeros(401, dtype=complex)
    M11 = np.ones(401, dtype=complex)
    
    for i in range(n_layers):
        n_i = n_mat[i]
        d_i = d[i]
        if d_i == 0:
            continue
        phase = 2.0 * np.pi * n_i * d_i / wls
        cos_p = np.cos(phase)
        sin_p = np.sin(phase)
        
        # B matrix elements
        B00 = cos_p
        B01 = 1j * sin_p / n_i
        B10 = 1j * n_i * sin_p
        B11 = cos_p
        
        # M = M @ B
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y = M01 / M11
    r = (n_amb - Y) / (n_amb + Y)
    R = np.abs(r)**2 * 100.0
    return float(np.mean(R))

def _fast_evaluate(mats, thick, problem, t0, deadline):
    """Evaluate a design with time check."""
    if time.time() > deadline:
        return float('inf')
    if not mats or not thick or len(mats) != len(thick):
        return float('inf')
    if len(mats) > problem["max_layers"]:
        return float('inf')
    if any(t < -1e-9 or t > problem["max_layer_nm"] + 1e-9 for t in thick):
        return float('inf')
    if sum(thick) > problem["max_total_nm"] + 1e-9:
        return float('inf')
    return _mean_reflectance(mats, [max(0, t) for t in thick], problem)

def _coordinate_descent(mats, x0, problem, t0, deadline, max_iter=50):
    """Coordinate descent optimization for thicknesses."""
    n_layers = len(mats)
    x = np.array(x0, dtype=float)
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    current_obj = _mean_reflectance(mats, list(x), problem)
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(n_layers):
            if time.time() > deadline - 5:
                break
            
            # Try a range of values for coordinate i
            best_val = x[i]
            best_obj = current_obj
            
            # Golden section search or simple grid
            # Use a coarse grid first, then refine
            step_coarse = max_layer_nm / 20
            for k in range(21):
                trial_val = k * step_coarse
                # Check if changing this coordinate violates constraints
                trial_x = x.copy()
                trial_x[i] = trial_val
                if np.sum(trial_x) > max_total_nm + 1e-9:
                    continue
                if trial_val < 0 or trial_val > max_layer_nm:
                    continue
                
                obj = _mean_reflectance(mats, list(trial_x), problem)
                if obj < best_obj:
                    best_obj = obj
                    best_val = trial_val
            
            # Refine around best_val
            if best_val > 0:
                step_fine = max_layer_nm / 200
                for k in range(-10, 11):
                    trial_val = best_val + k * step_fine
                    if trial_val < 0 or trial_val > max_layer_nm:
                        continue
                    trial_x = x.copy()
                    trial_x[i] = trial_val
                    if np.sum(trial_x) > max_total_nm + 1e-9:
                        continue
                    
                    obj = _mean_reflectance(mats, list(trial_x), problem)
                    if obj < best_obj:
                        best_obj = obj
                        best_val = trial_val
            
            if best_val != x[i]:
                x[i] = best_val
                current_obj = best_obj
                improved = True
        
        if not improved:
            break
    
    x = np.clip(x, 0, max_layer_nm)
    if np.sum(x) > max_total_nm:
        scale = max_total_nm / np.sum(x)
        x = x * scale
    
    final_obj = _mean_reflectance(mats, list(x), problem)
    return list(x), final_obj

def _nelder_mead_optimize(mats, x0, problem, t0, deadline):
    """Nelder-Mead optimization with time constraints."""
    try:
        from scipy.optimize import minimize
        
        n_layers = len(mats)
        max_layer_nm = problem["max_layer_nm"]
        max_total_nm = problem["max_total_nm"]
        
        def obj(x):
            if time.time() > deadline - 5:
                return 1000.0
            if np.any(x < -1e-9) or np.any(x > max_layer_nm + 1e-9):
                return 1000.0
            if np.sum(x) > max_total_nm + 1e-9:
                return 1000.0 + (np.sum(x) - max_total_nm) * 10
            return _mean_reflectance(mats, list(x), problem)
        
        res = minimize(obj, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-8,
                               'initial_simplex': None})
        
        if res.fun < 100.0:
            x_opt = np.clip(res.x, 0, max_layer_nm)
            if np.sum(x_opt) > max_total_nm:
                scale = max_total_nm / np.sum(x_opt)
                x_opt = x_opt * scale
            score = _mean_reflectance(mats, list(x_opt), problem)
            return list(x_opt), score
    except Exception:
        pass
    
    return None, float('inf')

def _l_bfgs_b_optimize(mats, x0, problem, t0, deadline):
    """L-BFGS-B optimization with time constraints."""
    try:
        from scipy.optimize import minimize
        
        n_layers = len(mats)
        max_layer_nm = problem["max_layer_nm"]
        max_total_nm = problem["max_total_nm"]
        
        def obj(x):
            if time.time() > deadline - 5:
                return 1000.0
            if np.any(x < -1e-9) or np.any(x > max_layer_nm + 1e-9):
                return 1000.0
            if np.sum(x) > max_total_nm + 1e-9:
                return 1000.0 + (np.sum(x) - max_total_nm) * 10
            return _mean_reflectance(mats, list(x), problem)
        
        res = minimize(obj, x0, method='L-BFGS-B',
                      bounds=[(0, max_layer_nm)] * n_layers,
                      options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.fun < 100.0:
            x_opt = np.clip(res.x, 0, max_layer_nm)
            if np.sum(x_opt) > max_total_nm:
                scale = max_total_nm / np.sum(x_opt)
                x_opt = x_opt * scale
            score = _mean_reflectance(mats, list(x_opt), problem)
            return list(x_opt), score
    except Exception:
        pass
    
    return None, float('inf')

def _generate_structures(problem, max_layers):
    """Generate diverse candidate material sequences."""
    structures = []
    
    # Quarter-wave alternating stacks
    for n in range(1, max_layers + 1):
        # Start with L
        mats_L = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        structures.append(mats_L)
        
        # Start with H
        mats_H = ['H' if i % 2 == 0 else 'L' for i in range(n)]
        structures.append(mats_H)
    
    # Double layers (pairs)
    for n_pairs in range(1, max_layers // 2 + 1):
        for pair in ['LH', 'HL']:
            mats = []
            for _ in range(n_pairs):
                mats.extend(pair)
            structures.append(mats)
    
    # Triplets
    for n_trip in range(1, max_layers // 3 + 1):
        for triplet in ['LHL', 'HLH', 'LLH', 'HLL', 'LHH', 'HHL']:
            mats = []
            for _ in range(n_trip):
                mats.extend(triplet)
            structures.append(mats)
    
    # Remove duplicates
    seen = set()
    unique = []
    for s in structures:
        key = tuple(s)
        if key not in seen and len(s) <= max_layers:
            seen.add(key)
            unique.append(s)
    
    # Sort by length (shorter first for faster evaluation)
    unique.sort(key=lambda s: (len(s), s[0]))
    
    return unique

def _random_search(mats, problem, t0, deadline, n_trials=50):
    """Random restarts with coordinate descent."""
    n_layers = len(mats)
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    centre = 0.5 * sum(problem["band_nm"])
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    best_thick = None
    best_score = float('inf')
    
    for trial in range(n_trials):
        if time.time() > deadline - 10:
            break
        
        # Random initial guess
        if trial == 0:
            # Quarter-wave at center
            x0 = np.array([centre / (4.0 * (n_L if m == 'L' else n_H)) for m in mats])
        else:
            # Random with some structure
            x0 = np.random.uniform(0, max_layer_nm, size=n_layers)
            # Scale to fit total constraint
            if np.sum(x0) > max_total_nm:
                x0 = x0 * (max_total_nm / np.sum(x0)) * 0.9
        
        thick_opt, score = _coordinate_descent(mats, x0, problem, t0, deadline, max_iter=30)
        
        if score < best_score:
            best_score = score
            best_thick = thick_opt
    
    return best_thick, best_score

def design(problem):
    """Multi-layer AR coating design with vectorized evaluation and multi-method optimization."""
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s kill
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre = 0.5 * (band[0] + band[1])
    
    # Fallback: single quarter-wave L layer
    best_mats = ["L"]
    best_thick = [centre / (4.0 * n_L)]
    best_score = _mean_reflectance(best_mats, best_thick, problem)
    
    # Try single H layer
    thick_h = [centre / (4.0 * n_H)]
    score_h = _mean_reflectance(["H"], thick_h, problem)
    if score_h < best_score:
        best_score = score_h
        best_mats = ["H"]
        best_thick = thick_h
    
    # Generate candidate structures
    candidates = _generate_structures(problem, max_layers)
    
    # Limit candidates to avoid too many
    max_candidates = 80
    candidates = candidates[:max_candidates]
    
    # Optimize each candidate
    for mats in candidates:
        if time.time() > deadline - 20:
            break
        
        n_layers = len(mats)
        # Initial guess: quarter-wave at center wavelength
        x0 = np.array([centre / (4.0 * (n_L if m == 'L' else n_H)) for m in mats])
        
        # Try Nelder-Mead first (good for small problems)
        thick_opt, score = _nelder_mead_optimize(mats, x0, problem, t0, deadline)
        
        if thick_opt is None or score >= best_score:
            # Try coordinate descent
            thick_opt, score = _coordinate_descent(mats, x0, problem, t0, deadline, max_iter=50)
        
        if thick_opt is not None and score < best_score:
            best_score = score
            best_mats = mats[:]
            best_thick = thick_opt[:]
            
            # Refine with L-BFGS-B if time permits
            if time.time() < deadline - 30 and len(best_mats) >= 2:
                thick_refined, score_refined = _l_bfgs_b_optimize(best_mats, best_thick, problem, t0, deadline)
                if thick_refined is not None and score_refined < best_score:
                    best_score = score_refined
                    best_thick = thick_refined
    
    # Final refinement with random restarts on the best structure
    if time.time() < deadline - 10 and len(best_mats) >= 2:
        thick_random, score_random = _random_search(best_mats, problem, t0, deadline, n_trials=30)
        if score_random < best_score:
            best_score = score_random
            best_thick = thick_random
    
    # Final safety check
    if time.time() > deadline - 5:
        # Quick validation
        if any(t < 0 or t > max_layer_nm for t in best_thick):
            best_thick = [max(0, min(max_layer_nm, t)) for t in best_thick]
        if sum(best_thick) > max_total_nm:
            scale = max_total_nm / sum(best_thick)
            best_thick = [t * scale for t in best_thick]
    
    return best_mats, best_thick
# EVOLVE-BLOCK-END
