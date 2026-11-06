"""Broadband antireflection coating design using characteristic-matrix method."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    """Compute reflectance (0-1) for a stack using the characteristic matrix method."""
    n = len(n_layers)
    if n == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.abs(r) ** 2
    
    M11 = np.ones(len(wavelengths), dtype=complex)
    M12 = np.zeros(len(wavelengths), dtype=complex)
    M21 = np.zeros(len(wavelengths), dtype=complex)
    M22 = np.ones(len(wavelengths), dtype=complex)
    
    for i in range(n):
        ni = n_layers[i]
        di = d_layers[i]
        delta = 2.0 * np.pi * ni * di / wavelengths
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        L11 = cos_d
        L12 = 1j * sin_d / ni
        L21 = 1j * ni * sin_d
        L22 = cos_d
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
    
    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    
    return R


def _mean_reflectance_percent(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    """Compute mean reflectance in percent."""
    R = _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers)
    return np.mean(R) * 100.0


def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if len(materials) == 0:
        return True
    
    total = sum(thicknesses)
    if total > max_total_nm:
        return False
    
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    
    for m in materials:
        if m not in problem["materials"]:
            return False
    
    return True


def _evaluate_design(materials, thicknesses, problem, wavelengths):
    """Evaluate a design and return mean reflectance in percent."""
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_layers = np.array([problem["materials"][m] for m in materials], dtype=float)
    d_layers = np.array(thicknesses, dtype=float)
    
    return _mean_reflectance_percent(wavelengths, n_ambient, n_substrate, n_layers, d_layers)


def _objective_wrapper(x, mats, problem, wavelengths):
    """Objective function for scipy minimize with bounds handling."""
    # Clip to valid range
    x_clipped = np.clip(x, 0.0, 400.0)
    thicknesses = x_clipped.tolist()
    
    if not _is_valid(mats, thicknesses, problem):
        return 100.0  # Penalty for invalid
    
    return _evaluate_design(mats, thicknesses, problem, wavelengths)


def _nelder_mead_optimize(mats, initial_thicks, problem, wavelengths, deadline, max_iter=500):
    """Optimize thicknesses using Nelder-Mead simplex method."""
    n = len(mats)
    if n == 0:
        return mats, initial_thicks, _evaluate_design(mats, initial_thicks, problem, wavelengths)
    
    x0 = np.array(initial_thicks, dtype=float)
    
    # Define bounds
    bounds = [(0.0, 400.0)] * n
    
    # Check total thickness constraint
    if sum(x0) > problem["max_total_nm"]:
        # Scale down to fit
        scale = problem["max_total_nm"] / sum(x0)
        x0 = x0 * scale
    
    # Run Nelder-Mead
    result = minimize(
        _objective_wrapper,
        x0,
        args=(mats, problem, wavelengths),
        method='Nelder-Mead',
        bounds=bounds,
        options={
            'maxiter': max_iter,
            'xatol': 0.01,
            'fatol': 0.0001,
            'adaptive': True
        }
    )
    
    # Clip result to valid range
    x_opt = np.clip(result.x, 0.0, 400.0)
    thicknesses = x_opt.tolist()
    
    # Ensure total thickness constraint
    if sum(thicknesses) > problem["max_total_nm"]:
        scale = problem["max_total_nm"] / sum(thicknesses)
        thicknesses = [t * scale for t in thicknesses]
    
    # Final evaluation
    score = _evaluate_design(mats, thicknesses, problem, wavelengths)
    
    return mats, thicknesses, score


def _fine_polish(mats, thicks, problem, wavelengths, deadline, max_iter=2000):
    """High-precision polish using coordinate descent with small steps."""
    current_thicks = list(thicks)
    current_score = _evaluate_design(mats, current_thicks, problem, wavelengths)
    
    n = len(mats)
    if n == 0:
        return mats, current_thicks, current_score
    
    step_sizes = [1.0] * n
    min_step = 0.001
    max_step = 5.0
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            best_local_score = current_score
            best_local_thick = current_thicks[i]
            
            for factor in [1.0, 0.5, 0.25, 0.1, 0.05]:
                step = step_sizes[i] * factor
                
                # Try increasing
                new_thick = min(current_thicks[i] + step, 400.0)
                new_thicks = list(current_thicks)
                new_thicks[i] = new_thick
                if _is_valid(mats, new_thicks, problem):
                    new_score = _evaluate_design(mats, new_thicks, problem, wavelengths)
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local_thick = new_thick
                
                # Try decreasing
                new_thick = max(current_thicks[i] - step, 0.0)
                new_thicks = list(current_thicks)
                new_thicks[i] = new_thick
                if _is_valid(mats, new_thicks, problem):
                    new_score = _evaluate_design(mats, new_thicks, problem, wavelengths)
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local_thick = new_thick
            
            if best_local_score < current_score:
                current_thicks[i] = best_local_thick
                current_score = best_local_score
                improved = True
        
        if not improved:
            step_sizes = [s * 0.7 for s in step_sizes]
            if min(step_sizes) < min_step:
                break
        else:
            step_sizes = [min(s * 1.1, max_step) for s in step_sizes]
    
    return mats, current_thicks, current_score


def _generate_initial_designs(problem, wavelengths):
    """Generate diverse initial designs for optimization."""
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    
    # Multiple center wavelengths for diversity
    center_wavelengths = [500, 550, 600, 650, 450, 700]
    
    designs = []
    
    # Generate alternating patterns of various lengths
    for n_layers in range(1, min(15, max_layers + 1)):
        # Starting with H
        seq_h = []
        for i in range(n_layers):
            seq_h.append("H" if i % 2 == 0 else "L")
        
        # Starting with L
        seq_l = []
        for i in range(n_layers):
            seq_l.append("L" if i % 2 == 0 else "H")
        
        for center in center_wavelengths:
            # Quarter-wave thicknesses
            thicks_h = [center / (4.0 * materials_dict[m]) for m in seq_h]
            thicks_l = [center / (4.0 * materials_dict[m]) for m in seq_l]
            
            # Check validity and add
            if _is_valid(seq_h, thicks_h, problem):
                designs.append((list(seq_h), list(thicks_h)))
            if _is_valid(seq_l, thicks_l, problem):
                designs.append((list(seq_l), list(thicks_l)))
    
    # Add some non-alternating patterns
    special_patterns = [
        ["H", "L", "L", "H"],
        ["L", "H", "H", "L"],
        ["H", "L", "L", "H", "L", "L", "H"],
        ["L", "H", "H", "L", "H", "H", "L"],
        ["H", "L", "H", "L", "L", "H", "L", "H"],
        ["L", "H", "L", "H", "H", "L", "H", "L"],
        ["H", "L", "L", "H", "L", "L", "H", "L", "L", "H"],
    ]
    
    for pattern in special_patterns:
        if len(pattern) > max_layers:
            continue
        for center in center_wavelengths:
            thicks = [center / (4.0 * materials_dict[m]) for m in pattern]
            if _is_valid(pattern, thicks, problem):
                designs.append((list(pattern), list(thicks)))
    
    # Deduplicate
    seen = set()
    unique_designs = []
    for mats, thicks in designs:
        key = (tuple(mats), tuple(round(t, 2) for t in thicks))
        if key not in seen:
            seen.add(key)
            unique_designs.append((mats, thicks))
    
    return unique_designs


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    
    # Generate initial designs
    initial_designs = _generate_initial_designs(problem, wavelengths)
    
    if not initial_designs:
        # Fallback: single layer
        centre = 0.5 * (band_nm[0] + band_nm[1])
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
        best_score = _evaluate_design(best_materials, best_thicknesses, problem, wavelengths)
        return best_materials, best_thicknesses
    
    # Evaluate all initial designs
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    initial_results = []
    
    for mats, thicks in initial_designs:
        if time.time() > deadline - 30.0:
            break
        score = _evaluate_design(mats, thicks, problem, wavelengths)
        initial_results.append((score, mats, thicks))
        if score < best_score:
            best_score = score
            best_materials = list(mats)
            best_thicknesses = list(thicks)
    
    # Sort by score and optimize the top candidates
    initial_results.sort(key=lambda x: x[0])
    
    # Optimize top 5 candidates with Nelder-Mead
    top_candidates = initial_results[:5]
    
    for score, mats, thicks in top_candidates:
        if time.time() > deadline - 20.0:
            break
        
        mats_opt, thicks_opt, score_opt = _nelder_mead_optimize(
            mats, thicks, problem, wavelengths, deadline, max_iter=300
        )
        
        if score_opt < best_score:
            best_score = score_opt
            best_materials = mats_opt
            best_thicknesses = thicks_opt
    
    # Fine polish on the best design
    if time.time() < deadline - 5.0:
        best_materials, best_thicknesses, best_score = _fine_polish(
            best_materials, best_thicknesses, problem, wavelengths, deadline, max_iter=2000
        )
    
    # Final validation
    if not _is_valid(best_materials, best_thicknesses, problem):
        centre = 0.5 * (band_nm[0] + band_nm[1])
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
