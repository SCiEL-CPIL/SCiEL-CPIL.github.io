"""Broadband antireflection coating design via Nelder-Mead optimization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _char_matrix(n, d, wl):
    """
    Characteristic matrix for a single lossless layer at normal incidence.
    n: refractive index (real)
    d: physical thickness (nm)
    wl: wavelength (nm)
    Returns 2x2 numpy complex array.
    """
    beta = 2.0 * np.pi * n * d / wl
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    return np.array([
        [cos_b, -1j * sin_b / n],
        [1j * n * sin_b, cos_b]
    ], dtype=complex)

def _compute_reflectance(materials, thicknesses, n_ambient, n_substrate, wl):
    """
    Compute reflectance (0 to 1) at a single wavelength using the
    characteristic-matrix method.
    """
    # Start with identity matrix
    m = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    
    # Multiply by each layer's characteristic matrix (air side to substrate)
    for i in range(len(materials)):
        n_layer = materials[materials[i]]
        d_layer = thicknesses[i]
        m = m @ _char_matrix(n_layer, d_layer, wl)
    
    # Substrate admittance
    y_s = n_substrate
    
    # Input admittance
    y_in = (m[0, 0] + m[0, 1] * y_s) / (m[1, 0] + m[1, 1] * y_s)
    
    # Reflectance
    r = (n_ambient - y_in) / (n_ambient + y_in)
    return np.abs(r) ** 2

def _mean_reflectance_percent(materials, thicknesses, n_ambient, n_substrate, band_nm):
    """
    Compute mean reflectance in percent over the band.
    Uses 401 equally spaced wavelengths.
    """
    wl_min, wl_max = band_nm
    wavelengths = np.linspace(wl_min, wl_max, 401)
    
    total_r = 0.0
    for wl in wavelengths:
        total_r += _compute_reflectance(materials, thicknesses, n_ambient, n_substrate, wl)
    
    mean_r = total_r / len(wavelengths)
    return mean_r * 100.0

def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if any(t < -1e-9 or t > problem["max_layer_nm"] + 1e-9 for t in thicknesses):
        return False
    if sum(thicknesses) > problem["max_total_nm"] + 1e-9:
        return False
    if any(m not in problem["materials"] for m in materials):
        return False
    return True

def _nelder_mead(f, x0, args=(), max_iter=1000, tol=1e-6, deadline=None):
    """
    Nelder-Mead simplex optimization.
    f: objective function
    x0: initial point (list)
    args: additional arguments for f
    Returns: (best_x, best_f)
    """
    n = len(x0)
    simplex = [list(x0)]
    f0 = f(x0, *args)
    simplex.append([x0[i] + 0.1 * (x0[i] + 1) if i < n else 0.1 for i in range(n)])
    # Better initialization
    simplex = [list(x0)]
    for i in range(n):
        point = list(x0)
        step = 0.1 if x0[i] == 0 else 0.1 * abs(x0[i])
        point[i] += step
        simplex.append(point)
    
    fvals = [f(x, *args) for x in simplex]
    best_x = list(x0)
    best_f = f0
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid of all points except worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_ref = centroid + alpha * (centroid - simplex[-1])
        f_ref = f(x_ref, *args)
        
        if fvals[0] <= f_ref < fvals[-2]:
            simplex[-1] = list(x_ref)
            fvals[-1] = f_ref
        elif f_ref < fvals[0]:
            # Expansion
            gamma = 2.0
            x_exp = centroid + gamma * (x_ref - centroid)
            f_exp = f(x_exp, *args)
            if f_exp < f_ref:
                simplex[-1] = list(x_exp)
                fvals[-1] = f_exp
            else:
                simplex[-1] = list(x_ref)
                fvals[-1] = f_ref
        else:
            # Contraction
            rho = 0.5
            x_con = centroid + rho * (simplex[-1] - centroid)
            f_con = f(x_con, *args)
            if f_con < fvals[-1]:
                simplex[-1] = list(x_con)
                fvals[-1] = f_con
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    fvals[i] = f(simplex[i], *args)
        
        # Track best
        min_idx = np.argmin(fvals)
        if fvals[min_idx] < best_f:
            best_f = fvals[min_idx]
            best_x = list(simplex[min_idx])
    
    return best_x, best_f

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_l = materials_dict["L"]
    n_h = materials_dict["H"]
    
    # Try different numbers of layers
    best_materials = ["L"]
    best_thicknesses = [100.0]
    best_score = _mean_reflectance_percent(best_materials, best_thicknesses, n_ambient, n_substrate, band_nm)
    
    # Objective function for optimization
    def objective(thicknesses, layer_pattern=None):
        if layer_pattern is None:
            layer_pattern = best_materials
        # Ensure validity
        if any(t < 0 for t in thicknesses):
            return 1000.0
        if any(t > max_layer_nm for t in thicknesses):
            return 1000.0
        if sum(thicknesses) > max_total_nm:
            return 1000.0
        score = _mean_reflectance_percent(layer_pattern, thicknesses, n_ambient, n_substrate, band_nm)
        return score
    
    # Try different layer patterns
    patterns_to_try = [
        ["L"],
        ["H"],
        ["L", "H"],
        ["H", "L"],
        ["L", "H", "L"],
        ["H", "L", "H"],
        ["L", "H", "L", "H"],
        ["H", "L", "H", "L"],
        ["L", "H", "L", "H", "L"],
        ["H", "L", "H", "L", "H"],
        ["L", "H", "L", "H", "L", "H"],
        ["H", "L", "H", "L", "H", "L"],
    ]
    
    for pattern in patterns_to_try:
        if time.time() > deadline - 10.0:
            break
        if len(pattern) > max_layers:
            continue
        
        # Initial guess: quarter-wave at center wavelength
        centre = 0.5 * (band_nm[0] + band_nm[1])
        initial_thicknesses = []
        for mat in pattern:
            n = materials_dict[mat]
            d = centre / (4.0 * n)
            initial_thicknesses.append(d)
        
        # Optimize
        x0 = initial_thicknesses
        best_x, best_f = _nelder_mead(objective, x0, args=(pattern,), max_iter=500, tol=1e-5, deadline=deadline)
        
        # Check validity
        if _is_valid(pattern, best_x, problem):
            # Clip to valid range
            clipped = [max(0.0, min(max_layer_nm, t)) for t in best_x]
            if sum(clipped) > max_total_nm:
                # Scale down
                scale = max_total_nm / sum(clipped)
                clipped = [t * scale for t in clipped]
            
            score = _mean_reflectance_percent(pattern, clipped, n_ambient, n_substrate, band_nm)
            if score < best_score:
                best_score = score
                best_materials = list(pattern)
                best_thicknesses = clipped
    
    # Additional local refinement on the best found
    if time.time() < deadline - 5.0 and len(best_materials) > 0:
        def obj_final(thicknesses):
            if any(t < 0 for t in thicknesses):
                return 1000.0
            if any(t > max_layer_nm for t in thicknesses):
                return 1000.0
            if sum(thicknesses) > max_total_nm:
                return 1000.0
            return _mean_reflectance_percent(best_materials, thicknesses, n_ambient, n_substrate, band_nm)
        
        x0 = best_thicknesses
        best_x, best_f = _nelder_mead(obj_final, x0, max_iter=300, tol=1e-6, deadline=deadline)
        
        if _is_valid(best_materials, best_x, problem):
            clipped = [max(0.0, min(max_layer_nm, t)) for t in best_x]
            if sum(clipped) > max_total_nm:
                scale = max_total_nm / sum(clipped)
                clipped = [t * scale for t in clipped]
            score = _mean_reflectance_percent(best_materials, clipped, n_ambient, n_substrate, band_nm)
            if score < best_score:
                best_score = score
                best_thicknesses = clipped
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
