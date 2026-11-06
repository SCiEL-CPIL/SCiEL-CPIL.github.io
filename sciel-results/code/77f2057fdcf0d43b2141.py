"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _reflectance_vectorized(materials, thicknesses, wls, ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """
    Compute reflectance at all wavelengths using the characteristic matrix method (vectorized).
    """
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    n_layers = len(materials)
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wls, r*r)
        
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for mat, thick in zip(materials, thicknesses):
        n = materials_dict[mat]
        if thick < 1e-9:
            continue
            
        beta = 2.0 * np.pi * n * thick / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        L11 = cos_b
        L12 = 1.0j * sin_b / n
        L21 = n * 1.0j * sin_b
        L22 = cos_b
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
        
    Y_s = substrate_index
    Y = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    
    Y_0 = ambient_index
    r = (Y_0 - Y) / (Y_0 + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance(materials, thicknesses, band_nm=(400, 800), n_points=401,
                       ambient_index=1.0, substrate_index=1.52, materials_dict=None):
    """Compute mean reflectance in percent over the band."""
    if materials_dict is None:
        materials_dict = {'L': 1.38, 'H': 2.35}
    
    wls = np.linspace(band_nm[0], band_nm[1], n_points)
    refls = _reflectance_vectorized(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict)
    return 100.0 * np.mean(refls)

def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True

def _objective(thicknesses, materials, problem, wls, materials_dict, 
               ambient_index, substrate_index, band_nm):
    """
    Objective function for scipy.optimize.minimize.
    Returns mean reflectance + penalty for constraint violations.
    """
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    # Constraint checks
    penalty = 0.0
    for t in thicknesses:
        if t < 0:
            penalty += 10.0 * abs(t)
        elif t > max_layer_nm:
            penalty += 10.0 * (t - max_layer_nm)
    
    total_thick = sum(thicknesses)
    if total_thick > max_total_nm:
        penalty += 10.0 * (total_thick - max_total_nm)
    
    if penalty > 0:
        return 10.0 + penalty # Return a large value to push optimizer away
    
    # Calculate reflectance
    return _mean_reflectance(materials, list(thicknesses), band_nm, 401, 
                             ambient_index, substrate_index, materials_dict)

def design(problem):
    """
    Design using Nelder-Mead optimization over thicknesses for fixed material sequences.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    band_nm = problem["band_nm"]
    wl_center = 0.5 * (band_nm[0] + band_nm[1])
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layers = problem.get("max_layers", 20)
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Generate candidate material sequences
    # We try lengths 3 to 11, with alternating L/H
    # Start with L or H
    candidates = []
    for length in range(3, 12, 2): # 3, 5, 7, 9, 11
        for start_mat in ['L', 'H']:
            mats = []
            for i in range(length):
                if i % 2 == 0:
                    mats.append(start_mat)
                else:
                    mats.append('H' if start_mat == 'L' else 'L')
            candidates.append(mats)
            
    # Sort candidates by length, shorter ones are faster to optimize
    candidates.sort(key=lambda x: len(x))
    
    for mats in candidates:
        if time.time() - start_time > deadline - 15:
            break
            
        n_layers = len(mats)
        
        # Initial guess: quarter wave thicknesses, scaled to fit total constraint
        init_thicks = []
        for mat in mats:
            n = nL if mat == 'L' else nH
            d = wl_center / (4.0 * n)
            init_thicks.append(d)
        
        # Scale down if total exceeds 90% of max_total_nm
        total = sum(init_thicks)
        if total > 0.9 * max_total_nm:
            scale = (0.9 * max_total_nm) / total
            init_thicks = [t * scale for t in init_thicks]
        
        # Ensure bounds
        init_thicks = [max(0, min(max_layer_nm, t)) for t in init_thicks]
        
        # Run Nelder-Mead
        try:
            # Use a tighter tolerance for better precision
            res = minimize(
                _objective, 
                init_thicks, 
                args=(mats, problem, wls, materials_dict, n_amb, n_sub, band_nm),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000, 
                    'xatol': 1e-4, 
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            # Check validity of the result
            res_thicks = list(res.x)
            # Clamp to valid range just in case
            res_thicks = [max(0, min(max_layer_nm, t)) for t in res_thicks]
            
            if _is_valid(mats, res_thicks, problem):
                score = _mean_reflectance(mats, res_thicks, band_nm, 401, n_amb, n_sub, materials_dict)
                if score < best_score:
                    best_score = score
                    best_materials = list(mats)
                    best_thicknesses = res_thicks
        except Exception:
            continue
            
    # If no valid result found, return a simple valid one
    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [wl_center / (4.0 * nL)]
        best_score = _mean_reflectance(best_materials, best_thicknesses, band_nm, 401, n_amb, n_sub, materials_dict)
        
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
