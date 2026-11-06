"""Broadband antireflection coating design via global optimization."""
import time
import numpy as np
from scipy.optimize import minimize

def _compute_reflectance(thicknesses, materials, ambient_index, substrate_index, wavelengths):
    """Compute reflectance spectrum using characteristic matrix method.
    
    Args:
        thicknesses: list/array of physical thicknesses in nm
        materials: list/array of refractive indices for each layer
        ambient_index: refractive index of ambient (air)
        substrate_index: refractive index of substrate (glass)
        wavelengths: array of wavelengths in nm
    
    Returns:
        Array of reflectance values (0 to 1) for each wavelength
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        # Bare substrate
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wavelengths, abs(r)**2, dtype=float)
    
    reflectances = np.zeros(len(wavelengths), dtype=float)
    k0 = 2.0 * np.pi / 1.0  # in vacuum, but we'll use wavelength in nm
    
    for i, wl in enumerate(wavelengths):
        # Characteristic matrix for each layer
        # M_j = [[cos(delta_j), (i/sin(delta_j))], [(i*n_j*sin(delta_j), cos(delta_j)]]
        # where delta_j = 2*pi*n_j*d_j / lambda
        
        # Start with identity
        M_total = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]], dtype=complex)
        
        for j in range(n_layers):
            n_j = materials[j]
            d_j = thicknesses[j]
            delta_j = 2.0 * np.pi * n_j * d_j / wl
            
            cos_d = np.cos(delta_j)
            sin_d = np.sin(delta_j)
            
            # Avoid division by zero when sin_d is very small
            if abs(sin_d) < 1e-15:
                # Layer acts as zero thickness
                continue
            
            M_j = np.array([
                [cos_d, 1.0j * cos_d / sin_d],
                [1.0j * n_j * sin_d, cos_d]
            ], dtype=complex)
            
            M_total = M_total @ M_j
        
        # Add substrate interface
        # The admittance of the substrate
        B, C = M_total[0,0], M_total[0,1]
        D, E = M_total[1,0], M_total[1,1]
        
        # Actually, let me redo this properly.
        # The characteristic matrix for the stack is:
        # [B C] = product of all layer matrices
        # [D E]
        # 
        # The admittance Y = (ambient_index * D + ambient_index * E * n_sub) / ... 
        # No, let me use the standard formula.
        #
        # For a stack on a substrate:
        # Y = (n_sub * D + n_sub * E * n_sub) / (D + E * n_sub)  -- no that's wrong
        #
        # Standard formula:
        # The characteristic matrix M = [M11, M12; M21, M22]
        # B = M11 + M12 * n_s
        # C = M21 + M22 * n_s
        # Y = B/C  (in units of n_ambient)
        # R = |(Y - n_ambient)/(Y + n_ambient)|^2
        
        n_s = substrate_index
        B = M_total[0,0] + M_total[0,1] * n_s
        C = M_total[1,0] + M_total[1,1] * n_s
        
        if abs(C) < 1e-15:
            reflectances[i] = 1.0
        else:
            Y = B / C
            r = (Y - ambient_index) / (Y + ambient_index)
            reflectances[i] = abs(r)**2
    
    return reflectances

def _mean_reflectance_percent(thicknesses, materials_list, ambient_index, substrate_index, band_nm):
    """Compute mean reflectance in percent over the band."""
    wl_start, wl_end = band_nm
    wavelengths = np.linspace(wl_start, wl_end, 401)
    
    # Convert materials_list (indices) to actual refractive indices
    n_L = 1.38
    n_H = 2.35
    n_ambient = ambient_index
    n_sub = substrate_index
    
    refractive_indices = []
    for m in materials_list:
        if m == 'L':
            refractive_indices.append(n_L)
        elif m == 'H':
            refractive_indices.append(n_H)
    
    refl = _compute_reflectance(thicknesses, refractive_indices, n_ambient, n_sub, wavelengths)
    return np.mean(refl) * 100.0

def _is_valid(materials, thicknesses, max_layers, max_total_nm, max_layer_nm):
    """Check if the design satisfies all constraints."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if len(materials) == 0:
        return True
    if any(t < 0 or t > max_layer_nm for t in thicknesses):
        return False
    if sum(thicknesses) > max_total_nm:
        return False
    if any(m not in ('L', 'H') for m in materials):
        return False
    return True

def _optimize_continuous(materials, problem):
    """Optimize thicknesses for a fixed layer sequence using Nelder-Mead."""
    n_layers = len(materials)
    if n_layers == 0:
        return [], 4.26  # Bare glass
    
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    band_nm = problem["band_nm"]
    
    # Initial guess: quarter-wave at center wavelength
    centre_wl = 0.5 * (band_nm[0] + band_nm[1])
    n_L = 1.38
    n_H = 2.35
    
    x0 = np.zeros(n_layers)
    for i, m in enumerate(materials):
        n = n_L if m == 'L' else n_H
        x0[i] = centre_wl / (4.0 * n)
    
    # Clip to valid range
    x0 = np.clip(x0, 0, max_layer_nm)
    
    # Ensure total thickness constraint
    total = np.sum(x0)
    if total > max_total_nm:
        scale = max_total_nm / total
        x0 = x0 * scale
    
    def objective(x):
        # Penalty for constraint violations
        if np.any(x < 0) or np.any(x > max_layer_nm):
            return 100.0
        if np.sum(x) > max_total_nm:
            return 100.0
        return _mean_reflectance_percent(list(x), materials, ambient_index, substrate_index, band_nm)
    
    # Use Nelder-Mead
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 0.1, 'fatol': 0.001})
        x_opt = result.x
        # Clip to valid range
        x_opt = np.clip(x_opt, 0, max_layer_nm)
        total = np.sum(x_opt)
        if total > max_total_nm:
            scale = max_total_nm / total
            x_opt = x_opt * scale
        
        obj_val = objective(x_opt)
        return list(x_opt), obj_val
    except Exception:
        return list(x0), _mean_reflectance_percent(list(x0), materials, ambient_index, substrate_index, band_nm)

def _optimize_structure(materials, thicknesses, problem, max_time_per_structure):
    """Optimize both structure and thicknesses using local search."""
    start_time = time.time()
    best_materials = list(materials)
    best_thicknesses = list(thicknesses)
    best_score = _mean_reflectance_percent(best_thicknesses, best_materials, 
                                           problem["ambient_index"], problem["substrate_index"], 
                                           problem["band_nm"])
    
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # First, optimize thicknesses for current structure
    if len(best_materials) > 0:
        opt_thick, opt_score = _optimize_continuous(best_materials, problem)
        if opt_score < best_score:
            best_thicknesses = opt_thick
            best_score = opt_score
    
    # Local search on structure
    while time.time() - start_time < max_time_per_structure:
        improved = False
        
        # Try swapping adjacent layers
        for i in range(len(best_materials) - 1):
            new_materials = list(best_materials)
            new_materials[i], new_materials[i+1] = new_materials[i+1], new_materials[i]
            new_thick, new_score = _optimize_continuous(new_materials, problem)
            if new_score < best_score - 1e-6:
                best_materials = new_materials
                best_thicknesses = new_thick
                best_score = new_score
                improved = True
                break
        
        if improved:
            continue
        
        # Try adding a layer
        if len(best_materials) < max_layers:
            # Try adding L or H at various positions
            for pos in range(len(best_materials) + 1):
                for new_m in ['L', 'H']:
                    new_materials = best_materials[:pos] + [new_m] + best_materials[pos:]
                    new_thick, new_score = _optimize_continuous(new_materials, problem)
                    if new_score < best_score - 1e-6:
                        best_materials = new_materials
                        best_thicknesses = new_thick
                        best_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        if improved:
            continue
        
        # Try removing a layer
        if len(best_materials) > 1:
            for pos in range(len(best_materials)):
                new_materials = best_materials[:pos] + best_materials[pos+1:]
                new_thick, new_score = _optimize_continuous(new_materials, problem)
                if new_score < best_score - 1e-6:
                    best_materials = new_materials
                    best_thicknesses = new_thick
                    best_score = new_score
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_materials, best_thicknesses, best_score

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = 1.38
    n_H = 2.35
    centre_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    # Generate initial candidate structures
    candidates = []
    
    # Quarter-wave stack designs (classic AR coatings)
    # For broadband, use multiple quarter-wave pairs
    
    # 1. Single quarter-wave L
    candidates.append((["L"], [centre_wl / (4.0 * n_L)]))
    
    # 2. Single quarter-wave H
    candidates.append((["H"], [centre_wl / (4.0 * n_H)]))
    
    # 3. L-H pair (quarter-wave each)
    candidates.append((["L", "H"], [centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H)]))
    
    # 4. H-L pair
    candidates.append((["H", "L"], [centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L)]))
    
    # 5. L-H-L (three layers)
    candidates.append((["L", "H", "L"], [centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L)]))
    
    # 6. H-L-H
    candidates.append((["H", "L", "H"], [centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H)]))
    
    # 7. L-H-L-H (four layers)
    candidates.append((["L", "H", "L", "H"], [centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H), 
                                               centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H)]))
    
    # 8. H-L-H-L
    candidates.append((["H", "L", "H", "L"], [centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                                               centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L)]))
    
    # 9. More layers: L-H-L-H-L
    candidates.append((["L", "H", "L", "H", "L"], [centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                                                     centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                                                     centre_wl / (4.0 * n_L)]))
    
    # 10. H-L-H-L-H
    candidates.append((["H", "L", "H", "L", "H"], [centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                                                     centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                                                     centre_wl / (4.0 * n_H)]))
    
    # 11. Six layers
    candidates.append((["L", "H", "L", "H", "L", "H"], 
                       [centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H)]))
    
    # 12. H-L-H-L-H-L
    candidates.append((["H", "L", "H", "L", "H", "L"],
                       [centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L)]))
    
    # 13. Eight layers
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H"],
                       [centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H)]))
    
    # 14. H-L-H-L-H-L-H-L
    candidates.append((["H", "L", "H", "L", "H", "L", "H", "L"],
                       [centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L)]))
    
    # 15. Ten layers
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H"],
                       [centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H)]))
    
    # 16. H-L-H-L-H-L-H-L-H-L
    candidates.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
                       [centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L)]))
    
    # 17. Twelve layers
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"],
                       [centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H)]))
    
    # 18. H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
                       [centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L)]))
    
    # 19. Fourteen layers
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"],
                       [centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H)]))
    
    # 20. H-L-H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
                       [centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L)]))
    
    # 21. Sixteen layers
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"],
                       [centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H)]))
    
    # 22. H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
                       [centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L)]))
    
    # 23. Eighteen layers
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"],
                       [centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H)]))
    
    # 24. H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
                       [centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L)]))
    
    # 25. Twenty layers (max)
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"],
                       [centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H),
                        centre_wl / (4.0 * n_L), centre_wl / (4.0 * n_H)]))
    
    # 26. H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L-H-L
    candidates.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
                       [centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L),
                        centre_wl / (4.0 * n_H), centre_wl / (4.0 * n_L)]))
    
    # Filter out candidates that violate constraints
    valid_candidates = []
    for mats, thicks in candidates:
        if _is_valid(mats, thicks, max_layers, max_total_nm, max_layer_nm):
            valid_candidates.append((mats, thicks))
    
    # Evaluate all candidates and find the best
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    for mats, thicks in valid_candidates:
        score = _mean_reflectance_percent(thicks, mats, ambient_index, substrate_index, band_nm)
        if score < best_score:
            best_score = score
            best_materials = list(mats)
            best_thicknesses = list(thicks)
    
    # If no valid candidates, use bare glass equivalent
    if best_materials is None:
        return [], []
    
    # Now try to improve the best candidate with local optimization
    if time.time() < deadline - 5.0:
        try:
            improved_mats, improved_thicks, improved_score = _optimize_structure(
                best_materials, best_thicknesses, problem, 
                max(1.0, deadline - time.time() - 5.0)
            )
            if improved_score < best_score:
                best_materials = improved_mats
                best_thicknesses = improved_thicks
                best_score = improved_score
        except Exception:
            pass
    
    # Verify final result is valid
    if not _is_valid(best_materials, best_thicknesses, max_layers, max_total_nm, max_layer_nm):
        # Fallback to a simple valid design
        return ["L"], [centre_wl / (4.0 * n_L)]
    
    return best_materials, best_thicknesses
