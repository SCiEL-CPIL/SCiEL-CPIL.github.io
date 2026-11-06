"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _reflectance_spectrum(materials, thicknesses, band_nm, n_ambient=1.0, n_substrate=1.52):
    """Compute mean reflectance (%) over equally spaced wavelengths using characteristic matrix method."""
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, 401)
    n_layers = len(materials)
    
    # Pre-compute indices
    n_list = np.array([n_ambient] + [materials[m] for m in materials] + [n_substrate])
    d_list = np.array([0.0] + thicknesses + [0.0])
    
    total_R = 0.0
    for lam in wavelengths:
        # Phase thickness: delta = 2*pi*n*d/lam
        # Build characteristic matrix product
        # M_total = M_1 * M_2 * ... * M_N
        # Each layer matrix: [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
        # At normal incidence
        
        # Use complex arithmetic
        # Start with identity for the product
        M = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]])
        
        for i in range(n_layers):
            n_i = n_list[i+1]
            d_i = d_list[i+1]
            delta = 2.0 * np.pi * n_i * d_i / lam
            c = np.cos(delta)
            s = np.sin(delta)
            # Layer matrix
            M_i = np.array([[c, 1j*s/n_i], [1j*n_i*s, c]])
            M = M @ M_i
        
        # The total field impedance at the air side
        # Y = M[1,0]/M[0,0] (in units of n_ambient for the admittance)
        # Actually, the characteristic matrix relates fields:
        # [E]   [M11 M12] [E]
        # [H] = [M21 M22] [H]
        # At the substrate: E_sub/H_sub = n_substrate (for normal incidence, lossless)
        # So the effective admittance at the air side: Y = M[1,0] * n_sub / M[0,0]
        # Wait, let me be more careful.
        
        # The characteristic matrix M satisfies:
        # [E(0)]   [M11 M12] [E(L)]
        # [H(0)] = [M21 M22] [H(L)]
        # where L is the total thickness. At the substrate boundary:
        # E(L)/H(L) = n_substrate (for a wave traveling into the substrate)
        # So H(L) = E(L)/n_substrate
        # Then:
        # E(0) = M11*E(L) + M12*H(L) = E(L)*(M11 + M12/n_substrate)
        # H(0) = M21*E(L) + M22*H(L) = E(L)*(M21 + M22/n_substrate)
        # The effective admittance at the air side: Y = H(0)/E(0) = (M21 + M22/n_sub) / (M11 + M12/n_sub)
        # Reflectance: R = |(n_ambient - Y)/(n_ambient + Y)|^2
        
        M11, M12 = M[0,0], M[0,1]
        M21, M22 = M[1,0], M[1,1]
        
        denom = M11 + M12 / n_substrate
        if abs(denom) < 1e-30:
            total_R += 1.0
            continue
        Y = (M21 + M22 / n_substrate) / denom
        R = abs((n_ambient - Y) / (n_ambient + Y))**2
        total_R += R * 100.0  # Convert to percent
    
    return total_R / len(wavelengths)

def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if len(materials) == 0:
        return False
    total = sum(thicknesses)
    if total > problem["max_total_nm"] + 1e-6:
        return False
    for t in thicknesses:
        if t < -1e-6 or t > problem["max_layer_nm"] + 1e-6:
            return False
    for m in materials:
        if m not in problem["materials"]:
            return False
    return True

def _optimize_thicknesses(materials, problem, max_iter=200, tol=1e-8):
    """Optimize thicknesses for a fixed set of materials using Nelder-Mead or simple gradient-free method."""
    n_layers = len(materials)
    if n_layers == 0:
        return [], 100.0
    
    best_thick = list(thicknesses_init(materials, problem))
    best_score = _reflectance_spectrum(materials, best_thick, problem["band_nm"])
    
    # Use scipy.optimize.minimize if available, otherwise fall back
    try:
        from scipy.optimize import minimize
        def objective(thick):
            if any(t < -1e-6 for t in thick):
                return 100.0
            return _reflectance_spectrum(materials, list(thick), problem["band_nm"])
        
        x0 = np.array(best_thick)
        bounds = [(0, problem["max_layer_nm"])] * n_layers
        
        # Use Nelder-Mead with bounds via penalty
        def penalized_obj(x):
            penalty = 0.0
            for i, (lo, hi) in enumerate(bounds):
                if x[i] < lo:
                    penalty += (lo - x[i])**2 * 100
                if x[i] > hi:
                    penalty += (x[i] - hi)**2 * 100
            total = np.sum(x)
            if total > problem["max_total_nm"]:
                penalty += (total - problem["max_total_nm"])**2 * 100
            return objective(x) + penalty
        
        result = minimize(penalized_obj, x0, method='Nelder-Mead', 
                         options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol})
        
        if result.fun < best_score:
            new_thick = [max(0, min(problem["max_layer_nm"], t)) for t in result.x]
            # Clamp and check
            if _is_valid(materials, new_thick, problem):
                best_thick = new_thick
                best_score = _reflectance_spectrum(materials, best_thick, problem["band_nm"])
    except Exception:
        pass
    
    return best_thick, best_score

def thicknesses_init(materials, problem):
    """Initialize thicknesses as quarter-wave at center wavelength."""
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    return [centre / (4.0 * problem["materials"][m]) for m in materials]

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n_ambient = problem.get("ambient_index", 1.0)
    n_substrate = problem.get("substrate_index", 1.52)
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    best_materials = None
    best_thicknesses = None
    best_score = 100.0
    
    # Generate candidate layer sequences
    # Start with analytical quarter-wave designs of various lengths
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    candidate_sets = []
    
    # Single layer
    candidate_sets.append(["L"])
    candidate_sets.append(["H"])
    
    # Two layers
    candidate_sets.append(["L", "H"])
    candidate_sets.append(["H", "L"])
    
    # Three layers
    candidate_sets.append(["L", "H", "L"])
    candidate_sets.append(["H", "L", "H"])
    
    # Four layers
    candidate_sets.append(["L", "H", "L", "H"])
    candidate_sets.append(["H", "L", "H", "L"])
    
    # Five layers
    candidate_sets.append(["L", "H", "L", "H", "L"])
    candidate_sets.append(["H", "L", "H", "L", "H"])
    
    # Six layers
    candidate_sets.append(["L", "H", "L", "H", "L", "H"])
    candidate_sets.append(["H", "L", "H", "L", "H", "L"])
    
    # Seven layers
    candidate_sets.append(["L", "H", "L", "H", "L", "H", "L"])
    candidate_sets.append(["H", "L", "H", "L", "H", "L", "H"])
    
    # Eight layers
    candidate_sets.append(["L", "H", "L", "H", "L", "H", "L", "H"])
    candidate_sets.append(["H", "L", "H", "L", "H", "L", "H", "L"])
    
    # Process candidates in order of increasing complexity
    for mats in candidate_sets:
        if time.time() > deadline - 5.0:
            break
        if len(mats) > max_layers:
            continue
        
        # Check if total quarter-wave thickness exceeds max_total_nm
        qw_thick = [centre / (4.0 * materials_dict[m]) for m in mats]
        if sum(qw_thick) > max_total_nm:
            # Try scaling down
            scale = max_total_nm / sum(qw_thick)
            qw_thick = [t * scale for t in qw_thick]
        
        if not _is_valid(mats, qw_thick, problem):
            continue
        
        # Evaluate initial score
        score = _reflectance_spectrum(mats, qw_thick, band_nm)
        if score < best_score:
            best_materials = mats
            best_thicknesses = qw_thick
            best_score = score
        
        # Optimize thicknesses
        if time.time() > deadline - 10.0:
            break
        
        opt_thick, opt_score = _optimize_thicknesses(mats, problem, max_iter=300)
        if opt_score < best_score and _is_valid(mats, opt_thick, problem):
            best_materials = mats
            best_thicknesses = opt_thick
            best_score = opt_score
    
    # If no valid design found, return a simple one
    if best_materials is None:
        centre = 0.5 * (band_nm[0] + band_nm[1])
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * materials_dict["L"])]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
