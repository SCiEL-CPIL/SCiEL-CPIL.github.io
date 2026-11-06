"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _characteristic_matrix(n, d, lam, lam0=1.0):
    """Compute the 2x2 characteristic matrix for a single layer."""
    beta = 2 * np.pi * n * d / lam
    c = np.cos(beta)
    s = np.sin(beta)
    return np.array([[c, 1j * s], [1j * n * s, c]])

def _reflectance(wavelengths, materials, thicknesses, n_substrate):
    """
    Compute reflectance (fraction) for given wavelengths using characteristic matrix method.
    wavelengths: array of wavelengths in nm
    materials: list of refractive indices
    thicknesses: list of thicknesses in nm
    n_substrate: substrate refractive index
    Returns array of reflectance fractions.
    """
    n_layers = len(materials)
    n_waves = len(wavelengths)
    reflectance = np.zeros(n_waves)
    
    for i, lam in enumerate(wavelengths):
        # Start with substrate matrix
        M = np.array([[1.0, 0.0], [n_substrate, 0.0]])
        
        # Multiply by each layer's matrix (from substrate side to air side, but we build from air to substrate)
        # Actually, the characteristic matrix is multiplied from the first layer (air side) to the last (substrate side)
        # The total matrix M_total = M_1 * M_2 * ... * M_N * M_substrate
        # Wait, the convention is: M_total = M_1 * M_2 * ... * M_N
        # where M_i is the matrix for layer i, and the boundary condition relates the fields.
        # Let's use the standard approach:
        # B = M_total * [1, n_substrate]^T
        # R = |(B[0,0] + B[0,1]*n_ambient) - (B[1,0] + B[1,1]*n_ambient) * n_ambient| / |(B[0,0] + B[0,1]*n_ambient) + (B[1,0] + B[1,1]*n_ambient) * n_ambient|
        # Actually, the standard formula is:
        # [E, H]^T at air side = M_total * [E, H]^T at substrate side
        # At substrate side, H/E = n_substrate (for forward wave)
        # So [E, H]^T at air side = M_total * [1, n_substrate]^T
        # Let [E, H]^T = [B0, B1]^T
        # Then r = (E - H/n_ambient) / (E + H/n_ambient) = (B0*n_ambient - B1) / (B0*n_ambient + B1)
        
        # Let's compute M_total
        M_total = np.array([[1.0, 0.0], [0.0, 1.0]])
        for j in range(n_layers):
            M_layer = _characteristic_matrix(materials[j], thicknesses[j], lam)
            M_total = M_total @ M_layer
        
        # Now multiply by substrate vector
        # The field at the substrate interface (just before substrate) is [1, n_substrate] for a unit incident wave
        # Actually, we need to be careful with the direction.
        # Standard: M_total = M_1 * M_2 * ... * M_N
        # [E_0, H_0]^T = M_total * [E_N, H_N]^T
        # At the last interface (layer N / substrate), the transmitted wave has H_N/E_N = n_sub
        # So [E_N, H_N]^T = [1, n_sub]^T (up to a scale factor)
        
        vec = np.array([1.0, n_substrate])
        result = M_total @ vec
        B0, B1 = result[0], result[1]
        
        n_ambient = 1.0
        r = (B0 * n_ambient - B1) / (B0 * n_ambient + B1)
        reflectance[i] = np.abs(r)**2
    
    return reflectance

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    n_substrate = problem["substrate_index"]
    refl = _reflectance(wavelengths, materials, thicknesses, n_substrate)
    return np.mean(refl) * 100.0

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if sum(thicknesses) > problem["max_total_nm"]:
        return False
    for t in thicknesses:
        if t < 0 or t > problem["max_layer_nm"]:
            return False
    return True

def design(problem):
    """
    Design a broadband antireflection coating using local search with layer add/remove
    and thickness optimization.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_sub = problem["substrate_index"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    # Precompute wavelength grid
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Analytical initializations: quarter-wave stacks
    # Try different numbers of layers
    best_materials = ["L"]
    best_thicknesses = [band[0] / (4.0 * n_L)]  # arbitrary initial
    best_score = _mean_reflectance(best_materials, best_thicknesses, problem)
    
    # Generate multiple initializations
    initializations = []
    
    # Single quarter-wave layer at center wavelength
    lam_c = 0.5 * (band[0] + band[1])
    initializations.append((["L"], [lam_c / (4.0 * n_L)]))
    initializations.append((["H"], [lam_c / (4.0 * n_H)]))
    
    # Two-layer quarter-wave stack
    initializations.append((["L", "H"], [lam_c / (4.0 * n_L), lam_c / (4.0 * n_H)]))
    initializations.append((["H", "L"], [lam_c / (4.0 * n_H), lam_c / (4.0 * n_L)]))
    
    # Three-layer quarter-wave stack
    initializations.append((["L", "H", "L"], [lam_c / (4.0 * n_L), lam_c / (4.0 * n_H), lam_c / (4.0 * n_L)]))
    initializations.append((["H", "L", "H"], [lam_c / (4.0 * n_H), lam_c / (4.0 * n_L), lam_c / (4.0 * n_H)]))
    
    # Four-layer quarter-wave stack
    initializations.append((["L", "H", "L", "H"], [lam_c / (4.0 * n_L), lam_c / (4.0 * n_H), lam_c / (4.0 * n_L), lam_c / (4.0 * n_H)]))
    initializations.append((["H", "L", "H", "L"], [lam_c / (4.0 * n_H), lam_c / (4.0 * n_L), lam_c / (4.0 * n_H), lam_c / (4.0 * n_L)]))
    
    # Five-layer
    initializations.append((["L", "H", "L", "H", "L"], [lam_c / (4.0 * n_L), lam_c / (4.0 * n_H), lam_c / (4.0 * n_L), lam_c / (4.0 * n_H), lam_c / (4.0 * n_L)]))
    
    # Six-layer
    initializations.append((["L", "H", "L", "H", "L", "H"], [lam_c / (4.0 * n_L), lam_c / (4.0 * n_H), lam_c / (4.0 * n_L), lam_c / (4.0 * n_H), lam_c / (4.0 * n_L), lam_c / (4.0 * n_H)]))
    
    # Optimize each initialization
    for mats, thicks in initializations:
        if time.time() > deadline:
            break
        if not _is_valid(mats, thicks, problem):
            continue
        
        # Optimize thicknesses
        n_layers = len(thicks)
        x0 = np.array(thicks, dtype=float)
        
        def objective(x):
            if time.time() > deadline:
                return 1e10
            if np.any(x < 0) or np.any(x > max_layer):
                return 1e10
            if np.sum(x) > max_total:
                return 1e10
            return _mean_reflectance(mats, list(x), problem)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                              bounds=[(0, max_layer)] * n_layers,
                              options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_materials = list(mats)
                best_thicknesses = list(result.x)
        except Exception:
            pass
    
    # Local search: try adding/removing layers and re-optimizing
    for _ in range(50):  # iterations
        if time.time() > deadline:
            break
        
        current_mats = list(best_materials)
        current_thicks = list(best_thicknesses)
        current_score = best_score
        
        # Try adding a layer
        for pos in range(len(current_mats) + 1):
            if len(current_mats) >= max_layers:
                break
            for new_mat in ["L", "H"]:
                new_mats = current_mats[:pos] + [new_mat] + current_mats[pos:]
                new_thicks = current_thicks[:pos] + [max_layer / 2.0] + current_thicks[pos:]
                if not _is_valid(new_mats, new_thicks, problem):
                    continue
                
                n_layers = len(new_thicks)
                x0 = np.array(new_thicks, dtype=float)
                
                def obj_add(x, m=new_mats):
                    if time.time() > deadline:
                        return 1e10
                    if np.any(x < 0) or np.any(x > max_layer):
                        return 1e10
                    if np.sum(x) > max_total:
                        return 1e10
                    return _mean_reflectance(m, list(x), problem)
                
                try:
                    result = minimize(obj_add, x0, method='L-BFGS-B',
                                      bounds=[(0, max_layer)] * n_layers,
                                      options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
                    if result.success and result.fun < current_score - 1e-10:
                        current_score = result.fun
                        current_mats = new_mats
                        current_thicks = list(result.x)
                except Exception:
                    pass
        
        # Try removing a layer
        if len(current_mats) > 1:
            for pos in range(len(current_mats)):
                new_mats = current_mats[:pos] + current_mats[pos+1:]
                new_thicks = current_thicks[:pos] + current_thicks[pos+1:]
                if not _is_valid(new_mats, new_thicks, problem):
                    continue
                
                n_layers = len(new_thicks)
                x0 = np.array(new_thicks, dtype=float)
                
                def obj_rem(x, m=new_mats):
                    if time.time() > deadline:
                        return 1e10
                    if np.any(x < 0) or np.any(x > max_layer):
                        return 1e10
                    if np.sum(x) > max_total:
                        return 1e10
                    return _mean_reflectance(m, list(x), problem)
                
                try:
                    result = minimize(obj_rem, x0, method='L-BFGS-B',
                                      bounds=[(0, max_layer)] * n_layers,
                                      options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
                    if result.success and result.fun < current_score - 1e-10:
                        current_score = result.fun
                        current_mats = new_mats
                        current_thicks = list(result.x)
                except Exception:
                    pass
        
        # Try swapping layer order
        for i in range(len(current_mats)):
            for j in range(i+1, len(current_mats)):
                new_mats = list(current_mats)
                new_mats[i], new_mats[j] = new_mats[j], new_mats[i]
                new_thicks = list(current_thicks)
                new_thicks[i], new_thicks[j] = new_thicks[j], new_thicks[i]
                if not _is_valid(new_mats, new_thicks, problem):
                    continue
                
                n_layers = len(new_thicks)
                x0 = np.array(new_thicks, dtype=float)
                
                def obj_swap(x, m=new_mats):
                    if time.time() > deadline:
                        return 1e10
                    if np.any(x < 0) or np.any(x > max_layer):
                        return 1e10
                    if np.sum(x) > max_total:
                        return 1e10
                    return _mean_reflectance(m, list(x), problem)
                
                try:
                    result = minimize(obj_swap, x0, method='L-BFGS-B',
                                      bounds=[(0, max_layer)] * n_layers,
                                      options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
                    if result.success and result.fun < current_score - 1e-10:
                        current_score = result.fun
                        current_mats = new_mats
                        current_thicks = list(result.x)
                except Exception:
                    pass
        
        if current_score < best_score:
            best_materials = current_mats
            best_thicknesses = current_thicks
            best_score = current_score
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        n_layers = len(best_thicknesses)
        x0 = np.array(best_thicknesses, dtype=float)
        
        def obj_final(x):
            if np.any(x < 0) or np.any(x > max_layer):
                return 1e10
            if np.sum(x) > max_total:
                return 1e10
            return _mean_reflectance(best_materials, list(x), problem)
        
        try:
            result = minimize(obj_final, x0, method='L-BFGS-B',
                              bounds=[(0, max_layer)] * n_layers,
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.success and result.fun < best_score:
                best_thicknesses = list(result.x)
                best_score = result.fun
        except Exception:
            pass
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
