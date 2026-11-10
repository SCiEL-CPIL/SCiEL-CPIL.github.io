"""Broadband AR coating design using characteristic-matrix method."""
import numpy as np
import time

def _characteristic_matrix(n0, ns, materials, thicknesses, lambdas):
    """
    Compute reflectance using the characteristic matrix method.
    n0: ambient index, ns: substrate index
    materials: list of layer indices (same order as thicknesses)
    thicknesses: list of physical thicknesses in nm
    lambdas: array of wavelengths in nm
    Returns: array of reflectance values (0 to 1)
    """
    N = len(thicknesses)
    if N == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full_like(lambdas, r * r, dtype=float)
    
    # Build total characteristic matrix M = M1 * M2 * ... * MN
    # Mj = [[cos(delta_j), i*sin(delta_j)/nj], [i*nj*sin(delta_j), cos(delta_j)]]
    # delta_j = 2*pi*nj*d_j/lambda
    
    M11 = np.ones_like(lambdas, dtype=complex)
    M12 = np.zeros_like(lambdas, dtype=complex)
    M21 = np.zeros_like(lambdas, dtype=complex)
    M22 = np.ones_like(lambdas, dtype=complex)
    
    for idx in range(N - 1, -1, -1):
        nj = materials[idx]
        dj = thicknesses[idx]
        delta = 2.0 * np.pi * nj * dj / lambdas
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # New M = M_old * Mj
        new_M11 = M11 * cos_d + M12 * 1j * nj * sin_d
        new_M12 = M11 * 1j * sin_d / nj + M12 * cos_d
        new_M21 = M21 * cos_d + M22 * 1j * nj * sin_d
        new_M22 = M21 * 1j * sin_d / nj + M22 * cos_d
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Reflectance: R = |Y - n0|^2 / |Y + n0|^2 where Y = M21/M11
    Y = M21 / M11
    r = (Y - n0) / (Y + n0)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(n0, ns, materials_list, thicknesses, lambdas):
    """Compute mean reflectance in percent."""
    R = _characteristic_matrix(n0, ns, materials_list, thicknesses, lambdas)
    return 100.0 * np.mean(R)

def _is_valid(thicknesses, max_layer_nm, max_total_nm, max_layers):
    """Check if thicknesses satisfy constraints."""
    if len(thicknesses) > max_layers:
        return False
    if any(t < -1e-10 or t > max_layer_nm + 1e-10 for t in thicknesses):
        return False
    if sum(thicknesses) > max_total_nm + 1e-10:
        return False
    return True

def _clipped(thicknesses, max_layer_nm):
    """Clip thicknesses to valid range."""
    return [max(0.0, min(max_layer_nm, t)) for t in thicknesses]

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    wl_min, wl_max = band[0], band[1]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation
    lambdas = np.linspace(wl_min, wl_max, 401)
    centre_wl = 0.5 * (wl_min + wl_max)
    
    # Start with a reasonable multi-layer stack
    # Use alternating L and H layers, starting with L (air side)
    n_layers = 8
    materials_list = []
    thicknesses = []
    
    for i in range(n_layers):
        if i % 2 == 0:
            materials_list.append(nL)
            # Quarter-wave at center wavelength
            thicknesses.append(centre_wl / (4.0 * nL))
        else:
            materials_list.append(nH)
            thicknesses.append(centre_wl / (4.0 * nH))
    
    # Ensure validity
    total = sum(thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
    
    # Coordinate descent optimization
    best_thicknesses = list(thicknesses)
    best_score = _mean_reflectance_percent(n0, ns, materials_list, best_thicknesses, lambdas)
    
    step_sizes = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5]
    
    for step in step_sizes:
        if time.time() > deadline:
            break
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            
            for i in range(len(thicknesses)):
                if time.time() > deadline:
                    break
                
                old_t = thicknesses[i]
                # Try increasing
                new_t = min(max_layer_nm, old_t + step)
                if abs(new_t - old_t) > 1e-10:
                    thicknesses[i] = new_t
                    if sum(thicknesses) <= max_total_nm:
                        score = _mean_reflectance_percent(n0, ns, materials_list, thicknesses, lambdas)
                        if score < best_score:
                            best_score = score
                            best_thicknesses = list(thicknesses)
                            improved = True
                        else:
                            thicknesses[i] = old_t
                    else:
                        thicknesses[i] = old_t
                
                # Try decreasing
                new_t = max(0.0, old_t - step)
                if abs(new_t - old_t) > 1e-10:
                    thicknesses[i] = new_t
                    score = _mean_reflectance_percent(n0, ns, materials_list, thicknesses, lambdas)
                    if score < best_score:
                        best_score = score
                        best_thicknesses = list(thicknesses)
                        improved = True
                    else:
                        thicknesses[i] = old_t
        
        # After each step size, reset to best
        thicknesses = list(best_thicknesses)
    
    # Final polish with smaller steps
    if time.time() < deadline:
        for step in [0.2, 0.1, 0.05]:
            if time.time() > deadline:
                break
            improved = True
            while improved:
                improved = False
                if time.time() > deadline:
                    break
                for i in range(len(thicknesses)):
                    if time.time() > deadline:
                        break
                    old_t = thicknesses[i]
                    for delta_t in [step, -step]:
                        new_t = old_t + delta_t
                        if new_t < 0 or new_t > max_layer_nm:
                            continue
                        thicknesses[i] = new_t
                        if sum(thicknesses) <= max_total_nm:
                            score = _mean_reflectance_percent(n0, ns, materials_list, thicknesses, lambdas)
                            if score < best_score:
                                best_score = score
                                best_thicknesses = list(thicknesses)
                                improved = True
                            else:
                                thicknesses[i] = old_t
                        else:
                            thicknesses[i] = old_t
            thicknesses = list(best_thicknesses)
    
    # Convert back to L/H labels
    labels = []
    final_thicknesses = []
    for i in range(len(best_thicknesses)):
        if best_thicknesses[i] < 1e-6:
            continue  # Skip zero-thickness layers
        if abs(materials_list[i] - nL) < 1e-10:
            labels.append("L")
        else:
            labels.append("H")
        final_thicknesses.append(best_thicknesses[i])
    
    # Ensure we don't exceed max_layers
    if len(labels) > max_layers:
        labels = labels[:max_layers]
        final_thicknesses = final_thicknesses[:max_layers]
    
    # Final validity check
    if not _is_valid(final_thicknesses, max_layer_nm, max_total_nm, max_layers):
        # Fallback to simple quarter-wave
        return ["L"], [centre_wl / (4.0 * nL)]
    
    return labels, final_thicknesses
# EVOLVE-BLOCK-END
