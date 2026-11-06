"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _char_matrix(n, d, wl):
    """Characteristic matrix for a single layer of index n, thickness d (nm) at wavelength wl (nm)."""
    beta = 2.0 * np.pi * n * d / wl
    c = np.cos(beta)
    s = np.sin(beta)
    return np.array([[c, 1j * s / n], [1j * n * s, c]], dtype=complex)

def _reflectance(materials, thicknesses, wl, n0, ns):
    """
    Compute reflectance at wavelength wl using the characteristic-matrix method.
    materials: list of 'L'/'H'
    thicknesses: list of physical thicknesses in nm
    n0: ambient index
    ns: substrate index
    Returns reflectance as a fraction (0 to 1).
    """
    if len(materials) == 0:
        r = (n0 - ns) / (n0 + ns)
        return abs(r) ** 2
    
    # Start with the substrate admittance
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    
    # Multiply layer matrices from air side to substrate
    for i, mat in enumerate(materials):
        n = n_mat[mat]
        d = thicknesses[i]
        Mi = _char_matrix(n, d, wl)
        M = M @ Mi
    
    # The input admittance Y_in = (M[0,1]) / (M[1,1]) * n0? No.
    # Standard formula:
    # Y_in = (M11 + M12 * Y_s) / (M21 + M22 * Y_s)
    # where Y_s = n_s (for normal incidence, admittance = n)
    # And the reflectance R = |(Y_in - n0) / (Y_in + n0)|^2
    
    Y_s = ns
    Y_in = (M[0, 0] + M[0, 1] * Y_s) / (M[1, 0] + M[1, 1] * Y_s)
    r = (Y_in - n0) / (Y_in + n0)
    return abs(r) ** 2

def _mean_reflectance(materials, thicknesses, problem):
    """
    Compute the mean reflectance over the band.
    Returns reflectance in percent.
    """
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    wls = np.linspace(band[0], band[1], 401)
    
    total_r = 0.0
    for wl in wls:
        r = _reflectance(materials, thicknesses, wl, n0, ns)
        total_r += r
    
    return (total_r / len(wls)) * 100.0

def _optimize_thicknesses(materials, problem, initial_thicknesses=None):
    """
    Local optimization of thicknesses using coordinate descent / Nelder-Mead.
    """
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    n_layers = len(materials)
    if n_layers == 0:
        return [], 0.0
    
    if initial_thicknesses is None:
        # Initialize at quarter-wave for center wavelength
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        thicknesses = []
        for mat in materials:
            n = n_mat[mat]
            thicknesses.append(centre / (4.0 * n))
    else:
        thicknesses = list(initial_thicknesses)
    
    # Ensure constraints
    for i in range(n_layers):
        if thicknesses[i] < 0:
            thicknesses[i] = 0
        if thicknesses[i] > max_layer_nm:
            thicknesses[i] = max_layer_nm
    
    # Coordinate descent
    best_thicknesses = list(thicknesses)
    best_score = _mean_reflectance(materials, best_thicknesses, problem)
    
    # Simple Nelder-Mead or coordinate descent
    # Let's use a simple gradient-free local search: coordinate descent with adaptive step
    step_size = 5.0  # nm
    for _ in range(200):  # max iterations
        improved = False
        for i in range(n_layers):
            # Try decreasing
            for delta in [-step_size, -step_size/2, -step_size/4]:
                new_t = thicknesses[i] + delta
                if new_t < 0:
                    new_t = 0
                if new_t > max_layer_nm:
                    continue
                # Check total thickness
                if sum(thicknesses[:i]) + new_t + sum(thicknesses[i+1:]) > max_total_nm:
                    continue
                
                old_t = thicknesses[i]
                thicknesses[i] = new_t
                score = _mean_reflectance(materials, thicknesses, problem)
                if score < best_score:
                    best_score = score
                    best_thicknesses = list(thicknesses)
                    improved = True
                else:
                    thicknesses[i] = old_t
                    break
            # Try increasing
            for delta in [step_size, step_size/2, step_size/4]:
                new_t = thicknesses[i] + delta
                if new_t < 0:
                    continue
                if new_t > max_layer_nm:
                    continue
                # Check total thickness
                if sum(thicknesses[:i]) + new_t + sum(thicknesses[i+1:]) > max_total_nm:
                    continue
                
                old_t = thicknesses[i]
                thicknesses[i] = new_t
                score = _mean_reflectance(materials, thicknesses, problem)
                if score < best_score:
                    best_score = score
                    best_thicknesses = list(thicknesses)
                    improved = True
                else:
                    thicknesses[i] = old_t
                    break
        
        if not improved:
            # Reduce step size
            step_size *= 0.5
            if step_size < 0.1:
                break
    
    return best_thicknesses, best_score

def design(problem):
    """
    Design a stack of at most twenty low- and high-index layers (1.38 and 2.35) on glass
    that minimizes the mean reflectance between 400 and 800 nm at normal incidence.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Global best
    best_materials = ["L"]
    best_thicknesses = [0.5 * (band[0] + band[1]) / (4.0 * n_L)]
    best_score = _mean_reflectance(best_materials, best_thicknesses, problem)
    
    # Try different stack configurations
    # Classic AR coatings:
    # 1. Single L layer
    # 2. LH or HL pair
    # 3. LHL or HLH triple
    # 4. LHHL or HLHL quadruple
    # 5. Up to 5-6 layers
    
    # Generate candidate stacks
    candidates = []
    
    # Single layer
    candidates.append(["L"])
    candidates.append(["H"])
    
    # Two layers
    candidates.append(["L", "H"])
    candidates.append(["H", "L"])
    
    # Three layers
    candidates.append(["L", "H", "L"])
    candidates.append(["H", "L", "H"])
    
    # Four layers
    candidates.append(["L", "H", "L", "H"])
    candidates.append(["H", "L", "H", "L"])
    candidates.append(["L", "H", "H", "L"])
    candidates.append(["H", "L", "L", "H"])
    
    # Five layers
    candidates.append(["L", "H", "L", "H", "L"])
    candidates.append(["H", "L", "H", "L", "H"])
    candidates.append(["L", "H", "L", "L", "H"])
    candidates.append(["H", "L", "H", "H", "L"])
    candidates.append(["L", "L", "H", "L", "H"])
    candidates.append(["H", "H", "L", "H", "L"])
    
    # Six layers
    candidates.append(["L", "H", "L", "H", "L", "H"])
    candidates.append(["H", "L", "H", "L", "H", "L"])
    candidates.append(["L", "H", "H", "L", "H", "L"])
    candidates.append(["H", "L", "L", "H", "L", "H"])
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for cand in candidates:
        key = tuple(cand)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)
    
    # Evaluate each candidate with local optimization
    for materials in unique_candidates:
        if time.time() > deadline:
            break
        
        if len(materials) > max_layers:
            continue
        
        # Check if a quarter-wave stack fits within total thickness
        centre = 0.5 * (band[0] + band[1])
        total_qw = sum(centre / (4.0 * n_mat[m]) for m in materials)
        if total_qw > max_total_nm:
            # Scale down
            scale = max_total_nm / total_qw
            initial_t = [centre / (4.0 * n_mat[m]) * scale for m in materials]
        else:
            initial_t = [centre / (4.0 * n_mat[m]) for m in materials]
        
        # Also try a few other initializations
        inits = [initial_t]
        
        # Try with slightly different center wavelengths
        for frac in [0.4, 0.6]:
            wl_c = band[0] + frac * (band[1] - band[0])
            t_alt = [wl_c / (4.0 * n_mat[m]) for m in materials]
            total_alt = sum(t_alt)
            if total_alt <= max_total_nm:
                inits.append(t_alt)
        
        for init_t in inits:
            if time.time() > deadline:
                break
            
            # Clip to valid range
            init_t_clipped = [max(0, min(max_layer_nm, t)) for t in init_t]
            
            # Check total
            if sum(init_t_clipped) > max_total_nm:
                scale = max_total_nm / sum(init_t_clipped)
                init_t_clipped = [t * scale for t in init_t_clipped]
            
            # Local optimization
            thicknesses, score = _optimize_thicknesses(materials, problem, init_t_clipped)
            
            # Verify constraints
            if len(thicknesses) != len(materials):
                continue
            if any(t < -1e-6 or t > max_layer_nm + 1e-6 for t in thicknesses):
                continue
            if sum(thicknesses) > max_total_nm + 1e-6:
                continue
            
            if score < best_score:
                best_score = score
                best_materials = list(materials)
                best_thicknesses = list(thicknesses)
    
    # Final polish: try more iterations with the best configuration
    if time.time() < deadline - 10:
        # Try more aggressive optimization on the best
        thicknesses, score = _optimize_thicknesses(best_materials, problem, best_thicknesses)
        if score < best_score:
            best_score = score
            best_thicknesses = list(thicknesses)
    
    # Ensure final result is valid
    # Clip thicknesses
    final_thicknesses = [max(0.0, min(max_layer_nm, t)) for t in best_thicknesses]
    
    # Check total
    total = sum(final_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        final_thicknesses = [t * scale for t in final_thicknesses]
    
    # Check number of layers
    if len(best_materials) > max_layers:
        # Truncate (shouldn't happen)
        best_materials = best_materials[:max_layers]
        final_thicknesses = final_thicknesses[:max_layers]
    
    return best_materials, final_thicknesses

# EVOLVE-BLOCK-END
