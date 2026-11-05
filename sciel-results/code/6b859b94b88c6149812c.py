"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _char_matrix(wavelengths, materials, thicknesses, ambient_index, substrate_index):
    """
    Compute reflectance using the characteristic matrix method.
    wavelengths: array of wavelengths in nm
    materials: list of refractive indices (complex or real)
    thicknesses: list of thicknesses in nm
    Returns: array of reflectance (0 to 1)
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        # Bare substrate
        n0 = ambient_index
        ns = substrate_index
        r = (n0 - ns) / (n0 + ns)
        return np.full_like(wavelengths, r**2, dtype=float)
    
    # k = 2*pi*n*d / lambda
    # For each wavelength, build the product of all layer matrices and the substrate matrix
    
    # Precompute for speed: we'll vectorize over wavelengths
    # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # where delta_j = 2*pi*n_j*d_j / lambda
    
    # Total matrix M = M_1 * M_2 * ... * M_N
    # Then Y = M[0,0] + i*M[0,1]  (wait, the standard formula is:)
    # Actually: [B, C]^T = M * [1, 1/n_s]^T
    # Y = B/C, and r = (n0 - Y)/(n0 + Y)
    
    # Let's use the standard characteristic matrix approach:
    # For s-polarization (TE) at normal incidence, it's the same for both.
    # M_total = product of all layer matrices
    # B = M_total[0,0] + M_total[0,1] * (1/n_s)
    # C = M_total[1,0] + M_total[1,1] * (1/n_s)
    # Y = B / C
    # r = (n0 - Y) / (n0 + Y)
    # R = |r|^2
    
    n0 = ambient_index
    ns = substrate_index
    
    # For efficiency, we'll compute this in a loop over wavelengths if needed,
    # but let's try to vectorize.
    
    # Actually, for up to 20 layers and 401 wavelengths, a loop over wavelengths
    # with vectorized matrix ops per wavelength should be fast enough.
    
    R = np.zeros(len(wavelengths), dtype=float)
    two_pi = 2.0 * np.pi
    
    for i, lam in enumerate(wavelengths):
        # Build total matrix, starting with identity
        M = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]])
        
        for j in range(n_layers):
            n_j = materials[j]
            d_j = thicknesses[j]
            delta_j = two_pi * n_j * d_j / lam
            
            cos_d = np.cos(delta_j)
            sin_d = np.sin(delta_j)
            
            # Layer matrix for normal incidence
            Mj = np.array([[cos_d, 1j * sin_d / n_j],
                          [1j * n_j * sin_d, cos_d]], dtype=complex)
            
            M = M @ Mj
        
        # Now multiply by substrate admittance
        # B = M[0,0] + M[0,1]/ns
        # C = M[1,0] + M[1,1]/ns
        B = M[0,0] + M[0,1] / ns
        C = M[1,0] + M[1,1] / ns
        
        Y = B / C
        r = (n0 - Y) / (n0 + Y)
        R[i] = np.abs(r)**2
    
    return R

def _mean_reflectance_percent(wavelengths, materials, thicknesses, ambient_index, substrate_index):
    """Compute mean reflectance in percent over the given wavelengths."""
    R = _char_matrix(wavelengths, materials, thicknesses, ambient_index, substrate_index)
    return np.mean(R) * 100.0

def _validate_and_score(materials_labels, thicknesses, problem):
    """Validate constraints and compute score."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials_labels) != len(thicknesses):
        return None
    if len(materials_labels) > max_layers:
        return None
    if any(t < -1e-9 or t > max_layer_nm + 1e-9 for t in thicknesses):
        return None
    if sum(thicknesses) > max_total_nm + 1e-9:
        return None
    
    # Clip small negative values
    thicknesses = [max(0.0, t) for t in thicknesses]
    
    mats = problem["materials"]
    n_list = [mats[label] for label in materials_labels]
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    
    score = _mean_reflectance_percent(wavelengths, n_list, thicknesses,
                                       problem["ambient_index"], problem["substrate_index"])
    return score

def _refine(materials_labels, thicknesses, problem, time_budget):
    """Local refinement using coordinate descent and Nelder-Mead."""
    start_time = time.time()
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    mats = problem["materials"]
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    
    n_list = [mats[label] for label in materials_labels]
    current_score = _mean_reflectance_percent(wavelengths, n_list, thicknesses, n0, ns)
    
    # Nelder-Mead on thicknesses
    def objective(x):
        x = np.clip(x, 0, max_layer_nm)
        if x.sum() > max_total_nm:
            return current_score + 1.0
        return _mean_reflectance_percent(wavelengths, n_list, x, n0, ns)
    
    x0 = np.array(thicknesses, dtype=float)
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 0.1, 'fatol': 1e-5,
                              'adaptive': True})
        if res.fun < current_score:
            thicknesses = [float(max(0, t)) for t in res.x]
            current_score = res.fun
    except Exception:
        pass
    
    # Try adding/removing layers
    for _ in range(10):
        if time.time() - start_time > time_budget:
            break
        
        best_new_score = current_score
        best_new_labels = None
        best_new_thick = None
        
        # Try removing a layer
        for i in range(len(materials_labels)):
            new_labels = materials_labels[:i] + materials_labels[i+1:]
            new_thick = thicknesses[:i] + thicknesses[i+1:]
            if not new_labels:
                new_thick = []
            s = _validate_and_score(new_labels, new_thick, problem)
            if s is not None and s < best_new_score:
                best_new_score = s
                best_new_labels = new_labels
                best_new_thick = new_thick
        
        # Try adding a layer at each position
        if len(materials_labels) < max_layers:
            for pos in range(len(materials_labels) + 1):
                for label in ['L', 'H']:
                    new_labels = materials_labels[:pos] + [label] + materials_labels[pos:]
                    new_thick = thicknesses[:pos] + [100.0] + thicknesses[pos:]
                    s = _validate_and_score(new_labels, new_thick, problem)
                    if s is not None and s < best_new_score:
                        best_new_score = s
                        best_new_labels = new_labels
                        best_new_thick = new_thick
        
        if best_new_labels is not None:
            materials_labels = best_new_labels
            thicknesses = best_new_thick
            current_score = best_new_score
            n_list = [mats[label] for label in materials_labels]
        else:
            break
    
    return materials_labels, thicknesses, current_score

def design(problem):
    """Design an antireflection coating using multi-start optimization."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    mats = problem["materials"]
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    
    # Bare glass baseline
    r_bare = (n0 - ns) / (n0 + ns)
    bare_score = r_bare**2 * 100.0
    
    best_labels = ["L"]
    best_thick = [band[0] / (4.0 * mats["L"])]
    best_score = _validate_and_score(best_labels, best_thick, problem)
    if best_score is None or best_score > bare_score:
        best_score = bare_score
        best_labels = []
        best_thick = []
    
    # Multi-start: try different initial configurations
    initial_configs = []
    
    # 1-layer designs
    for label in ['L', 'H']:
        for frac in [0.25, 0.33, 0.5, 0.75]:
            d = frac * (band[1] - band[0]) / mats[label]
            initial_configs.append(([label], [d]))
    
    # 2-layer designs (quarter-wave at center wavelength)
    centre = 0.5 * (band[0] + band[1])
    dL = centre / (4.0 * mats["L"])
    dH = centre / (4.0 * mats["H"])
    initial_configs.append((["L", "H"], [dL, dH]))
    initial_configs.append((["H", "L"], [dH, dL]))
    
    # 3-layer designs
    initial_configs.append((["L", "H", "L"], [dL, dH, dL]))
    initial_configs.append((["H", "L", "H"], [dH, dL, dH]))
    
    # 4-layer designs
    initial_configs.append((["L", "H", "L", "H"], [dL, dH, dL, dH]))
    initial_configs.append((["H", "L", "H", "L"], [dH, dL, dH, dL]))
    
    # 5-layer designs
    initial_configs.append((["L", "H", "L", "H", "L"], [dL, dH, dL, dH, dL]))
    initial_configs.append((["H", "L", "H", "L", "H"], [dH, dL, dH, dL, dH]))
    
    # Random starts
    np.random.seed(42)
    for _ in range(20):
        n_layers = np.random.randint(1, min(8, max_layers + 1))
        labels = list(np.random.choice(['L', 'H'], size=n_layers))
        thicks = list(np.random.uniform(10, min(200, max_layer_nm), size=n_layers))
        # Ensure total doesn't exceed max
        total = sum(thicks)
        if total > max_total_nm:
            scale = max_total_nm / total
            thicks = [t * scale for t in thicks]
        initial_configs.append((labels, thicks))
    
    # Evaluate and refine each initial config
    for labels, thicks in initial_configs:
        if time.time() > deadline:
            break
        
        # Quick validation
        s = _validate_and_score(labels, thicks, problem)
        if s is None:
            continue
        
        if s < best_score:
            best_score = s
            best_labels = labels[:]
            best_thick = thicks[:]
        
        # Refine if promising
        if s < best_score + 2.0:  # Within 2% of best
            time_left = deadline - time.time()
            if time_left > 1.0:
                refine_budget = min(5.0, time_left * 0.5)
                new_labels, new_thicks, new_score = _refine(
                    labels[:], thicks[:], problem, refine_budget)
                if new_score < best_score:
                    best_score = new_score
                    best_labels = new_labels
                    best_thick = new_thicks
    
    # Final refinement of the best solution
    time_left = deadline - time.time()
    if time_left > 1.0 and best_labels:
        refine_budget = min(10.0, time_left * 0.8)
        final_labels, final_thicks, final_score = _refine(
            best_labels[:], best_thick[:], problem, refine_budget)
        if final_score < best_score:
            best_score = final_score
            best_labels = final_labels
            best_thick = final_thicks
    
    # Final validation
    final_score = _validate_and_score(best_labels, best_thick, problem)
    if final_score is None:
        return ["L"], [band[0] / (4.0 * mats["L"])]
    
    return best_labels, best_thick
# EVOLVE-BLOCK-END
