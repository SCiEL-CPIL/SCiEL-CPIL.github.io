"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _reflectance_spectrum(materials, thicknesses, n_layers, n_amb, n_sub, wavelengths):
    """
    Compute mean reflectance (fraction, 0..1) over a set of wavelengths using
    the characteristic matrix method for a multilayer stack.
    
    Parameters:
    - materials: list of refractive indices for each layer (length n_layers)
    - thicknesses: list of physical thicknesses in nm (length n_layers)
    - n_layers: number of layers
    - n_amb: ambient index
    - n_sub: substrate index
    - wavelengths: array of wavelengths in nm
    
    Returns:
    - mean reflectance as a fraction (0..1)
    """
    if n_layers == 0:
        # Bare substrate reflectance
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return r * r
    
    # Phase factors: delta_j = 2*pi*n_j*d_j / lambda
    # delta shape: (n_layers, n_wavelengths)
    delta = 2.0 * np.pi * np.outer(materials, thicknesses) / wavelengths  # (n_layers, n_wl)
    
    # cos and sin of delta
    cos_d = np.cos(delta)  # (n_layers, n_wl)
    sin_d = np.sin(delta)  # (n_layers, n_wl)
    
    # Matrix M for each layer: [[cos_d, i*sin_d/n], [i*n*sin_d, cos_d]]
    # We'll compute the product M_1 * M_2 * ... * M_n
    # Start with identity matrix
    # M_total shape: (2, 2, n_wavelengths)
    m11 = np.ones(len(wavelengths), dtype=complex)
    m12 = np.zeros(len(wavelengths), dtype=complex)
    m21 = np.zeros(len(wavelengths), dtype=complex)
    m22 = np.ones(len(wavelengths), dtype=complex)
    
    for j in range(n_layers):
        n_j = materials[j]
        c = cos_d[j]  # (n_wl,)
        s = sin_d[j]  # (n_wl,)
        # Layer matrix: [[c, i*s/n_j], [i*n_j*s, c]]
        # New M = M_old * M_layer
        # M_old = [[m11, m12], [m21, m22]]
        # M_layer = [[c, i*s/n_j], [i*n_j*s, c]]
        # New m11 = m11*c + m12*i*n_j*s
        # New m12 = m11*i*s/n_j + m12*c
        # New m21 = m21*c + m22*i*n_j*s
        # New m22 = m21*i*s/n_j + m22*c
        
        new_m11 = m11 * c + m12 * (1j * n_j * s)
        new_m12 = m11 * (1j * s / n_j) + m12 * c
        new_m21 = m21 * c + m22 * (1j * n_j * s)
        new_m22 = m21 * (1j * s / n_j) + m22 * c
        
        m11, m12, m21, m22 = new_m11, new_m12, new_m21, new_m22
    
    # Characteristic matrix Y = (m11 + m12*n_sub) / (m21 + m22*n_sub)
    # Actually, the admittance is:
    # Y = (m22 * n_sub + m21) / (m12 * n_sub + m11)  ... wait, let me be careful
    
    # For the characteristic matrix M = [[A, B], [C, D]], the input admittance is:
    # Y_in = (C + D * n_sub) / (A + B * n_sub)
    # where n_sub is the substrate index
    
    Y = (m21 + m22 * n_sub) / (m11 + m12 * n_sub)
    
    # Reflectance: R = |(n_amb - Y) / (n_amb + Y)|^2
    r = (n_amb - Y) / (n_amb + Y)
    R = np.abs(r) ** 2
    
    return np.mean(R)


def design(problem):
    """
    Design a broadband antireflection coating stack.
    Uses a grid search over layer configurations followed by local optimization.
    """
    deadline = time.time() + 55.0  # Safety margin
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    lam_min, lam_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation: 401 equally spaced points
    wavelengths = np.linspace(lam_min, lam_max, 401)
    
    # Pre-compute quarter-wave thicknesses for the center wavelength
    lam_center = 0.5 * (lam_min + lam_max)
    qw_L = lam_center / (4.0 * n_L)
    qw_H = lam_center / (4.0 * n_H)
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # Strategy: Try various numbers of layers and use a combination of
    # structured initialization and local optimization (Nelder-Mead or similar)
    
    # First, try a few well-known configurations as starting points
    # Then do a local search to refine
    
    def evaluate(materials_list, thicknesses_list):
        """Evaluate a design and return mean reflectance fraction."""
        if len(materials_list) == 0:
            r = (n_amb - n_sub) / (n_amb + n_sub)
            return r * r
        n_layers = len(materials_list)
        materials_arr = [n_L if m == 'L' else n_H for m in materials_list]
        thicknesses_arr = np.array(thicknesses_list, dtype=float)
        return _reflectance_spectrum(materials_arr, thicknesses_arr, n_layers, n_amb, n_sub, wavelengths)
    
    def local_optimize(materials_list, thicknesses_list, max_iter=50):
        """
        Simple local optimization using a coordinate descent / Nelder-Mead-like approach.
        """
        n_layers = len(materials_list)
        if n_layers == 0:
            return materials_list, thicknesses_list, evaluate(materials_list, thicknesses_list)
        
        current_thicknesses = list(thicknesses_list)
        current_score = evaluate(materials_list, current_thicknesses)
        
        # Use a simple gradient-free local search: perturb each thickness and keep improvement
        step_size = 10.0  # nm
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(n_layers):
                # Try increasing
                new_t = list(current_thicknesses)
                new_t[i] = min(max_layer_nm, new_t[i] + step_size)
                # Check total thickness constraint
                if sum(new_t) <= max_total_nm:
                    new_score = evaluate(materials_list, new_t)
                    if new_score < current_score - 1e-10:
                        current_thicknesses = new_t
                        current_score = new_score
                        improved = True
                # Try decreasing
                new_t = list(current_thicknesses)
                new_t[i] = max(0.0, new_t[i] - step_size)
                new_score = evaluate(materials_list, new_t)
                if new_score < current_score - 1e-10:
                    current_thicknesses = new_t
                    current_score = new_score
                    improved = True
            if not improved:
                step_size *= 0.5
                if step_size < 1.0:
                    break
        
        return materials_list, current_thicknesses, current_score
    
    # Try different numbers of layers
    for n_layers in range(1, min(max_layers + 1, 13)):
        if time.time() > deadline - 10.0:
            break
        
        # Generate several candidate configurations for this number of layers
        candidates = []
        
        # Candidate 1: Alternating L/H starting with L
        mats_alt = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        # Ensure total thickness is within limit
        # Distribute thickness roughly as quarter-wave
        total_qw = n_layers * lam_center / 4.0
        scale = 1.0
        if total_qw > max_total_nm:
            scale = max_total_nm / total_qw
        thicks_alt = [scale * (lam_center / (4.0 * (n_L if m == 'L' else n_H))) for m in mats_alt]
        candidates.append((mats_alt, thicks_alt))
        
        # Candidate 2: All L layers
        mats_allL = ['L'] * n_layers
        thicks_allL = [lam_center / (4.0 * n_L) * 0.8] * n_layers
        if sum(thicks_allL) > max_total_nm:
            scale = max_total_nm / sum(thicks_allL)
            thicks_allL = [t * scale for t in thicks_allL]
        candidates.append((mats_allL, thicks_allL))
        
        # Candidate 3: All H layers
        mats_allH = ['H'] * n_layers
        thicks_allH = [lam_center / (4.0 * n_H) * 0.8] * n_layers
        if sum(thicks_allH) > max_total_nm:
            scale = max_total_nm / sum(thicks_allH)
            thicks_allH = [t * scale for t in thicks_allH]
        candidates.append((mats_allH, thicks_allH))
        
        # Candidate 4: Random configuration
        mats_rand = ['L' if np.random.rand() < 0.5 else 'H' for _ in range(n_layers)]
        thicks_rand = [np.random.uniform(20, max_layer_nm) for _ in range(n_layers)]
        if sum(thicks_rand) > max_total_nm:
            scale = max_total_nm / sum(thicks_rand)
            thicks_rand = [t * scale for t in thicks_rand]
        candidates.append((mats_rand, thicks_rand))
        
        # Evaluate and locally optimize each candidate
        for mats, thicks in candidates:
            if time.time() > deadline - 5.0:
                break
            
            # Ensure constraints
            thicks = [max(0.0, min(max_layer_nm, t)) for t in thicks]
            if sum(thicks) > max_total_nm:
                scale = max_total_nm / sum(thicks)
                thicks = [t * scale for t in thicks]
            
            score = evaluate(mats, thicks)
            if score < best_score:
                best_score = score
                best_materials = mats[:]
                best_thicknesses = thicks[:]
            
            # Local optimization
            if time.time() > deadline - 8.0:
                break
            _, opt_thicks, opt_score = local_optimize(mats, thicks, max_iter=30)
            if opt_score < best_score:
                best_score = opt_score
                best_materials = mats[:]
                best_thicknesses = opt_thicks[:]
    
    # If no valid solution found, return a default
    if best_materials is None:
        return ["L"], [qw_L]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
