"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START

def _compute_reflectance(materials, thicknesses, wavelengths, ambient_index, substrate_index):
    """Compute normal-incidence reflectance using characteristic matrix method.
    
    Parameters:
    -----------
    materials : list of str
        'L' or 'H' for each layer
    thicknesses : list of float
        Physical thicknesses in nm
    wavelengths : array-like
        Wavelengths in nm
    ambient_index : float
        Refractive index of ambient (air)
    substrate_index : float
        Refractive index of substrate (glass)
    
    Returns:
    --------
    reflectance : ndarray
        Reflectance (0 to 1) at each wavelength
    """
    n_ambient = ambient_index
    n_substrate = substrate_index
    n_layers = len(materials)
    
    # Refractive indices of each layer
    n_array = np.array([1.0 if m == 'L' else 2.35 for m in materials])  # placeholder, will use actual values
    
    # Actually, we need to pass the actual refractive indices
    # Let's restructure - this function should take n_array directly
    raise NotImplementedError("Use _compute_reflectance_with_n instead")


def _compute_reflectance_with_n(n_array, thicknesses, wavelengths, n_ambient, n_substrate):
    """Compute normal-incidence reflectance using characteristic matrix method.
    
    Parameters:
    -----------
    n_array : ndarray
        Refractive indices of each layer (length = number of layers)
    thicknesses : ndarray
        Physical thicknesses in nm
    wavelengths : ndarray
        Wavelengths in nm
    n_ambient : float
        Refractive index of ambient
    n_substrate : float
        Refractive index of substrate
    
    Returns:
    --------
    reflectance : ndarray
        Reflectance (0 to 1) at each wavelength
    """
    n_wl = len(wavelengths)
    n_layers = len(n_array)
    
    reflectance = np.zeros(n_wl)
    
    for i_wl in range(n_wl):
        lam = wavelengths[i_wl]
        
        # Optical thicknesses (phase)
        # d_j * n_j / lambda (in radians, 2*pi factor included)
        delta = 2.0 * np.pi * n_array * thicknesses / lam
        
        # Characteristic matrix for each layer
        # M_j = [[cos(delta_j), i*sin(delta_j)/n_j],
        #        [i*n_j*sin(delta_j), cos(delta_j)]]
        
        # Start with identity matrix
        M = np.array([[1.0 + 0j, 0.0 + 0j],
                      [0.0 + 0j, 1.0 + 0j]])
        
        for j in range(n_layers):
            c = np.cos(delta[j])
            s = np.sin(delta[j])
            
            # Layer matrix
            M_j = np.array([[c, 1j * s / n_array[j]],
                            [1j * n_array[j] * s, c]])
            
            M = M @ M_j
        
        # Admittance of the stack
        # B = M[0,0], C = M[1,0]
        B = M[0, 0]
        C = M[1, 0]
        
        # Input admittance Y = n_ambient * C / B
        Y = n_ambient * C / B
        
        # Reflectance
        # r = (Y - n_ambient) / (Y + n_ambient)  -- wait, this isn't right
        # Actually: r = (n_ambient * B - C) / (n_ambient * B + C)
        # Let me use the standard formula:
        # Y_in = C / B (admittance)
        # r = (Y_in - n_ambient) / (Y_in + n_ambient)
        
        Y_in = C / B
        r = (Y_in - n_ambient) / (Y_in + n_ambient)
        reflectance[i_wl] = np.abs(r) ** 2
    
    return reflectance


def _mean_reflectance_pct(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    n_array = np.array([problem["materials"][m] for m in materials])
    thicknesses_arr = np.array(thicknesses, dtype=float)
    
    refl = _compute_reflectance_with_n(
        n_array, thicknesses_arr, wavelengths,
        problem["ambient_index"], problem["substrate_index"]
    )
    
    return 100.0 * np.mean(refl)


def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    if len(materials) > problem["max_layers"]:
        return False
    if len(materials) != len(thicknesses):
        return False
    
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    
    total = 0.0
    for m, d in zip(materials, thicknesses):
        if m not in ('L', 'H'):
            return False
        if d < -1e-6 or d > max_layer + 1e-6:
            return False
        total += d
    
    if total > max_total + 1e-6:
        return False
    
    return True


def design(problem):
    """Design an antireflection coating stack.
    
    Uses a multi-stage approach:
    1. Start with a good initial guess (quarter-wave stack)
    2. Use differential evolution to search over layer counts and thicknesses
    3. Refine with local optimization
    """
    deadline = time.time() + 120.0  # 120s wall-clock budget
    
    max_layers = problem["max_layers"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    band = problem["band_nm"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    
    # Wavelengths for evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Precompute for speed
    def _reflectance_from_params(params):
        """params: [n_layers, label_0, ..., label_{n-1}, th_0, ..., th_{n-1}]
        labels are 0 for L, 1 for H (encoded as 0 or 1 in continuous space)
        """
        n_layers = int(round(params[0]))
        n_layers = max(1, min(n_layers, max_layers))
        
        labels = params[1:1+n_layers]
        thicknesses = params[1+n_layers:1+2*n_layers]
        
        # Convert labels to material indices
        n_array = np.where(labels < 0.5, n_L, n_H)
        
        # Clip thicknesses
        thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
        
        # Check total thickness
        if np.sum(thicknesses) > max_total_nm:
            return 100.0  # penalty
        
        refl = _compute_reflectance_with_n(
            n_array, thicknesses, wavelengths, n_amb, n_sub
        )
        return 100.0 * np.mean(refl)
    
    # Build parameter vector for a given number of layers
    def _build_params(n_layers, labels, thicknesses):
        """Build a full parameter vector for optimization."""
        labels_enc = np.array([0.0 if m == 'L' else 1.0 for m in labels])
        return np.concatenate([[float(n_layers)], labels_enc, thicknesses])
    
    def _decode_params(params):
        """Decode a parameter vector into (materials, thicknesses)."""
        n_layers = int(round(params[0]))
        n_layers = max(1, min(n_layers, max_layers))
        
        labels_enc = params[1:1+n_layers]
        thicknesses = params[1+n_layers:1+2*n_layers]
        
        materials = ['L' if l < 0.5 else 'H' for l in labels_enc]
        thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
        
        return materials, thicknesses.tolist()
    
    # Objective function for differential evolution
    def objective(params):
        return _reflectance_from_params(params)
    
    # Define bounds for a fixed number of layers
    def _make_bounds(n_layers):
        bounds = [(1.0, float(n_layers))]  # n_layers itself
        for _ in range(n_layers):
            bounds.append((0.0, 1.0))  # label
        for _ in range(n_layers):
            bounds.append((0.0, max_layer_nm))  # thickness
        return bounds
    
    # Try several different numbers of layers and find the best
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Initial guess: quarter-wave stack
    center_wl = 0.5 * (band[0] + band[1])
    q_L = center_wl / (4.0 * n_L)
    q_H = center_wl / (4.0 * n_H)
    
    # Try a few hand-crafted designs first
    candidate_designs = [
        # Single quarter-wave L
        (['L'], [q_L]),
        # Single quarter-wave H
        (['H'], [q_H]),
        # LH quarter-wave pair
        (['L', 'H'], [q_L, q_H]),
        # HL quarter-wave pair
        (['H', 'L'], [q_H, q_L]),
        # LHL triple
        (['L', 'H', 'L'], [q_L, q_H, q_L]),
        # HLH triple
        (['H', 'L', 'H'], [q_H, q_L, q_H]),
    ]
    
    for mats, ths in candidate_designs:
        if _is_valid(mats, ths, problem):
            score = _mean_reflectance_pct(mats, ths, problem)
            if score < best_score:
                best_score = score
                best_materials = mats[:]
                best_thicknesses = ths[:]
    
    # Now use differential evolution for a few layer counts
    for n_layers in [2, 3, 4, 5, 6, 8, 10]:
        if time.time() > deadline - 30:
            break
        
        bounds = _make_bounds(n_layers)
        
        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=50,
                tol=1e-6,
                popsize=15,
                mutation=(0.3, 0.8),
                recombination=0.7,
                init='latinhypercube',
                polish=True,
                update_interval=10
            )
            
            if result.success and result.fun < best_score:
                mats, ths = _decode_params(result.x)
                if _is_valid(mats, ths, problem):
                    best_score = result.fun
                    best_materials = mats
                    best_thicknesses = ths
                    
        except Exception:
            pass
    
    # Refine the best design with local optimization
    if best_materials is not None and time.time() < deadline - 15:
        n_best = len(best_materials)
        # Optimize only thicknesses, keep labels fixed
        def obj_thick(thick_params):
            thicknesses = np.clip(thick_params, 0.0, max_layer_nm)
            if np.sum(thicknesses) > max_total_nm:
                return 100.0
            n_array = np.array([problem["materials"][m] for m in best_materials])
            refl = _compute_reflectance_with_n(
                n_array, thicknesses, wavelengths, n_amb, n_sub
            )
            return 100.0 * np.mean(refl)
        
        x0 = np.array(best_thicknesses, dtype=float)
        try:
            res = minimize(
                obj_thick, x0,
                method='L-BFGS-B',
                bounds=[(0.0, max_layer_nm)] * n_best,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if res.fun < best_score:
                thicknesses_opt = np.clip(res.x, 0.0, max_layer_nm)
                if np.sum(thicknesses_opt) <= max_total_nm + 1e-6:
                    best_score = res.fun
                    best_thicknesses = thicknesses_opt.tolist()
        except Exception:
            pass
    
    # Final validation
    if best_materials is None:
        # Fallback: single quarter-wave L
        best_materials = ['L']
        best_thicknesses = [q_L]
    
    # Ensure all thicknesses are within bounds
    best_thicknesses = [max(0.0, min(max_layer_nm, t)) for t in best_thicknesses]
    
    # Final check
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ['L']
        best_thicknesses = [q_L]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
