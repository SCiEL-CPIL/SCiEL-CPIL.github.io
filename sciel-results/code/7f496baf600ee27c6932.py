"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _reflectance_stack(n_layers, thicknesses, wavelengths, n_ambient, n_sub, n_L, n_H):
    """Compute reflectance (0-1) for a stack using characteristic-matrix method."""
    # n_layers: list of 'L'/'H'
    # thicknesses: list of physical thicknesses in nm
    # wavelengths: array of wavelengths in nm
    # Returns: array of reflectance values (0-1)
    
    # Build the layer refractive indices
    n_layers_arr = np.array([n_L if x == 'L' else n_H for x in n_layers])
    thicknesses_arr = np.array(thicknesses, dtype=float)
    
    # Optical thicknesses: n * d
    # Phase: delta = 2*pi*n*d/lambda
    # For each wavelength, compute the characteristic matrix
    
    # Vectorized: wavelengths shape (W,), layers shape (N,)
    # delta[i, j] = 2*pi * n_i * d_j / lambda_i
    # where i = wavelength index, j = layer index
    
    W = len(wavelengths)
    N = len(n_layers)
    
    if N == 0:
        # Bare interface
        r = (n_ambient - n_sub) / (n_ambient + n_sub)
        return np.full(W, r * r)
    
    # Phase for each (wavelength, layer)
    # wavelengths shape (W, 1), n_layers_arr shape (1, N), thicknesses_arr shape (1, N)
    delta = 2.0 * np.pi * n_layers_arr[None, :] * thicknesses_arr[None, :] / wavelengths[:, None]
    
    # cos and sin of delta
    cos_d = np.cos(delta)  # shape (W, N)
    sin_d = np.sin(delta)  # shape (W, N)
    
    # Build characteristic matrices
    # For layer j: M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # We multiply all matrices: M_total = M_1 * M_2 * ... * M_N
    # Then Y = M_total[1,1] / M_total[0,1] ... wait, let me be careful.
    
    # Standard formulation:
    # M_j = [[cos(delta_j), (i/n_j)*sin(delta_j)], [i*n_j*sin(delta_j), cos(delta_j)]]
    # M_total = M_1 * M_2 * ... * M_N
    # Y = M_total[1,0] / M_total[0,0]  ... no
    
    # Let me use the standard thin-film formula:
    # The characteristic matrix for a layer with refractive index n and phase delta:
    # B = [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # Total matrix M = B_1 * B_2 * ... * B_N
    # Y = M[1,0] / M[0,0]  (admittance)
    # Wait, I need to be more careful.
    
    # Standard: For normal incidence, the characteristic matrix of layer j is:
    # M_j = [[cos(delta_j), (i/n_j)sin(delta_j)], [i*n_j*sin(delta_j), cos(delta_j)]]
    # M = M_1 M_2 ... M_N
    # Y = M[1,0]/M[0,0]
    # R = |(Y - n_0)/(Y + n_0)|^2 where n_0 is the ambient index... 
    # Actually: R = |(n_0*Y - n_s)/(n_0*Y + n_s)|^2? No.
    
    # Let me use the well-known formula:
    # The admittance at the entrance is Y = M[1,0]/M[0,0]
    # Reflectance R = |(Y/n_0 - 1)/(Y/n_0 + 1)|^2 = |(Y - n_0)/(Y + n_0)|^2
    # where n_0 is the ambient index.
    
    # Initialize M as identity for each wavelength
    M = np.zeros((W, 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for j in range(N):
        n_j = n_layers_arr[j]
        c = cos_d[:, j]  # shape (W,)
        s = sin_d[:, j]  # shape (W,)
        
        # B_j matrix elements
        B00 = c
        B01 = 1j * s / n_j
        B10 = 1j * n_j * s
        B11 = c
        
        # M = M * B_j (matrix multiplication, 2x2)
        new_M00 = M[:, 0, 0] * B00 + M[:, 0, 1] * B10
        new_M01 = M[:, 0, 0] * B01 + M[:, 0, 1] * B11
        new_M10 = M[:, 1, 0] * B00 + M[:, 1, 1] * B10
        new_M11 = M[:, 1, 0] * B01 + M[:, 1, 1] * B11
        
        M[:, 0, 0] = new_M00
        M[:, 0, 1] = new_M01
        M[:, 1, 0] = new_M10
        M[:, 1, 1] = new_M11
    
    # Admittance Y = M[1,0] / M[0,0]
    Y = M[:, 1, 0] / M[:, 0, 0]
    
    # Reflectance R = |(Y - n_ambient)/(Y + n_ambient)|^2
    R = np.abs((Y - n_ambient) / (Y + n_ambient)) ** 2
    
    return R


def _mean_reflectance_percent(n_layers, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    wavelengths = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    n_ambient = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    R = _reflectance_stack(n_layers, thicknesses, wavelengths, n_ambient, n_sub, n_L, n_H)
    return np.mean(R) * 100.0


def _is_valid(n_layers, thicknesses, problem):
    """Check if the design is valid."""
    if len(n_layers) > problem["max_layers"]:
        return False
    if len(n_layers) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0 or t > problem["max_layer_nm"]:
            return False
    if sum(thicknesses) > problem["max_total_nm"]:
        return False
    return True


def _objective(x, n_labels, problem):
    """Objective function for optimization. x contains thicknesses."""
    thicknesses = list(x)
    if not _is_valid(n_labels, thicknesses, problem):
        return 1e10
    return _mean_reflectance_percent(n_labels, thicknesses, problem)


def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    n_ambient = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Helper to compute mean reflectance quickly
    def mean_ref(n_labels, thicknesses):
        if not _is_valid(n_labels, thicknesses, problem):
            return 1e10
        R = _reflectance_stack(n_labels, thicknesses, wavelengths, n_ambient, n_sub, n_L, n_H)
        return np.mean(R) * 100.0
    
    # Start with a good initial design: alternating L and H layers
    # Quarter-wave at center wavelength
    centre = 0.5 * (band[0] + band[1])
    qL = centre / (4.0 * n_L)
    qH = centre / (4.0 * n_H)
    
    # Try several numbers of layers and find the best
    best_labels = None
    best_thicknesses = None
    best_score = 1e10
    
    # First, try some structured designs
    for n_layers in range(1, min(max_layers, 12) + 1):
        if time.time() > deadline - 30:
            break
        
        # Alternating starting with L
        labels = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        thicknesses = [qL if i % 2 == 0 else qH for i in range(n_layers)]
        
        # Check constraints
        if sum(thicknesses) > max_total_nm:
            # Scale down
            scale = max_total_nm / sum(thicknesses)
            thicknesses = [t * scale for t in thicknesses]
        
        score = mean_ref(labels, thicknesses)
        if score < best_score:
            best_score = score
            best_labels = labels[:]
            best_thicknesses = thicknesses[:]
        
        # Try starting with H
        labels2 = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        thicknesses2 = [qH if i % 2 == 0 else qL for i in range(n_layers)]
        
        if sum(thicknesses2) > max_total_nm:
            scale = max_total_nm / sum(thicknesses2)
            thicknesses2 = [t * scale for t in thicknesses2]
        
        score2 = mean_ref(labels2, thicknesses2)
        if score2 < best_score:
            best_score = score2
            best_labels = labels2[:]
            best_thicknesses = thicknesses2[:]
    
    # Now do local optimization on the best candidate
    if best_labels is not None and time.time() < deadline - 20:
        n_opt = len(best_labels)
        x0 = np.array(best_thicknesses, dtype=float)
        
        # Nelder-Mead optimization
        try:
            res = minimize(
                _objective,
                x0,
                args=(best_labels, problem),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 0.1, 'fatol': 1e-4, 'adaptive': True}
            )
            if time.time() < deadline - 10:
                if _is_valid(best_labels, list(res.x), problem):
                    score = _mean_reflectance_percent(best_labels, list(res.x), problem)
                    if score < best_score:
                        best_score = score
                        best_thicknesses = list(res.x)
        except Exception:
            pass
    
    # Try a few random restarts if time permits
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            
            # Random number of layers
            n_rand = rng.integers(3, min(max_layers, 10) + 1)
            rand_labels = [rng.choice(['L', 'H']) for _ in range(n_rand)]
            
            # Random thicknesses
            rand_thicknesses = list(rng.uniform(20, max_layer_nm, size=n_rand))
            
            # Scale to fit total thickness constraint
            total = sum(rand_thicknesses)
            if total > max_total_nm:
                scale = max_total_nm / total
                rand_thicknesses = [t * scale for t in rand_thicknesses]
            
            score = mean_ref(rand_labels, rand_thicknesses)
            
            if score < best_score and time.time() < deadline - 15:
                # Refine this candidate
                try:
                    res = minimize(
                        _objective,
                        np.array(rand_thicknesses, dtype=float),
                        args=(rand_labels, problem),
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 0.1, 'fatol': 1e-4, 'adaptive': True}
                    )
                    if _is_valid(rand_labels, list(res.x), problem):
                        score2 = _mean_reflectance_percent(rand_labels, list(res.x), problem)
                        if score2 < best_score:
                            best_score = score2
                            best_labels = rand_labels[:]
                            best_thicknesses = list(res.x)
                except Exception:
                    if score < best_score:
                        best_score = score
                        best_labels = rand_labels[:]
                        best_thicknesses = rand_thicknesses[:]
    
    # Ensure we have a valid result
    if best_labels is None or not _is_valid(best_labels, best_thicknesses, problem):
        # Fallback: single quarter-wave L layer
        best_labels = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    # Final validation
    if not _is_valid(best_labels, best_thicknesses, problem):
        # Force validity
        best_labels = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
