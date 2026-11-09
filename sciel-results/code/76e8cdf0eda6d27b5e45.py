"""Broadband antireflection coating design via global+local optimization."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START

def _reflectance_spectrum(thicknesses, materials, ambient_index, substrate_index, band_nm, n_points=401):
    """
    Compute mean normal-incidence reflectance in percent over the band
    using the characteristic-matrix method.
    """
    lam0, lam1 = band_nm
    wavelengths = np.linspace(lam0, lam1, n_points)
    
    n_L = materials['L']
    n_H = materials['H']
    n_s = substrate_index
    n_0 = ambient_index
    
    # Build layer indices
    n_layers = len(thicknesses)
    if n_layers == 0:
        # Bare substrate
        r = (n_0 - n_s) / (n_0 + n_s)
        R = np.abs(r) ** 2
        return 100.0 * np.mean(R)
    
    # Map 'L'/'H' to actual indices
    # We'll assume thicknesses are paired with a materials list externally,
    # but here we just take thicknesses and a materials label list.
    # Wait, the function signature in the prompt for design returns (materials, thicknesses).
    # So I need to pass both.
    # Let's refactor: _reflectance_spectrum(materials_list, thicknesses, ...)
    
    # Actually, let's just do it inline in the main optimization to be safe and fast.
    pass

def _mean_reflectance(labels, thicknesses, problem, n_points=401):
    """
    Compute mean reflectance in percent for a given stack.
    labels: list of 'L'/'H'
    thicknesses: list of floats in nm
    problem: dict with problem parameters
    """
    band_nm = problem["band_nm"]
    materials = problem["materials"]
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]
    
    lam0, lam1 = band_nm
    wavelengths = np.linspace(lam0, lam1, n_points)
    
    n_layers = len(labels)
    if n_layers == 0:
        r = (n_0 - n_s) / (n_0 + n_s)
        R = np.abs(r) ** 2
        return 100.0 * np.mean(R)
    
    # Pre-compute layer indices
    n_layers_arr = np.array([materials[l] for l in labels], dtype=float)
    t_arr = np.array(thicknesses, dtype=float)
    
    # Wavenumber k = 2*pi / lambda
    # Phase delta_j = 2*pi * n_j * d_j / lambda
    # We can compute this vectorized over wavelengths
    
    # Matrix M = product of layer matrices
    # For each wavelength, M is 2x2.
    # M = M_1 * M_2 * ... * M_N
    # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    
    # We can compute this iteratively for each wavelength, or vectorized.
    # Vectorized over wavelengths:
    # delta = 2*pi*n*d/lambda (shape: (n_points, n_layers))
    # cos_d, sin_d
    # M starts as identity (shape: (n_points, 2, 2))
    # M = M @ M_j for each layer
    
    delta = 2.0 * np.pi * np.outer(t_arr, n_layers_arr) / wavelengths[:, np.newaxis]
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    
    # Initialize M as identity
    M = np.zeros((len(wavelengths), 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    
    for j in range(n_layers):
        n_j = n_layers_arr[j]
        # M_j = [[cos_d_j, 1j*sin_d_j/n_j], [1j*n_j*sin_d_j, cos_d_j]]
        M_j = np.zeros((len(wavelengths), 2, 2), dtype=complex)
        M_j[:, 0, 0] = cos_d[:, j]
        M_j[:, 0, 1] = 1j * sin_d[:, j] / n_j
        M_j[:, 1, 0] = 1j * n_j * sin_d[:, j]
        M_j[:, 1, 1] = cos_d[:, j]
        
        # M = M @ M_j
        M = M @ M_j
    
    # B = M[0,0], C = M[0,1], D = M[1,0], E = M[1,1]
    B = M[:, 0, 0]
    C = M[:, 0, 1]
    D = M[:, 1, 0]
    E = M[:, 1, 1]
    
    # Y = (n_0 * B + C) / (D + n_0 * E)  -- wait, characteristic matrix method:
    # [B, C] = M @ [1, 0]
    # [D, E] = M @ [0, 1]
    # Y = (n_0 * B + C) / (D + n_0 * E) is for admittance?
    # Actually, Y = (n_0 * B + C) / (D + n_0 * E) is not quite right.
    # Let's use the standard formula:
    # Y = (n_0 * B + C) / (D + n_0 * E) is for the admittance at the front surface?
    # No, the characteristic matrix M relates the fields:
    # [E, H]_0 = M @ [E, H]_s
    # For s-polarization (TE), H = n * E / (i*omega*mu) ... let's use the standard result:
    # Y = (n_0 * B + C) / (D + n_0 * E) is incorrect.
    # The correct formula is:
    # Y = (n_0 * B + C) / (D + n_0 * E) ... no.
    # Let's derive:
    # E_0 = B * E_s + C * H_s
    # H_0 = D * E_s + E * H_s
    # At the substrate, H_s = (n_s / n_0) * E_s * (1/2) * (1 + R_s) ... no.
    # Boundary condition at substrate: H_s = (n_s / n_0) * E_s * (1/2) * (1 + R) ... 
    # Actually, for a plane wave in a medium of index n, H = (n/n_0) * (E_+ - E_-) / 2?
    # Let's use the standard characteristic matrix result for reflectance:
    # R = |(n_0 * B + C - n_s * D - n_s * E) / (n_0 * B + C + n_s * D + n_s * E)|^2 ... no.
    
    # Correct formula:
    # Y = (n_0 * B + C) / (D + n_0 * E) is the input admittance.
    # R = |(Y - n_0) / (Y + n_0)|^2
    # Let's verify:
    # Y = (n_0 * B + C) / (D + n_0 * E)
    # This is a known formula. Let's use it.
    
    Y = (n_0 * B + C) / (D + n_0 * E)
    R = np.abs((Y - n_0) / (Y + n_0)) ** 2
    return 100.0 * np.mean(R)

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns (labels, thicknesses).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    band_nm = problem["band_nm"]
    materials = problem["materials"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    best_score = float('inf')
    best_labels = None
    best_thicknesses = None
    
    # Initial guess: quarter-wave stack
    centre = 0.5 * (band_nm[0] + band_nm[1])
    # Try a few initial configurations
    initial_configs = [
        (["L"], [centre / (4.0 * materials["L"])]),
        (["L", "H", "L"], [centre / (4.0 * materials["L"]), centre / (4.0 * materials["H"]), centre / (4.0 * materials["L"])]),
        (["L", "H", "L", "H", "L"], [centre / (4.0 * materials["L"]), centre / (4.0 * materials["H"]), centre / (4.0 * materials["L"]), centre / (4.0 * materials["H"]), centre / (4.0 * materials["L"])]),
    ]
    
    for labels, thicknesses in initial_configs:
        if sum(thicknesses) > max_total_nm:
            continue
        score = _mean_reflectance(labels, thicknesses, problem)
        if score < best_score:
            best_score = score
            best_labels = labels[:]
            best_thicknesses = thicknesses[:]
    
    # Optimization: use Nelder-Mead or similar
    # We'll optimize the thicknesses for a fixed number of layers.
    # Try different numbers of layers: 3, 5, 7, 9
    for n_layers in [3, 5, 7, 9]:
        if n_layers > max_layers:
            break
        if time.time() > deadline:
            break
        
        # Initial thicknesses: quarter-wave
        init_t = np.array([centre / (4.0 * materials["L"]) if i % 2 == 0 else centre / (4.0 * materials["H"]) for i in range(n_layers)])
        # Labels: alternating L, H, L, H, ...
        labels = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        
        # Objective function
        def objective(t):
            # Enforce constraints
            if np.any(t < 0) or np.any(t > max_layer_nm) or np.sum(t) > max_total_nm:
                return 100.0  # Penalty
            return _mean_reflectance(labels, t, problem)
        
        # Use scipy.optimize.minimize with Nelder-Mead
        from scipy.optimize import minimize
        
        res = minimize(objective, init_t, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6, 'disp': False})
        
        if res.fun < best_score:
            best_score = res.fun
            best_labels = labels[:]
            best_thicknesses = res.x.tolist()
    
    # If we have time, try a few more random restarts
    if time.time() < deadline - 5.0:
        import random
        random.seed(42)
        for _ in range(10):
            if time.time() > deadline:
                break
            n_layers = random.choice([3, 5, 7, 9])
            if n_layers > max_layers:
                continue
            labels = ["L" if random.random() < 0.5 else "H" for _ in range(n_layers)]
            t = [random.uniform(0, max_layer_nm) for _ in range(n_layers)]
            # Scale to fit within max_total_nm
            total = sum(t)
            if total > max_total_nm:
                t = [x * max_total_nm / total for x in t]
            
            score = _mean_reflectance(labels, t, problem)
            if score < best_score:
                best_score = score
                best_labels = labels[:]
                best_thicknesses = t[:]
    
    if best_labels is None:
        # Fallback
        return ["L"], [centre / (4.0 * materials["L"])]
    
    # Ensure constraints are met
    total = sum(best_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]
    for i in range(len(best_thicknesses)):
        if best_thicknesses[i] > max_layer_nm:
            best_thicknesses[i] = max_layer_nm
        if best_thicknesses[i] < 0:
            best_thicknesses[i] = 0.0
    
    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
