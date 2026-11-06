"""Broadband antireflection coating design via multi-start Powell hybrid with dynamic layer count."""
import numpy as np
import time
from scipy.optimize import minimize

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    """
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0)
    
    k = 2.0 * np.pi / wavelengths_nm
    
    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    return R

def _objective(params, problem, wavelengths_nm, labels):
    """
    Objective function: mean reflectance in percent.
    params: flat array of thicknesses
    """
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    thicknesses = np.clip(params, 0, max_layer_nm)
    
    # Penalty for exceeding total thickness
    total_thick = np.sum(thicknesses)
    penalty = 0.0
    if total_thick > max_total_nm:
        penalty = 10.0 * (total_thick - max_total_nm)
    
    indices = [problem["materials"][label] for label in labels]
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses, 
                             problem["ambient_index"], problem["substrate_index"])
    
    return np.mean(R) * 100.0 + penalty

def _generate_initial_guesses(n_layers, problem, n_guesses=10):
    """
    Generate diverse initial guesses for the optimizer.
    """
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    labels = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
    indices = [materials[label] for label in labels]
    
    guesses = []
    
    # 1. Quarter-wave stack at center wavelength
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    qw = np.array([center_wl / (4.0 * n) for n in indices])
    if np.sum(qw) > max_total_nm:
        qw = qw * (max_total_nm / np.sum(qw))
    guesses.append(qw)
    
    # 2. Quarter-wave stack at lower wavelength
    qw_low = np.array([band_nm[0] / (4.0 * n) for n in indices])
    if np.sum(qw_low) > max_total_nm:
        qw_low = qw_low * (max_total_nm / np.sum(qw_low))
    guesses.append(qw_low)
    
    # 3. Quarter-wave stack at higher wavelength
    qw_high = np.array([band_nm[1] / (4.0 * n) for n in indices])
    if np.sum(qw_high) > max_total_nm:
        qw_high = qw_high * (max_total_nm / np.sum(qw_high))
    guesses.append(qw_high)
    
    # 4. Random guesses
    rng = np.random.default_rng(42)
    for _ in range(n_guesses - 3):
        guess = rng.uniform(0, max_layer_nm, size=n_layers)
        if np.sum(guess) > max_total_nm:
            guess = guess * (max_total_nm / np.sum(guess))
        guesses.append(guess)
    
    return guesses

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    best_result = None
    best_score = np.inf
    
    # Try layer counts from 8 to max_layers
    # Prioritize counts that are likely to perform well
    counts_to_try = list(range(8, max_layers + 1))
    
    # Shuffle to avoid bias, but keep a mix
    # Actually, let's try a specific order: start with mid-range, then expand
    # Based on literature, 10-15 layers often give good broadband performance
    # Let's try: 12, 10, 14, 8, 16, 9, 15, 11, 17, 13, 18, 19, 20
    counts_to_try = [12, 10, 14, 8, 16, 9, 15, 11, 17, 13, 18, 19, 20]
    counts_to_try = [c for c in counts_to_try if c <= max_layers]
    
    for n_layers in counts_to_try:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 3.0  # Leave 3s buffer
        
        if remaining < 5.0:
            break
        
        labels = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        
        def obj_func(x, bounds_local, start_time_local):
            return _objective(x, problem, wavelengths_nm, labels)
        
        bounds = [(0, max_layer_nm)] * n_layers
        bounds_lo = np.zeros(n_layers)
        bounds_hi = np.full(n_layers, max_layer_nm)
        
        # Generate initial guesses
        guesses = _generate_initial_guesses(n_layers, problem, n_guesses=5)
        
        # Time per optimization run
        # We have 'remaining' seconds left. We'll try up to 5 starts.
        # Each start should take about remaining/5 seconds, but we'll cap it.
        time_per_start = min(remaining / 5.0, 10.0)  # Cap at 10s per start
        
        for guess in guesses:
            elapsed = time.time() - start_time
            if time.time() - start_time > time_limit - 2.0:
                break
            
            # Use Powell method with tight tolerance
            try:
                res = minimize(
                    lambda x: _objective(x, problem, wavelengths_nm, labels),
                    guess,
                    method='Powell',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxfev': 10000, 'xtol': 1e-8, 'ftol': 1e-8}
                )
                
                # Clip and enforce total thickness constraint
                x_final = np.clip(res.x, 0, max_layer_nm)
                if np.sum(x_final) > max_total_nm:
                    x_final = x_final * (max_total_nm / np.sum(x_final))
                
                f_final = _objective(x_final, problem, wavelengths_nm, labels)
                
                if f_final < best_score:
                    best_score = f_final
                    best_result = (labels, x_final.tolist())
                    
            except Exception:
                continue
    
    # Fallback if no result found
    if best_result is None:
        labels = ['L']
        thicknesses = [600.0 / (4.0 * materials["L"])]
        return labels, thicknesses
    
    labels, thicknesses = best_result
    
    # Final constraint enforcement
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return labels, thicknesses.tolist()
