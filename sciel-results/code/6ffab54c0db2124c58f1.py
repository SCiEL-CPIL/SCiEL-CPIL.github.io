import numpy as np
import time
import scipy.optimize

def _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    n = len(n_layers)
    if n == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.abs(r) ** 2

    M11 = np.ones(len(wavelengths), dtype=complex)
    M12 = np.zeros(len(wavelengths), dtype=complex)
    M21 = np.zeros(len(wavelengths), dtype=complex)
    M22 = np.ones(len(wavelengths), dtype=complex)

    for i in range(n):
        ni = n_layers[i]
        di = d_layers[i]
        delta = 2.0 * np.pi * ni * di / wavelengths
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        L11 = cos_d
        L12 = 1j * sin_d / ni
        L21 = 1j * ni * sin_d
        L22 = cos_d
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22

    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    r = (n_ambient - Y) / (n_ambient + Y)
    return np.abs(r) ** 2

def _objective(x, materials, n_ambient, n_substrate, wavelengths, max_total_nm, max_layer_nm):
    n_layers = np.array([1.38 if m == 'L' else 2.35 for m in materials])
    x_clipped = np.clip(x, 0.0, max_layer_nm)
    total_thick = np.sum(x_clipped)
    penalty = 0.0
    if total_thick > max_total_nm:
        penalty += (total_thick - max_total_nm) * 10.0
    R = _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, x_clipped)
    return np.mean(R) * 100.0 + penalty

def _objective_symmetric(x_half, materials, n_ambient, n_substrate, wavelengths, max_total_nm, max_layer_nm):
    n = len(materials)
    full_thicks = np.zeros(n)
    n_layers = np.array([1.38 if m == 'L' else 2.35 for m in materials])
    
    if n % 2 == 1:
        mid = n // 2
        full_thicks[:mid+1] = x_half
        full_thicks[mid+1:] = x_half[mid-1::-1]
    else:
        half = n // 2
        full_thicks[:half] = x_half
        full_thicks[half:] = x_half[::-1]
        
    full_thicks = np.clip(full_thicks, 0.0, max_layer_nm)
    total_thick = np.sum(full_thicks)
    penalty = 0.0
    if total_thick > max_total_nm:
        penalty += (total_thick - max_total_nm) * 10.0
        
    R = _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, full_thicks)
    return np.mean(R) * 100.0 + penalty

def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    def time_left():
        return deadline - time.time()
        
    candidates = [
        ["L"], ["H"],
        ["L", "H"], ["H", "L"],
        ["L", "H", "L"], ["H", "L", "H"], ["L", "L", "H"], ["H", "H", "L"],
        ["L", "H", "L", "H", "L"], ["H", "L", "H", "L", "H"],
        ["L", "H", "H", "H", "L"], ["H", "L", "L", "L", "H"],
        ["L", "H", "L", "H", "L", "H", "L"], ["H", "L", "H", "L", "H", "L", "H"],
        ["L", "H", "H", "H", "H", "H", "L"], ["H", "L", "L", "L", "L", "L", "H"],
        ["L", "H", "L", "H", "L", "H", "L", "H", "L"], ["H", "L", "H", "L", "H", "L", "H", "L", "H"],
        ["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], ["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"]
    ]
    
    unique_candidates = []
    seen = set()
    for c in candidates:
        if len(c) <= max_layers:
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)
                
    for mats in unique_candidates:
        if time_left() < 5.0:
            break
            
        n = len(mats)
        is_symmetric = (mats == mats[::-1])
        
        # Determine dimensionality
        if is_symmetric:
            dim = (n + 1) // 2
        else:
            dim = n
            
        # Generate initial guesses
        inits = []
        for lam in [450, 550, 600, 650, 750]:
            x0_full = np.zeros(n)
            n_layers_full = np.array([1.38 if m == 'L' else 2.35 for m in mats])
            for i in range(n):
                x0_full[i] = lam / (4.0 * n_layers_full[i])
                
            if is_symmetric:
                x0 = x0_full[:dim]
            else:
                x0 = x0_full
            inits.append(x0)
            
        # Random inits
        np.random.seed(123)
        for _ in range(3):
            x0_full = np.random.uniform(20, 150, n)
            if is_symmetric:
                x0 = x0_full[:dim]
            else:
                x0 = x0_full
            inits.append(x0)
            
        for x0 in inits:
            if time_left() < 2.0:
                break
                
            x0 = np.clip(x0, 0, max_layer_nm)
            
            try:
                if is_symmetric:
                    res = scipy.optimize.minimize(
                        _objective_symmetric, 
                        x0, 
                        args=(mats, n_ambient, n_substrate, wavelengths, max_total_nm, max_layer_nm),
                        method='Powell',
                        options={'maxiter': 200, 'xtol': 1e-4, 'ftol': 1e-6}
                    )
                    x_opt_half = np.clip(res.x, 0, max_layer_nm)
                    # Reconstruct full
                    full_thicks = np.zeros(n)
                    if n % 2 == 1:
                        mid = n // 2
                        full_thicks[:mid+1] = x_opt_half
                        full_thicks[mid+1:] = x_opt_half[mid-1::-1]
                    else:
                        half = n // 2
                        full_thicks[:half] = x_opt_half
                        full_thicks[half:] = x_opt_half[::-1]
                else:
                    res = scipy.optimize.minimize(
                        _objective, 
                        x0, 
                        args=(mats, n_ambient, n_substrate, wavelengths, max_total_nm, max_layer_nm),
                        method='Powell',
                        options={'maxiter': 200, 'xtol': 1e-4, 'ftol': 1e-6}
                    )
                    full_thicks = np.clip(res.x, 0, max_layer_nm)
                    
                # Final constraint check
                if np.sum(full_thicks) > max_total_nm:
                    full_thicks = full_thicks * (max_total_nm / np.sum(full_thicks))
                    
                score = _objective(full_thicks, mats, n_ambient, n_substrate, wavelengths, max_total_nm, max_layer_nm)
                
                if score < best_score:
                    best_score = score
                    best_materials = mats
                    best_thicknesses = full_thicks.tolist()
                    
            except Exception:
                pass

    # Polish
    if best_materials is not None and time_left() > 2.0:
        current_thicks = np.array(best_thicknesses)
        current_score = _objective(current_thicks, best_materials, n_ambient, n_substrate, wavelengths, max_total_nm, max_layer_nm)
        step = 5.0
        for _ in range(20):
            if time_left() < 1.0: break
            improved = False
            for i in range(len(current_thicks)):
                for sign in [1, -1]:
                    new_t = current_thicks[i] + sign * step
                    if new_t < 0 or new_t > max_layer_nm: continue
                    new_thicks = current_thicks.copy()
                    new_thicks[i] = new_t
                    if np.sum(new_thicks) > max_total_nm: continue
                    new_score = _objective(new_thicks, best_materials, n_ambient, n_substrate, wavelengths, max_total_nm, max_layer_nm)
                    if new_score < current_score:
                        current_thicks = new_thicks
                        current_score = new_score
                        improved = True
            if not improved:
                step *= 0.5
                if step < 0.01: break
            else:
                step = min(step * 1.2, 20.0)
        best_thicknesses = current_thicks.tolist()
        best_score = current_score

    if best_materials is None:
        return ["L"], [600 / (4 * 1.38)]
    return best_materials, best_thicknesses
