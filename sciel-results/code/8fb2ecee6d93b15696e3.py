import time
import math
import numpy as np
import random
from scipy.optimize import minimize

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses using
    the thin-film characteristic-matrix method.
    Returns array of reflectance values (0 to 1).
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    n_layers = len(materials)
    n_wls = len(wls)

    # Precompute beta = 2*pi*n*d/wl for all layers and wavelengths
    ns = np.array([n_mat[m] for m in materials])
    ds = np.array(thicknesses)
    beta = 2.0 * math.pi * np.outer(ns * ds, 1.0 / wls)

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i in range(n_layers):
        n = ns[i]
        c = cosb[i]
        s = sinb[i]

        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D

        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wls, n_mat):
    """
    Compute mean reflectance in percent with constraint penalties.
    """
    if len(materials) != len(thicknesses):
        return 1e6
    if len(materials) > problem["max_layers"]:
        return 1e6

    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]

    # Check individual layer bounds
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6

    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        # Penalty for exceeding total thickness
        return 1e6 + (total_th - max_total) * 100.0

    if len(materials) == 0:
        r = (problem["ambient_index"] - problem["substrate_index"]) / (
            problem["ambient_index"] + problem["substrate_index"])
        return r * r * 100.0

    R = _reflectance_array(materials, thicknesses, problem["ambient_index"],
                           problem["substrate_index"], wls, n_mat)
    return float(np.mean(R)) * 100.0

def _quarter_wave_thickness(n, wl):
    """Compute quarter-wave optical thickness for wavelength wl and index n."""
    return wl / (4.0 * n)

def design(problem):
    """
    Design a broadband antireflection coating using a multi-start Powell optimization
    with physically motivated initial guesses.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    band_start, band_end = problem["band_nm"]
    wls = np.linspace(band_start, band_end, 401)
    n_mat = problem["materials"]

    best_val = float('inf')
    best_mats = None
    best_ths = None

    def update_best(mats, ths):
        nonlocal best_val, best_mats, best_ths
        # Ensure types are lists of floats/strings for return
        ths_clipped = [min(max(float(t), 0.0), max_layer) for t in ths]
        s = sum(ths_clipped)
        if s > max_total:
            if s > 1e-9:
                f = max_total / s
                ths_clipped = [t * f for t in ths_clipped]
        
        # Remove zero layers to reduce layer count
        final_mats = []
        final_ths = []
        for m, t in zip(mats, ths_clipped):
            if t > 1e-6:
                final_mats.append(m)
                final_ths.append(t)
        
        if len(final_mats) > max_layers:
            return float('inf')
            
        if len(final_mats) == 0:
            return _mean_reflectance_percent([], [], problem, wls, n_mat)

        val = _mean_reflectance_percent(final_mats, final_ths, problem, wls, n_mat)
        if val < best_val:
            best_val = val
            best_mats = list(final_mats)
            best_ths = list(final_ths)
        return val

    # Generate initial candidates
    candidates = []
    
    # 1. Quarter-wave stacks
    # Try different numbers of layers
    for n_layers in range(2, max_layers + 1, 2):
        # Alternate L and H
        for start_char in ['L', 'H']:
            mats = []
            for i in range(n_layers):
                if (i % 2 == 0) == (start_char == 'L'):
                    mats.append('L')
                else:
                    mats.append('H')
            
            # Try quarter-wave thicknesses at different center wavelengths
            for wl_center in [400, 450, 500, 550, 600, 650, 700, 750, 800]:
                ths = []
                for m in mats:
                    n = n_mat[m]
                    th = _quarter_wave_thickness(n, wl_center)
                    ths.append(min(th, max_layer))
                
                # Scale to fit total thickness if necessary
                total = sum(ths)
                if total > max_total:
                    f = max_total / total
                    ths = [t * f for t in ths]
                
                candidates.append((mats, ths))

    # 2. Random stacks
    random.seed(42)
    for _ in range(50):
        n_layers = random.randint(4, max_layers)
        mats = [random.choice(['L', 'H']) for _ in range(n_layers)]
        ths = [random.uniform(50, max_layer) for _ in range(n_layers)]
        total = sum(ths)
        if total > max_total:
            f = max_total / total
            ths = [t * f for t in ths]
        candidates.append((mats, ths))

    # Evaluate initial candidates
    scored_candidates = []
    for mats, ths in candidates:
        val = update_best(mats, ths)
        if val < 1e6:
            scored_candidates.append((val, mats, ths))
    
    # Sort by initial score
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take the top 10 candidates for refinement
    top_candidates = scored_candidates[:10]
    
    # Refine with Powell's Method
    for val, mats, ths in top_candidates:
        if time.time() > deadline - 5:
            break
            
        x0 = np.array(ths)
        bounds = [(0, max_layer) for _ in range(len(x0))]
        
        def objective(x):
            # x is the array of thicknesses
            # Check constraints
            if np.any(x < 0) or np.any(x > max_layer):
                return 1e6
            if np.sum(x) > max_total:
                return 1e6 + (np.sum(x) - max_total) * 10.0
            
            # Compute reflectance
            return _mean_reflectance_percent(mats, list(x), problem, wls, n_mat)
        
        try:
            result = minimize(objective, x0, method='Powell', bounds=bounds, options={'maxiter': 100, 'disp': False})
            if result.success or result.fun < best_val:
                update_best(mats, result.x)
        except Exception:
            pass

    # If we have time left, do a second pass on the best solution found
    if best_ths is not None and time.time() < deadline - 10:
        # Perturb the best solution and optimize again
        x0 = np.array(best_ths)
        bounds = [(0, max_layer) for _ in range(len(x0))]
        
        # Add small noise to escape local minima
        x0_noisy = x0 + np.random.normal(0, 10, len(x0))
        x0_noisy = np.clip(x0_noisy, 0, max_layer)
        
        def objective(x):
            if np.any(x < 0) or np.any(x > max_layer):
                return 1e6
            if np.sum(x) > max_total:
                return 1e6 + (np.sum(x) - max_total) * 10.0
            return _mean_reflectance_percent(best_mats, list(x), problem, wls, n_mat)
        
        try:
            result = minimize(objective, x0_noisy, method='Powell', bounds=bounds, options={'maxiter': 100, 'disp': False})
            if result.success or result.fun < best_val:
                update_best(best_mats, result.x)
        except Exception:
            pass

    # Fallback
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        update_best(best_mats, best_ths)
    
    # Final constraint check and cleanup
    best_ths = [min(max(float(t), 0.0), max_layer) for t in best_ths]
    s = sum(best_ths)
    if s > max_total + 1e-6:
        f = max_total / s
        best_ths = [t * f for t in best_ths]
        best_ths = [min(max(float(t), 0.0), max_layer) for t in best_ths]
    
    return list(best_mats), list(best_ths)
