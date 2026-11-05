import time
import math
import numpy as np
import random

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

    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return 1e6

    total_th = sum(thicknesses)
    if total_th > max_total + 1e-6:
        return 1e6 + (total_th - max_total) * 10.0

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

def _nelder_mead_fast(objective, x0, lo, hi, deadline, max_iter=200, tol=1e-8):
    """
    Fast Nelder-Mead with bound handling and deadline check.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    
    if n == 0:
        return x0, objective(x0)

    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 * (hi[i] - lo[i]) if hi[i] - lo[i] > 1e-6 else 1.0
        xi[i] += step
        simplex.append(np.clip(xi, lo, hi))

    f_vals = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break

        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]

        if abs(f_vals[0] - f_vals[-1]) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)

        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue

        if f_r < f_vals[0]:
            x_e = np.clip(centroid + gamma * (x_r - centroid), lo, hi)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue

        x_c = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue

        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])

    order = np.argsort(f_vals)
    best_idx = order[0]
    return np.clip(simplex[best_idx], lo, hi), f_vals[best_idx]

def _coordinate_descent(objective, x0, lo, hi, deadline, max_iter=100, step_size=0.05):
    """
    Coordinate descent with adaptive step size.
    """
    x = np.array(x0, dtype=float)
    f_best = objective(x)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        improved = False
        for i in range(len(x)):
            if hi[i] - lo[i] < 1e-6:
                continue
                
            # Try positive and negative steps
            for direction in [1, -1]:
                x_trial = x.copy()
                x_trial[i] = np.clip(x[i] + direction * step_size * (hi[i] - lo[i]), lo[i], hi[i])
                f_trial = objective(x_trial)
                
                if f_trial < f_best - 1e-10:
                    x = x_trial
                    f_best = f_trial
                    improved = True
                    break
        
        if not improved:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-4:
                break
    
    return x, f_best

def design(problem):
    """
    Design a broadband antireflection coating using systematic initialization
    with quarter-wave stacks and local optimization.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

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
        ths_clipped = [min(max(t, 0.0), max_layer) for t in ths]
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
            return
        
        val = _mean_reflectance_percent(final_mats, final_ths, problem, wls, n_mat)
        if val < best_val:
            best_val = val
            best_mats = list(final_mats)
            best_ths = list(final_ths)
        return val

    # Generate systematic initial guesses based on quarter-wave stacks
    # at different wavelengths within the band
    candidate_configs = []
    
    # Try different numbers of layers and different starting materials
    for n_layers in [4, 6, 8, 10, 12]:
        if n_layers > max_layers:
            continue
        for start_char in ['L', 'H']:
            mats = []
            for i in range(n_layers):
                if (i % 2 == 0) == (start_char == 'L'):
                    mats.append('L')
                else:
                    mats.append('H')
            
            # Try quarter-wave thicknesses at different wavelengths
            for wl in [400, 450, 500, 550, 600, 650, 700, 750, 800]:
                ths = []
                for m in mats:
                    n = n_mat[m]
                    th = _quarter_wave_thickness(n, wl)
                    ths.append(min(th, max_layer))
                
                # Check total thickness
                total = sum(ths)
                if total > max_total:
                    # Scale down
                    f = max_total / total
                    ths = [t * f for t in ths]
                
                candidate_configs.append((mats, ths))
    
    # Also add some random configurations
    for _ in range(20):
        n_layers = random.randint(4, min(12, max_layers))
        mats = []
        for i in range(n_layers):
            mats.append(random.choice(['L', 'H']))
        
        ths = []
        for m in mats:
            th = random.uniform(50, min(200, max_layer))
            ths.append(th)
        
        total = sum(ths)
        if total > max_total:
            f = max_total / total
            ths = [t * f for t in ths]
        
        candidate_configs.append((mats, ths))
    
    # Sort candidates by initial reflectance (best first)
    scored_configs = []
    for mats, ths in candidate_configs:
        val = update_best(mats, ths)
        if val < 1e6:
            scored_configs.append((val, mats, ths))
    
    scored_configs.sort(key=lambda x: x[0])
    
    # Refine the best candidates with local optimization
    for val, mats, ths in scored_configs[:10]:
        if time.time() > deadline - 10:
            break
        
        # Nelder-Mead refinement
        lo = np.zeros(len(ths))
        hi = np.full(len(ths), max_layer)
        
        def objective(th):
            th_clipped = np.clip(th, 0, max_layer)
            total = np.sum(th_clipped)
            if total > max_total:
                th_clipped = th_clipped * (max_total / total)
            return _mean_reflectance_percent(mats, list(th_clipped), problem, wls, n_mat)
        
        x0 = np.array(ths)
        x_opt, f_opt = _nelder_mead_fast(objective, x0, lo, hi, deadline, max_iter=150)
        
        # Update best with refined result
        update_best(mats, x_opt)
        
        # Coordinate descent as a second pass
        if time.time() < deadline - 5:
            x_cd, f_cd = _coordinate_descent(objective, x_opt, lo, hi, deadline, max_iter=50)
            update_best(mats, x_cd)
    
    # Fallback
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        update_best(best_mats, best_ths)
    
    # Final constraint check
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    s = sum(best_ths)
    if s > max_total + 1e-6:
        f = max_total / s
        best_ths = [t * f for t in best_ths]
        best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    
    return list(best_mats), list(best_ths)
