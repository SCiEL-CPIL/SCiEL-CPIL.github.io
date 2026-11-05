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
    beta = np.zeros((n_layers, n_wls))
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        if d == 0:
            continue
        beta[i, :] = 2.0 * math.pi * n * d / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Build the total characteristic matrix
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
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
    Compute mean reflectance in percent.
    Returns penalty if constraints are violated.
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

def _project_to_constraints(th, max_layer, max_total):
    """Project thicknesses onto the constraint box [0, max_layer] and sum <= max_total."""
    th = np.clip(th, 0.0, max_layer)
    s = np.sum(th)
    if s > max_total:
        th = th * (max_total / s)
        th = np.clip(th, 0.0, max_layer)
    return th

def _nelder_mead(objective, x0, bounds, deadline, max_iter=500, tol=1e-8):
    """
    Nelder-Mead optimization with bounds handling and adaptive step sizing.
    x0: initial point
    bounds: tuple (lo, hi) arrays
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    lo, hi = bounds
    
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        range_i = hi[i] - lo[i]
        step = 0.1 * range_i if range_i > 1e-6 else 1.0
        xi[i] += step
        simplex.append(np.clip(xi, lo, hi))
        
    f_vals = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    best_f = min(f_vals)
    stagnant_count = 0
    best_x = x0.copy()
    
    for iter_count in range(max_iter):
        if time.time() > deadline:
            break
            
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        spread = abs(f_vals[0] - f_vals[-1])
        if spread < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), lo, hi)
        f_r = objective(x_r)
        
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            stagnant_count = 0
            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()
            continue
            
        if f_r < f_vals[0]:
            x_e = np.clip(centroid + gamma * (x_r - centroid), lo, hi)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
                if f_e < best_f:
                    best_f = f_e
                    best_x = x_e.copy()
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
            stagnant_count = 0
            continue
            
        x_c = np.clip(centroid + rho * (simplex[-1] - centroid), lo, hi)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            stagnant_count = 0
            if f_c < best_f:
                best_f = f_c
                best_x = x_c.copy()
            continue
            
        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), lo, hi)
            f_vals[i] = objective(simplex[i])
            
        if min(f_vals) > best_f - 1e-10:
            stagnant_count += 1
        else:
            stagnant_count = 0
            best_f = min(f_vals)
            best_x = simplex[0].copy()
            
        if stagnant_count > 15:
            # Restart around best point with larger spread
            x_best = best_x.copy()
            simplex = [x_best.copy()]
            for i in range(n):
                xi = x_best.copy()
                range_i = hi[i] - lo[i]
                step = 0.5 * range_i if range_i > 1e-6 else 5.0
                xi[i] += step * random.uniform(0.5, 1.5)
                simplex.append(np.clip(xi, lo, hi))
            f_vals = [objective(x) for x in simplex]
            stagnant_count = 0
            
    order = np.argsort(f_vals)
    best_idx = order[0]
    return np.clip(simplex[best_idx], lo, hi), f_vals[best_idx]

def _generate_initial_guesses(mats, n_mat, max_total, max_layer):
    """Generate diverse initial guesses for optimization."""
    guesses = []
    n_layers = len(mats)
    
    # 1. Quarter-wave stacks at different center wavelengths
    for cw in [450, 500, 550, 600, 650, 700]:
        ths = []
        for m in mats:
            n = n_mat[m]
            q = cw / (4.0 * n)
            ths.append(q)
        # Scale to fit max_total if needed
        total = sum(ths)
        if total > max_total:
            scale = max_total / total
            ths = [t * scale for t in ths]
        # Ensure within max_layer
        ths = [min(t, max_layer) for t in ths]
        guesses.append(ths)
        
    # 2. Linear taper approximation (optical thickness proportional to index step?)
    # Try to distribute optical thicknesses such that the phase varies smoothly
    # Simple heuristic: equal optical thickness
    total_optical = max_total * 1.5 # Average index approx
    opt_th = total_optical / n_layers
    ths = []
    for m in mats:
        n = n_mat[m]
        d = opt_th / n
        ths.append(min(d, max_layer))
    guesses.append(ths)
    
    # 3. Random points
    for _ in range(5):
        ths = [random.uniform(10, max_layer) for _ in range(n_layers)]
        # Project to constraints
        ths_arr = np.array(ths)
        ths_arr = _project_to_constraints(ths_arr, max_layer, max_total)
        guesses.append(list(ths_arr))
        
    return guesses

def design(problem):
    """
    Design a broadband antireflection coating.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

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

    # Define candidate material sequences
    # Prioritize sequences starting with L and ending with L
    sequences = []
    
    # Length 2
    sequences.append(['L', 'H'])
    sequences.append(['H', 'L'])
    
    # Length 4
    sequences.append(['L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L'])
    sequences.append(['L', 'H', 'H', 'L'])
    sequences.append(['H', 'L', 'L', 'H'])
    
    # Length 6
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L', 'H', 'L'])
    sequences.append(['L', 'H', 'H', 'L', 'H', 'L'])
    sequences.append(['H', 'L', 'L', 'H', 'L', 'H'])
    sequences.append(['L', 'H', 'L', 'H', 'H', 'L']) # Ends in L
    
    # Length 8
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    sequences.append(['L', 'H', 'L', 'H', 'H', 'L', 'H', 'L'])
    
    # Length 10
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # Length 12
    sequences.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])
    sequences.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    sequences = [seq for seq in sequences if len(seq) <= max_layers]
    
    # Shuffle sequences to explore different lengths early?
    # Or sort by length?
    # Let's stick to the order but maybe interleave lengths.
    
    for mats in sequences:
        if time.time() > deadline - 5.0: # Stop if not enough time for one more run
            break
            
        n_layers = len(mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        initial_guesses = _generate_initial_guesses(mats, n_mat, max_total, max_layer)
        
        for x0 in initial_guesses:
            if time.time() > deadline - 2.0:
                break
                
            x0 = _project_to_constraints(np.array(x0), max_layer, max_total)
            
            def objective(th, m=mats):
                return _mean_reflectance_percent(m, list(th), problem, wls, n_mat)
            
            # Use fewer iterations for longer stacks?
            # Evaluation is fast, so we can afford more iterations.
            max_iter = 300
            x_opt, f_opt = _nelder_mead(objective, x0, (lo, hi), deadline, max_iter=max_iter)
            
            x_opt = _project_to_constraints(x_opt, max_layer, max_total)
            val = _mean_reflectance_percent(mats, list(x_opt), problem, wls, n_mat)
            
            if val < best_val:
                best_val = val
                best_mats = mats
                best_ths = list(x_opt)
                
    # Final polish on the best result found
    if best_mats is not None and time.time() < deadline - 10.0:
        n_layers = len(best_mats)
        lo = np.zeros(n_layers)
        hi = np.full(n_layers, max_layer)
        
        def objective(th, m=best_mats):
            return _mean_reflectance_percent(m, list(th), problem, wls, n_mat)
        
        # Polish with tighter tolerance
        x_polish, f_polish = _nelder_mead(objective, np.array(best_ths), (lo, hi), 
                                          deadline, max_iter=500, tol=1e-10)
        
        x_polish = _project_to_constraints(x_polish, max_layer, max_total)
        val_polish = _mean_reflectance_percent(best_mats, list(x_polish), problem, wls, n_mat)
        
        if val_polish < best_val:
            best_val = val_polish
            best_ths = list(x_polish)
            
    if best_mats is None:
        # Fallback: single layer
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        best_val = _mean_reflectance_percent(best_mats, best_ths, problem, wls, n_mat)
        
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    if sum(best_ths) > max_total + 1e-6:
        f = max_total / sum(best_ths)
        best_ths = [t * f for t in best_ths]
        
    return list(best_mats), list(best_ths)
