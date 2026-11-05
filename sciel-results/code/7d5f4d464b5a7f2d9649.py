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

    ns = np.array([n_mat[m] for m in materials])
    ds = np.array(thicknesses)
    beta = 2.0 * math.pi * np.outer(ns * ds, 1.0 / wls)

    cosb = np.cos(beta)
    sinb = np.sin(beta)

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

def _project_to_feasible(x, max_layer, max_total):
    """Project x to feasible region: [0, max_layer] per layer, sum <= max_total."""
    x = np.clip(x, 0, max_layer)
    total = np.sum(x)
    if total > max_total:
        if total > 1e-12:
            x = x * (max_total / total)
        else:
            x = np.zeros_like(x)
    return x

def _nelder_mead(objective, x0, lo, hi, deadline, max_iter=200, tol=1e-10):
    """
    Nelder-Mead with adaptive step size, projection, and stagnation detection.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    
    if n == 0:
        return x0, objective(x0)

    max_layer = hi[0]
    max_total = sum(hi)
    
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 * (hi[i] - lo[i]) if hi[i] - lo[i] > 1e-6 else 1.0
        xi[i] += step
        xi = _project_to_feasible(xi, max_layer, max_total)
        simplex.append(xi)

    f_vals = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    stagnation_count = 0
    best_f_so_far = min(f_vals)
    stagnation_window = 15
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break

        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]

        if abs(f_vals[0] - f_vals[-1]) < tol:
            break

        if f_vals[0] < best_f_so_far:
            best_f_so_far = f_vals[0]
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        if stagnation_count >= stagnation_window:
            # Perturb non-best vertices
            for i in range(1, n + 1):
                perturbation = np.random.uniform(-0.2, 0.2, n) * (hi - lo)
                simplex[i] = _project_to_feasible(simplex[0] + perturbation, max_layer, max_total)
                f_vals[i] = objective(simplex[i])
            stagnation_count = 0
            continue

        centroid = np.mean(simplex[:-1], axis=0)

        x_r = _project_to_feasible(centroid + alpha * (centroid - simplex[-1]), max_layer, max_total)
        f_r = objective(x_r)

        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue

        if f_r < f_vals[0]:
            x_e = _project_to_feasible(centroid + gamma * (x_r - centroid), max_layer, max_total)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            continue

        x_c = _project_to_feasible(centroid + rho * (simplex[-1] - centroid), max_layer, max_total)
        f_c = objective(x_c)
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            continue

        for i in range(1, n + 1):
            simplex[i] = _project_to_feasible(simplex[0] + sigma * (simplex[i] - simplex[0]), max_layer, max_total)
            f_vals[i] = objective(simplex[i])

    order = np.argsort(f_vals)
    best_idx = order[0]
    return simplex[best_idx], f_vals[best_idx]

def _coordinate_descent(objective, x0, lo, hi, deadline, max_iter=100):
    """
    Coordinate descent with adaptive step size and line search.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    f = objective(x)
    
    max_layer = hi[0]
    max_total = sum(hi)
    
    step_sizes = np.array([(hi[i] - lo[i]) * 0.1 if hi[i] - lo[i] > 1e-6 else 1.0 for i in range(n)])
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            # Try both directions with decreasing step
            for direction in [-1, 1]:
                x_trial = x.copy()
                x_trial[i] += direction * step_sizes[i]
                x_trial = _project_to_feasible(x_trial, max_layer, max_total)
                f_trial = objective(x_trial)
                
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    improved = True
                    step_sizes[i] *= 1.3
                else:
                    step_sizes[i] *= 0.6
        
        if not improved:
            break
    
    return x, f

def _quarter_wave_init(mats, wls, n_mat, n_ambient, n_substrate):
    """Initialize with quarter-wave thicknesses at center wavelength."""
    wl_center = np.mean(wls)
    thicknesses = []
    for m in mats:
        n = n_mat[m]
        d = wl_center / (4.0 * n)
        thicknesses.append(d)
    return np.array(thicknesses)

def _multi_center_init(mats, wls, n_mat):
    """Initialize with different center wavelengths for different layers."""
    n_layers = len(mats)
    # Use multiple center wavelengths spread across the band
    n_centers = min(n_layers, 5)
    center_wls = np.linspace(wls[50], wls[-50], n_centers)
    
    thicknesses = []
    for i, m in enumerate(mats):
        n = n_mat[m]
        wl = center_wls[i % n_centers]
        d = wl / (4.0 * n)
        thicknesses.append(d)
    return np.array(thicknesses)

def _random_init(n_layers, max_layer, max_total):
    """Generate random thicknesses."""
    x = np.random.uniform(0, max_layer * 0.8, n_layers)
    total = np.sum(x)
    if total > max_total:
        x = x * (max_total / total)
    return x

def _generate_patterns(n_layers, max_layers):
    """Generate diverse layer patterns for given n_layers."""
    if n_layers > max_layers:
        return []
    
    patterns = []
    
    # Alternating starting with L
    mats_l = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
    patterns.append(mats_l)
    
    # Alternating starting with H
    mats_h = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
    patterns.append(mats_h)
    
    # Some random patterns for diversity
    for _ in range(min(3, max_layers - n_layers)):
        mats = [random.choice(['L', 'H']) for _ in range(n_layers)]
        if mats not in patterns:
            patterns.append(mats)
    
    return patterns

def design(problem):
    """
    Design a broadband antireflection coating using multi-start Nelder-Mead
    with physics-informed initializations and coordinate descent refinement.
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
        
        if len(final_mats) == 0:
            return float('inf')
        
        if len(final_mats) > max_layers:
            return float('inf')
        
        val = _mean_reflectance_percent(final_mats, final_ths, problem, wls, n_mat)
        if val < best_val:
            best_val = val
            best_mats = list(final_mats)
            best_ths = list(final_ths)
        return val

    # Define objective factory
    def make_objective(mats):
        def objective(th):
            th_clipped = _project_to_feasible(th, max_layer, max_total)
            return _mean_reflectance_percent(mats, list(th_clipped), problem, wls, n_mat)
        return objective

    # Try different layer counts
    # Focus on even numbers (better for symmetric designs) and a range
    layer_counts = [2, 4, 6, 8, 10, 12, 14, 16, 18]
    layer_counts = [lc for lc in layer_counts if lc <= max_layers]
    
    # Prioritize certain layer counts
    priority_counts = [4, 6, 8, 10, 12]
    other_counts = [lc for lc in layer_counts if lc not in priority_counts]
    ordered_counts = priority_counts + other_counts
    
    for n_layers in ordered_counts:
        if time.time() > deadline - 8:
            break
        
        patterns = _generate_patterns(n_layers, max_layers)
        
        for mats in patterns:
            if time.time() > deadline - 5:
                break
            
            objective = make_objective(mats)
            lo = np.zeros(n_layers)
            hi = np.full(n_layers, max_layer)
            
            # Multiple initializations
            inits = []
            
            # Quarter-wave at center
            inits.append(_quarter_wave_init(mats, wls, n_mat, n_ambient, n_substrate))
            
            # Multi-center initialization
            inits.append(_multi_center_init(mats, wls, n_mat))
            
            # Random initializations
            for _ in range(2):
                inits.append(_random_init(n_layers, max_layer, max_total))
            
            # Try each initialization
            for x0 in inits:
                if time.time() > deadline - 3:
                    break
                
                x0 = _project_to_feasible(x0, max_layer, max_total)
                
                # Nelder-Mead
                x_nm, f_nm = _nelder_mead(objective, x0, lo, hi, deadline, max_iter=80)
                
                # Coordinate descent refinement
                if time.time() < deadline - 2:
                    x_cd, f_cd = _coordinate_descent(objective, x_nm, lo, hi, deadline, max_iter=50)
                    if f_cd < f_nm:
                        x_nm = x_cd
                        f_nm = f_cd
                
                # Update best
                val = update_best(mats, x_nm)
                
                # Also check the initial point in case optimization diverged
                f_init = objective(x0)
                if f_init < f_nm:
                    update_best(mats, x0)
    
    # Fallback: simple single-layer AR coating
    if best_mats is None or best_val > 4.0:
        # Quarter-wave L layer
        wl_center = np.mean(wls)
        d = wl_center / (4.0 * n_L)
        if d <= max_layer:
            update_best(['L'], [d])
        
        # Quarter-wave H layer  
        d_h = wl_center / (4.0 * n_H)
        if d_h <= max_layer:
            update_best(['H'], [d_h])
        
        # Two-layer quarter-wave
        if 2 * d <= max_total and d <= max_layer:
            update_best(['L', 'H'], [d, d_h])
    
    # Final constraint check
    if best_mats is not None:
        best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
        s = sum(best_ths)
        if s > max_total + 1e-6:
            f = max_total / s
            best_ths = [t * f for t in best_ths]
            best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    
    if best_mats is None:
        best_mats = ['L']
        best_ths = [100.0]
    
    return list(best_mats), list(best_ths)
