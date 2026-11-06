"""
Broadband antireflection coating design via scipy differential evolution + Nelder-Mead polish.

Strategy:
- Use scipy's differential evolution for global search over continuous thicknesses
- Polish best results with Nelder-Mead for local convergence
- Try multiple layer configurations (different counts and alternating patterns)
- Select the best result within the time budget
"""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

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

def _objective(thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx, 
               max_layer_nm, max_total_nm):
    """
    Objective: mean reflectance in percent. Returns large penalty if constraints violated.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    if np.any(thicknesses < -1e-6) or np.any(thicknesses > max_layer_nm + 1e-6):
        return 100.0
    if np.sum(thicknesses) > max_total_nm + 1e-6:
        return 100.0
    
    thicknesses_clipped = np.clip(thicknesses, 0.0, max_layer_nm)
    if np.sum(thicknesses_clipped) > max_total_nm:
        return 100.0
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses_clipped, 
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _nelder_mead_polish(f, x0, bounds, max_iter=2000, time_deadline=None):
    """
    Nelder-Mead simplex with bounds for local polish.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x0 = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    
    for i in range(n):
        simplex[i + 1] = x0.copy()
        lo, hi = bounds[i]
        step = 0.05 * (hi - lo) if hi > lo else 0.05
        simplex[i + 1][i] += step
    
    fvals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    tol = 1e-10
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, lo_bounds, hi_bounds)
        f_r = f(x_r)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, lo_bounds, hi_bounds)
            f_e = f(x_e)
            
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, lo_bounds, hi_bounds)
            f_c = f(x_c)
            
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], lo_bounds, hi_bounds)
                    fvals[i] = f(simplex[i])
    
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def _de_search(f, bounds, n_layers, max_total_nm, time_deadline):
    """
    Run differential evolution with a constraint penalty for total thickness.
    """
    n = len(bounds)
    
    def constrained_f(x):
        base_f = f(x)
        total = np.sum(x)
        if total > max_total_nm:
            base_f += 10.0 * (total - max_total_nm) / max_total_nm
        return base_f
    
    try:
        result = differential_evolution(
            constrained_f,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False,
            init='sobol'
        )
        return result.x, result.fun
    except Exception:
        return None, None

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    # Try different layer counts and patterns
    configs = []
    for n_layers in [5, 7, 9, 11]:
        if n_layers > max_layers:
            continue
        # LHLH... pattern
        labels_l = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        configs.append(labels_l)
        # HLHL... pattern
        labels_h = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        configs.append(labels_h)
    
    for labels in configs:
        if time.time() > deadline - 30:
            break
        
        indices = [materials[label] for label in labels]
        n = len(indices)
        bounds = [(0.0, float(max_layer_nm))] * n
        
        def make_f(indices):
            def f(x):
                return _objective(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                  max_layer_nm, max_total_nm)
            return f
        
        f = make_f(indices)
        
        # Initial quarter-wave estimate
        x0 = np.array([center_wl / (4.0 * n_i) for n_i in indices])
        if np.sum(x0) > max_total_nm:
            x0 = x0 * (max_total_nm / np.sum(x0))
        x0 = np.clip(x0, 0, max_layer_nm)
        
        f0 = f(x0)
        
        # Run DE if time permits
        if time.time() < deadline - 40:
            de_deadline = min(time.time() + 30.0, deadline - 25)
            x_de, f_de = _de_search(f, bounds, n, max_total_nm, de_deadline)
            if x_de is not None:
                x_de = np.clip(x_de, 0, max_layer_nm)
                if np.sum(x_de) > max_total_nm:
                    x_de = x_de * (max_total_nm / np.sum(x_de))
                f_de = f(x_de)
                if f_de < f0:
                    x0 = x_de
                    f0 = f_de
        
        # Polish with Nelder-Mead
        nm_deadline = min(time.time() + 10.0, deadline - 15)
        try:
            x_nm, f_nm = _nelder_mead_polish(f, x0, bounds, max_iter=1500, time_deadline=nm_deadline)
            if f_nm < f0:
                f0 = f_nm
                x0 = x_nm
        except Exception:
            pass
        
        if f0 < best_score:
            best_score = f0
            best_labels = labels.copy()
            best_thicknesses = x0.copy()
    
    # If no valid result, use a simple quarter-wave single layer
    if best_labels is None:
        best_labels = ["L"]
        best_thicknesses = np.array([center_wl / (4.0 * materials["L"])])
    
    # Final constraint check
    thicknesses = np.array(best_thicknesses)
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
