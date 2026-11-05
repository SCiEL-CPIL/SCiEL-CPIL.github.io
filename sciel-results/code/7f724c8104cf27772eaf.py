"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
import sys
import numpy as np
from scipy.optimize import minimize

def _char_matrix(n, d, lam, n0=1.0):
    """Compute the 2x2 characteristic matrix for a single layer."""
    beta = 2.0 * np.pi * n * d / lam
    return np.array([[np.cos(beta), 1.0j / n * np.sin(beta)],
                     [1.0j * n * np.sin(beta), np.cos(beta)]])

def _total_matrix(mats, thicknesses, lam, n0=1.0):
    """Compute total transfer matrix for all layers."""
    M = np.eye(2, dtype=complex)
    for n, d in zip(mats, thicknesses):
        M = M @ _char_matrix(n, d, lam, n0)
    return M

def _reflectance(n0, ns, mats, thicknesses, lam):
    """Compute reflectance at a single wavelength."""
    M = _total_matrix(mats, thicknesses, lam, n0)
    m11, m12, m21, m22 = M[0,0], M[0,1], M[1,0], M[1,1]
    Y = (m11 + m12 * ns) / (m21 + m22 * ns)
    r = (n0 - Y) / (n0 + Y)
    return np.abs(r)**2

def _mean_reflectance(mats, thicknesses, n0, ns, band_nm):
    """Compute mean reflectance percent over the band."""
    lam_min, lam_max = band_nm
    wls = np.linspace(lam_min, lam_max, 401)
    total = 0.0
    for lam in wls:
        total += _reflectance(n0, ns, mats, thicknesses, lam)
    return 100.0 * total / len(wls)

def _design_quarterwave_stack(n_layers, n0, ns, nL, nH, band_nm):
    """Design a quarter-wave stack for the center wavelength."""
    center = 0.5 * (band_nm[0] + band_nm[1])
    labels = []
    thicknesses = []
    # Start with L on air side for AR coating
    for i in range(n_layers):
        if i % 2 == 0:
            labels.append('L')
            thicknesses.append(center / (4.0 * nL))
        else:
            labels.append('H')
            thicknesses.append(center / (4.0 * nH))
    return labels, thicknesses

def design(problem):
    """Design an antireflection coating using systematic initialization and Nelder-Mead."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n0 = problem["ambient_index"]  # 1.0
    ns = problem["substrate_index"]  # 1.52
    nL = problem["materials"]["L"]  # 1.38
    nH = problem["materials"]["H"]  # 2.35
    band_nm = problem["band_nm"]  # [400, 800]
    max_layers = problem["max_layers"]  # 20
    max_total_nm = problem["max_total_nm"]  # 1000
    max_layer_nm = problem["max_layer_nm"]  # 400
    
    best_labels = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Try different numbers of layers
    for n_layers in [4, 6, 8, 10, 12]:
        if n_layers > max_layers:
            break
        if time.time() > deadline - 30:
            break
        
        # Initialize with quarter-wave stack
        labels, thicknesses = _design_quarterwave_stack(n_layers, n0, ns, nL, nH, band_nm)
        
        # Check initial constraints
        if sum(thicknesses) > max_total_nm:
            # Scale down
            scale = max_total_nm / sum(thicknesses)
            thicknesses = [t * scale for t in thicknesses]
        
        # Initial evaluation
        score = _mean_reflectance([nL if l == 'L' else nH for l in labels], thicknesses, n0, ns, band_nm)
        
        if score < best_score:
            best_score = score
            best_labels = labels[:]
            best_thicknesses = thicknesses[:]
        
        # Optimize thicknesses using Nelder-Mead
        def objective(x):
            # x is the array of thicknesses
            t = np.clip(x, 0, max_layer_nm)
            total_t = np.sum(t)
            if total_t > max_total_nm:
                t = t * (max_total_nm / total_t)
                t = np.clip(t, 0, max_layer_nm)
            mats = [nL if l == 'L' else nH for l in labels]
            return _mean_reflectance(mats, t.tolist(), n0, ns, band_nm)
        
        x0 = np.array(thicknesses, dtype=float)
        
        # Set up bounds (Nelder-Mead doesn't support bounds directly, so we'll use a penalty)
        options = {
            'maxiter': 200,
            'maxfev': 1000,
            'xatol': 1e-4,
            'fatol': 1e-6,
            'adaptive': True
        }
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', options=options)
            if result.success or result.fun < best_score:
                x_opt = np.clip(result.x, 0, max_layer_nm)
                total_t = np.sum(x_opt)
                if total_t > max_total_nm:
                    x_opt = x_opt * (max_total_nm / total_t)
                    x_opt = np.clip(x_opt, 0, max_layer_nm)
                
                opt_score = _mean_reflectance([nL if l == 'L' else nH for l in labels], x_opt.tolist(), n0, ns, band_nm)
                if opt_score < best_score:
                    best_score = opt_score
                    best_labels = labels[:]
                    best_thicknesses = x_opt.tolist()
        except Exception:
            pass
        
        if time.time() > deadline - 20:
            break
    
    # If we haven't found a good solution, fall back to simple quarter-wave
    if best_labels is None:
        labels, thicknesses = _design_quarterwave_stack(4, n0, ns, nL, nH, band_nm)
        if sum(thicknesses) > max_total_nm:
            scale = max_total_nm / sum(thicknesses)
            thicknesses = [t * scale for t in thicknesses]
        best_labels = labels
        best_thicknesses = thicknesses
    
    # Ensure constraints are satisfied
    best_thicknesses = [max(0, min(t, max_layer_nm)) for t in best_thicknesses]
    total = sum(best_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_labels, best_thicknesses
# EVOLVE-BLOCK-END
