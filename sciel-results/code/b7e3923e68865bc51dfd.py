"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def _reflectance_matrix(n, t, lam, n_amb):
    """Compute reflectance for a given stack using characteristic matrix method.
    n: array of layer indices, t: array of layer thicknesses (nm), lam: wavelength (nm)
    Returns reflectance (0 to 1).
    """
    delta = 2.0 * np.pi * n * t / lam
    # Characteristic matrix for each layer
    # For normal incidence TE: eta = n
    c = np.cos(delta)
    s = np.sin(delta)
    # Build total matrix
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    for i in range(len(n)):
        Mi = np.array([[c[i], 1j * s[i] / n[i]],
                       [1j * n[i] * s[i], c[i]]], dtype=complex)
        M = M @ Mi
    # Admittance
    Y = M[1, 0] / M[0, 0]
    # Reflectance
    r = (Y - n_amb) / (Y + n_amb)
    return np.abs(r) ** 2

def _mean_reflectance(n, t, lam_grid, n_amb):
    """Mean reflectance over wavelength grid."""
    Rs = np.array([_reflectance_matrix(n, t, lam, n_amb) for lam in lam_grid])
    return np.mean(Rs) * 100.0  # in percent

def _optimize_thicknesses(materials_list, n_layers, n_amb, n_sub, lam_grid, max_total_nm, max_layer_nm, max_time, initial_guess=None):
    """Optimize thicknesses for a given layer structure."""
    n_indices = np.array([materials_list[i] for i in range(n_layers)], dtype=float)
    
    # Objective: mean reflectance
    def objective(t):
        t = np.abs(t)  # ensure positive
        # Clip to valid range
        t = np.clip(t, 0, max_layer_nm)
        # Check total
        if np.sum(t) > max_total_nm:
            return 100.0  # large penalty
        return _mean_reflectance(n_indices, t, lam_grid, n_amb)
    
    # Initial guess: quarter-wave at center wavelength
    lam_center = 0.5 * (lam_grid[0] + lam_grid[-1])
    if initial_guess is None:
        t0 = np.array([lam_center / (4.0 * n_indices[i]) for i in range(n_layers)], dtype=float)
    else:
        t0 = np.array(initial_guess, dtype=float)
    
    # Bounds
    bounds = [(0, max_layer_nm)] * n_layers
    
    # Use L-BFGS-B with time limit
    best_t = t0.copy()
    best_obj = objective(best_t)
    
    start_time = time.time()
    deadline = start_time + max_time
    
    # Try a few optimization methods
    for method in ['L-BFGS-B', 'BFGS']:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, t0, method=method, bounds=bounds,
                          options={'maxiter': 200, 'maxfun': 500, 'maxfev': 500})
            if res.success or res.fun < best_obj:
                t_new = np.abs(res.x)
                t_new = np.clip(t_new, 0, max_layer_nm)
                if np.sum(t_new) <= max_total_nm:
                    obj_new = objective(t_new)
                    if obj_new < best_obj:
                        best_obj = obj_new
                        best_t = t_new
        except Exception:
            pass
    
    return best_t, best_obj

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    max_wall_time = 130.0  # safety margin before 150s kill
    
    n_amb = problem["ambient_index"]  # 1.0
    n_sub = problem["substrate_index"]  # 1.52
    materials = problem["materials"]  # {'L': 1.38, 'H': 2.35}
    band_nm = problem["band_nm"]  # [400, 800]
    max_layers = problem["max_layers"]  # 20
    max_total_nm = problem["max_total_nm"]  # 1000
    max_layer_nm = problem["max_layer_nm"]  # 400
    
    # Wavelength grid: 401 equally spaced wavelengths
    lam_grid = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Try several layer structures and pick the best
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Candidate structures: alternating H/L, starting with L (to match air better) or H
    # Common broadband AR: start with low index, end with low index (to match substrate better via gradual transition)
    # Actually, for AR on glass: air (1.0) -> coating -> glass (1.52)
    # Start with L (1.38) is closer to air, end with L is closer to glass
    
    # Try different numbers of layers
    layer_counts = [3, 5, 7, 9, 11]
    
    # Time budget per structure
    per_structure_time = 15.0
    
    for n_layers in layer_counts:
        if time.time() - start_time > max_wall_time - 10.0:  # leave 10s margin
            break
        
        # Structure: alternate L, H, L, H, ...
        # Start with L, end with L if odd number of layers
        materials_list = []
        for i in range(n_layers):
            if i % 2 == 0:
                materials_list.append('L')
            else:
                materials_list.append('H')
        
        n_indices = np.array([materials[m] for m in materials_list], dtype=float)
        
        # Check if total quarter-wave thickness fits
        lam_center = 0.5 * (band_nm[0] + band_nm[1])
        t_qw = np.array([lam_center / (4.0 * n_indices[i]) for i in range(n_layers)], dtype=float)
        if np.sum(t_qw) > max_total_nm:
            # Scale down
            scale = max_total_nm / np.sum(t_qw)
            t_qw = t_qw * scale
        
        # Optimize
        remaining_time = max_wall_time - (time.time() - start_time)
        opt_time = min(per_structure_time, remaining_time - 5.0)
        if opt_time < 2.0:
            break
        
        t_opt, obj = _optimize_thicknesses(materials_list, n_layers, n_amb, n_sub, lam_grid, 
                                           max_total_nm, max_layer_nm, opt_time, initial_guess=t_qw)
        
        if obj < best_score:
            best_score = obj
            best_materials = materials_list
            best_thicknesses = t_opt
    
    # If no valid result, use a simple 1-layer AR
    if best_materials is None:
        lam_center = 0.5 * (band_nm[0] + band_nm[1])
        best_materials = ['L']
        best_thicknesses = np.array([lam_center / (4.0 * materials['L'])])
    
    # Ensure thicknesses are valid
    best_thicknesses = np.clip(best_thicknesses, 0, max_layer_nm)
    if np.sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / np.sum(best_thicknesses)
        best_thicknesses = best_thicknesses * scale
    
    # Convert to lists
    return best_materials, [float(t) for t in best_thicknesses]
# EVOLVE-BLOCK-END
