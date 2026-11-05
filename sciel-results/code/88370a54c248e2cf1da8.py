"""Broadband antireflection coating design using Differential Evolution with vectorized evaluation."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def design(problem):
    """
    Design a broadband antireflection coating.
    Uses Differential Evolution (global search) with vectorized reflectance computation.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    band_nm = problem["band_nm"]
    wl_min, wl_max = band_nm[0], band_nm[1]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # 401 equally spaced wavelengths
    wls = np.linspace(wl_min, wl_max, 401)
    
    # Fixed structure: 10 layers alternating H, L, H, L, ... (starts with H, ends with L)
    # This is a common broadband AR coating structure
    N_LAYERS = 10
    materials = ["H", "L", "H", "L", "H", "L", "H", "L", "H", "L"]
    
    # Validate structure
    if len(materials) > max_layers:
        # Reduce to max_layers
        N_LAYERS = max_layers
        materials = ["H" if i % 2 == 0 else "L" for i in range(N_LAYERS)]
    
    # Vectorized reflectance computation for all wavelengths at once
    def compute_reflectance_vectorized(thicknesses):
        """
        Compute reflectance for all 401 wavelengths simultaneously.
        thicknesses: array of N_LAYERS thicknesses in nm
        Returns: array of 401 reflectance values (fraction)
        """
        n_layers = len(thicknesses)
        
        # Initialize total matrix as identity for all wavelengths
        # M_total shape: (401, 2, 2) complex
        M_total = np.zeros((len(wls), 2, 2), dtype=complex)
        M_total[:, 0, 0] = 1.0
        M_total[:, 1, 1] = 1.0
        
        for i, mat in enumerate(materials):
            n = n_H if mat == "H" else n_L
            d = thicknesses[i]
            beta = 2.0 * np.pi * n * d / wls  # shape: (401,)
            cos_b = np.cos(beta)
            sin_b = np.sin(beta)
            
            # Layer matrix for each wavelength
            # M_layer[i, 0, 0] = cos_b[i]
            # M_layer[i, 0, 1] = -1j * sin_b[i] / n
            # M_layer[i, 1, 0] = 1j * n * sin_b[i]
            # M_layer[i, 1, 1] = cos_b[i]
            
            # Multiply M_total = M_total @ M_layer
            # New M_total[i, 0, 0] = M_total[i, 0, 0]*cos_b[i] + M_total[i, 0, 1]*1j*n*sin_b[i]
            # New M_total[i, 0, 1] = M_total[i, 0, 0]*(-1j*sin_b[i]/n) + M_total[i, 0, 1]*cos_b[i]
            # New M_total[i, 1, 0] = M_total[i, 1, 0]*cos_b[i] + M_total[i, 1, 1]*1j*n*sin_b[i]
            # New M_total[i, 1, 1] = M_total[i, 1, 0]*(-1j*sin_b[i]/n) + M_total[i, 1, 1]*cos_b[i]
            
            M00 = M_total[:, 0, 0].copy()
            M01 = M_total[:, 0, 1].copy()
            M10 = M_total[:, 1, 0].copy()
            M11 = M_total[:, 1, 1].copy()
            
            M_total[:, 0, 0] = M00 * cos_b + M01 * 1j * n * sin_b
            M_total[:, 0, 1] = M00 * (-1j * sin_b / n) + M01 * cos_b
            M_total[:, 1, 0] = M10 * cos_b + M11 * 1j * n * sin_b
            M_total[:, 1, 1] = M10 * (-1j * sin_b / n) + M11 * cos_b
        
        # Compute reflectance
        M11 = M_total[:, 0, 0]
        M12 = M_total[:, 0, 1]
        M21 = M_total[:, 1, 0]
        M22 = M_total[:, 1, 1]
        
        Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
        r = (n_ambient - Y) / (n_ambient + Y)
        R = np.abs(r) ** 2
        
        return R
    
    def objective(thicknesses):
        """
        Objective function: mean reflectance in percent.
        Includes penalty for invalid designs.
        """
        thicknesses = np.asarray(thicknesses, dtype=float)
        
        # Check constraints
        if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
            return 1e6
        if np.sum(thicknesses) > max_total_nm:
            return 1e6
        
        R = compute_reflectance_vectorized(thicknesses)
        mean_R = np.mean(R) * 100.0
        return mean_R
    
    # Initial guess: quarter-wave thicknesses at center wavelength
    wl_center = 0.5 * (wl_min + wl_max)
    d_H_qw = wl_center / (4.0 * n_H)
    d_L_qw = wl_center / (4.0 * n_L)
    
    x0 = np.zeros(N_LAYERS)
    for i in range(N_LAYERS):
        x0[i] = d_H_qw if materials[i] == "H" else d_L_qw
    
    # Bounds for DE
    bounds = [(0.0, max_layer_nm)] * N_LAYERS
    
    # Try Differential Evolution from scipy
    best_score = objective(x0)
    best_thicknesses = x0.copy()
    
    try:
        from scipy.optimize import differential_evolution
        
        # Use DE with moderate population size to balance exploration and speed
        # maxiter=50, popsize=20 => ~1000 evaluations
        # Each evaluation is vectorized over 401 wavelengths, should be fast
        
        elapsed = time.time() - start_time
        if elapsed < deadline - 10.0:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=50,
                popsize=20,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.8,
                seed=42,
                polish=False  # Skip local polish to save time; we'll do our own
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_thicknesses = result.x.copy()
        
        # If time permits, try a second DE run with different seed for diversity
        elapsed = time.time() - start_time
        if elapsed < deadline - 15.0:
            result2 = differential_evolution(
                objective,
                bounds,
                maxiter=40,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 0.8),
                recombination=0.7,
                seed=123,
                polish=False
            )
            
            if result2.fun < best_score:
                best_score = result2.fun
                best_thicknesses = result2.x.copy()
        
        # Local refinement with L-BFGS-B if time permits
        elapsed = time.time() - start_time
        if elapsed < deadline - 20.0:
            from scipy.optimize import minimize
            
            result_lbfgs = minimize(
                objective,
                best_thicknesses,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result_lbfgs.fun < best_score:
                best_score = result_lbfgs.fun
                best_thicknesses = result_lbfgs.x.copy()
                
    except Exception as e:
        # If DE fails, fall back to the initial guess
        pass
    
    # Ensure validity and clamp
    best_thicknesses = np.clip(best_thicknesses, 0.0, max_layer_nm)
    
    # Final validity check
    if np.any(best_thicknesses < 0) or np.any(best_thicknesses > max_layer_nm) or np.sum(best_thicknesses) > max_total_nm:
        # Fall back to quarter-wave design
        best_thicknesses = x0.copy()
    
    return materials, [float(t) for t in best_thicknesses]

# EVOLVE-BLOCK-END
