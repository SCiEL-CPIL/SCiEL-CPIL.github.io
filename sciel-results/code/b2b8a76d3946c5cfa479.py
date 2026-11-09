"""Broadband antireflection coating design via global + local optimization."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START
def _reflectance_spectrum(materials, thicknesses, ambient_index, substrate_index, wavelengths):
    """Compute normal-incidence reflectance (0-1) using the characteristic-matrix method."""
    n_amb = ambient_index
    n_sub = substrate_index
    N = len(thicknesses)
    # Build the product of all layer matrices
    # Each layer matrix: [[cos(delta), (i/sin(delta))/n], [i*n*sin(delta), cos(delta)]]
    # For normal incidence
    # We use the standard 2x2 characteristic matrix approach
    # M = product over layers of M_j
    # Then Y = (M11 + M12*n_sub) / (M21 + M22*n_sub)
    # R = |(n_amb - Y)/(n_amb + Y)|^2
    
    # Vectorized over wavelengths
    # wavelengths: shape (W,)
    # thicknesses: shape (N,)
    # materials: dict mapping 'L' -> n_L, 'H' -> n_H
    
    # We need to know which material each layer is
    # Actually, the function signature in the task says design returns (materials, thicknesses)
    # where materials is a list of 'L'/'H' labels
    # So _reflectance_spectrum should take the list of materials and thicknesses
    
    # Let's redesign: this helper takes labels and thicknesses
    pass

def _compute_R(labels, thicknesses, problem, wavelengths):
    """Compute reflectance spectrum for given design."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    mat_dict = problem["materials"]
    N = len(labels)
    W = len(wavelengths)
    
    # For each wavelength, compute the characteristic matrix product
    # M_total = M_1 * M_2 * ... * M_N
    # M_j = [[cos(d_j), i*sin(d_j)/n_j], [i*n_j*sin(d_j), cos(d_j)]]
    # d_j = 2*pi*n_j*t_j / lambda
    
    # Initialize M_total as identity for each wavelength
    M11 = np.ones(W, dtype=complex)
    M12 = np.zeros(W, dtype=complex)
    M21 = np.zeros(W, dtype=complex)
    M22 = np.ones(W, dtype=complex)
    
    for i in range(N):
        n_j = mat_dict[labels[i]]
        t_j = thicknesses[i]
        delta = 2.0 * np.pi * n_j * t_j / wavelengths  # shape (W,)
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # Layer matrix
        L11 = cos_d
        L12 = 1j * sin_d / n_j
        L21 = 1j * n_j * sin_d
        L22 = cos_d
        
        # M_total = M_total * L
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Admittance Y = (M11 + M12 * n_sub) / (M21 + M22 * n_sub)
    Y = (M11 + M12 * n_sub) / (M21 + M22 * n_sub)
    
    # Reflectance R = |(n_amb - Y) / (n_amb + Y)|^2
    R = np.abs((n_amb - Y) / (n_amb + Y))**2
    
    return R

def _objective(x, problem, labels, wavelengths):
    """Objective function: mean reflectance in percent, with penalty for constraint violations."""
    N = len(labels)
    thicknesses = x[:N]
    
    # Check constraints
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    total = np.sum(thicknesses)
    if total > max_total_nm:
        return 100.0 + 0.1 * (total - max_total_nm)
    if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
        return 100.0 + 0.1 * (np.max(np.abs(thicknesses - np.clip(thicknesses, 0, max_layer_nm))))
    
    R = _compute_R(labels, thicknesses, problem, wavelengths)
    return np.mean(R) * 100.0

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin: kill at 120s, evaluator kills at 150s
    
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Use 401 equally spaced wavelengths for evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Try different numbers of layers and material sequences
    # For broadband AR coatings, common strategies:
    # 1. Alternating L-H layers (quarter-wave stack)
    # 2. Optimized layer sequences
    
    # Start with a reasonable number of layers
    # Let's try N=6, 8, 10, 12 layers with alternating patterns
    
    best_score = 100.0
    best_result = None
    
    # Try different layer counts
    for N in [4, 6, 8, 10, 12]:
        if time.time() > deadline - 30.0:
            break
        
        # Try alternating starting with L
        labels_L = ['L' if i % 2 == 0 else 'H' for i in range(N)]
        # Try alternating starting with H
        labels_H = ['H' if i % 2 == 0 else 'L' for i in range(N)]
        
        for labels in [labels_L, labels_H]:
            if time.time() > deadline - 30.0:
                break
            
            # Bounds for differential evolution
            bounds = [(0, max_layer_nm)] * N
            
            # Initial guess: quarter-wave at center wavelength
            centre = 0.5 * (band[0] + band[1])
            n_L = problem["materials"]["L"]
            n_H = problem["materials"]["H"]
            x0 = np.array([centre / (4.0 * (n_L if labels[i] == 'L' else n_H)) for i in range(N)])
            
            # Use differential evolution for global optimization
            try:
                result = differential_evolution(
                    _objective,
                    bounds,
                    args=(problem, labels, wavelengths),
                    maxiter=100,
                    popsize=15,
                    tol=1e-8,
                    seed=42,
                    init='latinhypercube',
                    polish=False
                )
                
                if result.fun < best_score:
                    best_score = result.fun
                    best_result = (labels, result.x.tolist())
            except Exception:
                pass
            
            if time.time() > deadline - 30.0:
                break
            
            # Refine with Nelder-Mead
            try:
                res2 = minimize(
                    _objective,
                    result.x if hasattr(result, 'x') else x0,
                    args=(problem, labels, wavelengths),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 0.1, 'fatol': 1e-8}
                )
                
                if res2.fun < best_score:
                    best_score = res2.fun
                    best_result = (labels, res2.x.tolist())
            except Exception:
                pass
    
    # If no valid result found, fall back to a simple design
    if best_result is None:
        centre = 0.5 * (band[0] + band[1])
        n_L = problem["materials"]["L"]
        return ["L"], [centre / (4.0 * n_L)]
    
    labels, thicknesses = best_result
    
    # Ensure constraints are satisfied
    thicknesses = [max(0.0, min(max_layer_nm, t)) for t in thicknesses]
    if sum(thicknesses) > max_total_nm:
        # Scale down
        scale = max_total_nm / sum(thicknesses)
        thicknesses = [t * scale for t in thicknesses]
    
    return labels, thicknesses
# EVOLVE-BLOCK-END
