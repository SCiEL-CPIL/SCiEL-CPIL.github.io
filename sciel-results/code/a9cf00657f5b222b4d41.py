"""Design broadband antireflection coating using differential evolution."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _reflectance_spectrum(thicknesses, indices, n_layers, wavelengths, ambient, substrate):
    """
    Compute normal-incidence reflectance using characteristic matrix method.
    Returns reflectance as a fraction (0 to 1).
    """
    n = n_layers
    # Build the total characteristic matrix M = M_1 * M_2 * ... * M_n
    # Each layer i has matrix:
    #   [[cos(delta_i), i*sin(delta_i)/n_i],
    #    [i*n_i*sin(delta_i), cos(delta_i)]]
    # where delta_i = 2*pi*n_i*d_i / wavelength
    
    # M_total = product of all layer matrices
    # We can compute this iteratively
    
    # Initialize M as identity
    m11 = 1.0 + 0.0j
    m12 = 0.0 + 0.0j
    m21 = 0.0 + 0.0j
    m22 = 1.0 + 0.0j
    
    for i in range(n):
        ni = indices[i]
        di = thicknesses[i]
        delta = 2.0 * np.pi * ni * di / wavelengths
        cd = np.cos(delta)
        sd = np.sin(delta)
        
        # Layer matrix
        li11 = cd
        li12 = 1j * sd / ni
        li21 = 1j * ni * sd
        li22 = cd
        
        # M_new = M_old * M_layer
        nm11 = m11 * li11 + m12 * li21
        nm12 = m11 * li12 + m12 * li22
        nm21 = m21 * li11 + m22 * li21
        nm22 = m21 * li12 + m22 * li22
        
        m11, m12, m21, m22 = nm11, nm12, nm21, nm22
    
    # The optical admittance Y = n_s * (m11 + m12) / (m21 + m22) ... wait
    # Actually, the characteristic matrix relates:
    # [E, H]_0 = M * [E, H]_N
    # where N is the substrate side.
    # For a stack on substrate with index n_s, the boundary condition at the last interface gives:
    # Y = n_s * (m22 + m21) / (m12 + m11) ... let me recall the exact formula.
    
    # The standard formula: 
    # The effective admittance of the stack as seen from air is:
    # Y = n_s * (m11 + m12) / (m21 + m22)
    # Wait, I need to be careful. Let me use the standard result.
    
    # Actually, the characteristic matrix M satisfies:
    # [E_0]   = M * [E_N]
    # [H_0]       [H_N]
    # where E_N and H_N are at the substrate interface.
    
    # At the substrate, the admittance is n_s (for normal incidence, lossless):
    # H_N = (n_s / n_0) * E_N ... no, H = (n/n_0) * E for the forward wave only.
    
    # More precisely, if we define H such that H = n * E for the forward wave in a medium of index n,
    # then at the substrate interface (last interface), we have a single forward wave (no back reflection from infinity),
    # so H_N = n_s * E_N.
    
    # Thus: [E_0; H_0] = M * [E_N; n_s * E_N] = E_N * [m11 + m12*n_s; m21 + m22*n_s]
    
    # The admittance at the air side is Y_0 = H_0 / E_0 = (m21 + m22*n_s) / (m11 + m12*n_s)
    
    # The reflectance is then:
    # r = (Y_0 - n_air) / (Y_0 + n_air)
    # R = |r|^2
    
    y0 = (m21 + m22 * substrate) / (m11 + m12 * substrate)
    r = (y0 - ambient) / (y0 + ambient)
    R = np.abs(r) ** 2
    return R

def design(problem):
    """
    Design a broadband antireflection coating using differential evolution.
    """
    ambient = problem["ambient_index"]
    substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wl_min, wl_max = band[0], band[1]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Use a subset of wavelengths for faster optimization
    opt_wavelengths = np.linspace(wl_min, wl_max, 100)
    
    # Define a 13-layer stack: L H L H L H L H L H L H L
    # This starts with L (close to air) and ends with L (close to glass 1.52 vs 1.38)
    n_opt_layers = 13
    labels = []
    for i in range(n_opt_layers):
        if i % 2 == 0:
            labels.append('L')
        else:
            labels.append('H')
            
    indices = [n_L if l == 'L' else n_H for l in labels]
    
    # Objective function
    def objective(thicknesses):
        # Check sum constraint
        total = sum(thicknesses)
        if total > max_total_nm:
            return 1.0 + (total - max_total_nm) * 0.01 # Penalty
        
        R = _reflectance_spectrum(thicknesses, indices, n_opt_layers, opt_wavelengths, ambient, substrate)
        return np.mean(R)
    
    # Bounds for each thickness
    bounds = [(0.0, max_layer_nm) for _ in range(n_opt_layers)]
    
    # Run differential evolution
    # Tuning parameters for speed: popsize=10, maxiter=50
    # This should be fast enough.
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10,
            tol=0.01,
            init='latinhypercube'
        )
        best_thicknesses = result.x
    except Exception:
        # Fallback if optimization fails
        best_thicknesses = [wl_min / (4.0 * n_L)] + [0.0] * (n_opt_layers - 1)
        
    # Final check and cleanup
    total = sum(best_thicknesses)
    if total > max_total_nm:
        # Scale down if penalty wasn't enough
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]
        
    # Ensure non-negative
    best_thicknesses = [max(0.0, t) for t in best_thicknesses]
    
    return labels, best_thicknesses

# EVOLVE-BLOCK-END
