"""Broadband antireflection coating design via multi-start Nelder-Mead on graded 20-layer stack."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _reflectance_spectrum(thicknesses, materials, ambient_index, substrate_index, wavelengths_nm):
    """
    Compute normal-incidence reflectance (fraction) for a stack of layers.
    thicknesses: array of physical thicknesses in nm (air -> substrate order)
    materials: dict mapping 'L'->n_L, 'H'->n_H (but here we pass indices directly)
    Actually, let's pass indices array directly.
    """
    # We'll pass indices array directly to this function
    pass

def _reflectance_spectrum_indices(indices, thicknesses, ambient_index, substrate_index, wavelengths_nm):
    """
    Compute normal-incidence reflectance (fraction) for a stack.
    indices: array of refractive indices for each layer (air-side to substrate-side)
    thicknesses: array of physical thicknesses in nm
    ambient_index: index of ambient medium (air)
    substrate_index: index of substrate (glass)
    wavelengths_nm: array of wavelengths in nm
    Returns: array of reflectance fractions
    """
    n_layers = len(indices)
    # Build the characteristic matrix product
    # For normal incidence, the characteristic matrix for layer i is:
    # M_i = [[cos(delta_i), (i/sin(delta_i))/n_i], [n_i * i * sin(delta_i), cos(delta_i)]]
    # where delta_i = 2*pi*n_i*d_i / lambda
    
    # We'll compute for each wavelength
    reflectances = np.zeros(len(wavelengths_nm))
    
    for j, lam in enumerate(wavelengths_nm):
        # Start with identity matrix
        M = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]])
        
        # Multiply through layers from air side to substrate side
        for i in range(n_layers):
            n_i = indices[i]
            d_i = thicknesses[i]
            delta_i = 2.0 * np.pi * n_i * d_i / lam
            cos_d = np.cos(delta_i)
            sin_d = np.sin(delta_i)
            
            # Characteristic matrix for layer i
            # [[cos(d), i*sin(d)/n], [i*n*sin(d), cos(d)]]
            M_i = np.array([
                [cos_d, 1j * sin_d / n_i],
                [1j * n_i * sin_d, cos_d]
            ])
            M = M @ M_i
        
        # Now compute reflectance
        # The total characteristic matrix M relates the fields at air side to substrate side
        # For a stack on a substrate with index n_s:
        # The effective admittance Y = (M[0,1]) / (M[0,0]) * n_s  ... wait, let me be careful
        # 
        # Actually, the standard formulation:
        # The characteristic matrix M = M_1 @ M_2 @ ... @ M_N
        # The reflection coefficient at normal incidence is:
        # r = (n_0 * M[0,0] + n_0 * M[0,1] * Y_s - M[1,0] - M[1,1] * Y_s) / 
        #     (n_0 * M[0,0] + n_0 * M[0,1] * Y_s + M[1,0] + M[1,1] * Y_s)
        # where Y_s = n_s (admittance of substrate at normal incidence)
        # and n_0 is the ambient index
        # 
        # Wait, I need to be more careful. Let me use the standard ABCD matrix formulation.
        # 
        # The characteristic matrix for the stack is M = [[A, B], [C, D]]
        # The reflection coefficient is:
        # r = (C + D*n_s - A*n_0 - B*n_0*n_s) / (C + D*n_s + A*n_0 + B*n_0*n_s)
        # 
        # Hmm, let me look at this more carefully. The standard result is:
        # r = (n_0*A + n_0*B*n_s - C - D*n_s) / (n_0*A + n_0*B*n_s + C + D*n_s)
        # 
        # Actually, I think the correct formula depends on the convention. Let me use a well-known form:
        # 
        # For a stack of layers between ambient (n_0) and substrate (n_s):
        # The total characteristic matrix M = [[A, B], [C, D]]
        # r = (A*n_0 + B*n_s*n_0 - C - D*n_s) / (A*n_0 + B*n_s*n_0 + C + D*n_s)
        # 
        # Let me verify with a single layer:
        # For one layer with index n_1, thickness d_1:
        # A = cos(delta_1), B = i*sin(delta_1)/n_1, C = i*n_1*sin(delta_1), D = cos(delta_1)
        # r = (cos*d*n_0 + i*sin/n_1 * n_1 * n_0 - i*n_1*sin - cos*n_s) / 
        #     (cos*d*n_0 + i*sin/n_1 * n_1 * n_0 + i*n_1*sin + cos*n_s)
        #   = (cos*n_0 + i*sin*n_0 - i*n_1*sin - cos*n_s) / 
        #     (cos*n_0 + i*sin*n_0 + i*n_1*sin + cos*n_s)
        #   = (cos*(n_0-n_s) + i*sin*(n_0-n_1)) / 
        #     (cos*(n_0+n_s) + i*sin*(n_0+n_1))
        # 
        # This looks correct for a single layer.
        
        A = M[0, 0]
        B = M[0, 1]
        C = M[1, 0]
        D = M[1, 1]
        
        n_0 = ambient_index
        n_s = substrate_index
        
        numerator = A * n_0 + B * n_s * n_0 - C - D * n_s
        denominator = A * n_0 + B * n_s * n_0 + C + D * n_s
        
        r = numerator / denominator
        reflectances[j] = np.abs(r) ** 2
    
    return reflectances

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    materials = problem["materials"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    
    n_L = materials["L"]
    n_H = materials["H"]
    
    # Wavelengths for evaluation: 401 equally spaced points from 400 to 800 nm
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Fixed 20-layer graded structure: L H L H ... L H (10 L, 10 H)
    # Indices from air side to substrate side
    fixed_indices = np.array([n_L, n_H] * 10)
    n_layers = len(fixed_indices)
    
    # Objective function: mean reflectance in percent
    def objective(x):
        # x is the thickness vector
        # Clip to valid range
        x_clipped = np.clip(x, 0.0, max_layer_nm)
        # Ensure total thickness constraint
        total = np.sum(x_clipped)
        if total > max_total_nm:
            # Scale down proportionally
            x_clipped = x_clipped * (max_total_nm / total)
        
        refl = _reflectance_spectrum_indices(fixed_indices, x_clipped, ambient_index, substrate_index, wavelengths)
        return np.mean(refl) * 100.0
    
    # Initial guess: graded thicknesses, thicker in the middle
    # Total budget: max_total_nm = 1000 nm
    # Distribute with a parabolic profile
    x0_base = np.linspace(0.1, 1.0, n_layers)
    x0_base = x0_base ** 2  # Make it more graded
    x0_base = x0_base / np.sum(x0_base) * max_total_nm
    x0_base = np.clip(x0_base, 0.0, max_layer_nm)
    
    best_score = np.inf
    best_x = x0_base.copy()
    
    # Multi-start Nelder-Mead
    np.random.seed(42)
    n_starts = 12
    
    for s in range(n_starts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate initial guess
        if s == 0:
            x0 = x0_base.copy()
        else:
            # Perturb the base
            perturbation = np.random.normal(0, 15.0, n_layers)
            x0 = x0_base + perturbation
            x0 = np.clip(x0, 0.0, max_layer_nm)
            # Ensure total constraint
            total = np.sum(x0)
            if total > max_total_nm:
                x0 = x0 * (max_total_nm / total)
        
        # Run Nelder-Mead
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 800,
                    'initial_simplex_size': 0.2,
                    'xatol': 0.1,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < best_score:
                x_opt = np.clip(result.x, 0.0, max_layer_nm)
                total = np.sum(x_opt)
                if total > max_total_nm:
                    x_opt = x_opt * (max_total_nm / total)
                
                score = objective(x_opt)
                if score < best_score:
                    best_score = score
                    best_x = x_opt.copy()
        except Exception:
            continue
    
    # Final validation
    best_x = np.clip(best_x, 0.0, max_layer_nm)
    total = np.sum(best_x)
    if total > max_total_nm:
        best_x = best_x * (max_total_nm / total)
    
    # Build the output
    labels = ['L' if i == n_L else 'H' for i in fixed_indices]
    thicknesses = best_x.tolist()
    
    return labels, thicknesses
# EVOLVE-BLOCK-END
