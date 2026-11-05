"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _compute_reflectance(wavelengths, indices, thicknesses, ambient, substrate):
    """
    Compute normal-incidence reflectance using the characteristic matrix method.
    
    Parameters:
    - wavelengths: array of wavelengths in nm
    - indices: list of refractive indices (including ambient and substrate)
    - thicknesses: list of physical thicknesses in nm (None for ambient/substrate)
    - ambient: refractive index of ambient
    - substrate: refractive index of substrate
    
    Returns:
    - reflectance array in percent
    """
    n = len(indices)
    # Build the characteristic matrix
    # For each wavelength, compute the total matrix
    # The stack is: ambient | layer1 | layer2 | ... | layerN | substrate
    
    # We'll compute this vectorized over wavelengths
    # M_total = M1 * M2 * ... * Mn
    # where each Mi is a 2x2 matrix
    
    # For a layer with index n_i and thickness d_i:
    # beta_i = 2*pi*n_i*d_i / wavelength
    # M_i = [[cos(beta_i), i*sin(beta_i)/n_i], [i*n_i*sin(beta_i), cos(beta_i)]]
    
    # For ambient and substrate, we treat them as the boundary conditions
    # Actually, the standard approach:
    # The total matrix M is the product of layer matrices
    # Then the reflectance is computed from M
    
    # Let me use the standard thin-film formula:
    # For a stack of N layers between ambient (n0) and substrate (ns):
    # The characteristic matrix is the product of individual layer matrices
    # Each layer matrix for layer j with index n_j and thickness d_j:
    #   M_j = [[cos(delta_j), (i/n_j)sin(delta_j)],
    #          [i*n_j*sin(delta_j), cos(delta_j)]]
    # where delta_j = 2*pi*n_j*d_j / wavelength
    #
    # The total matrix M = M_1 * M_2 * ... * M_N
    # M = [[m11, m12], [m21, m22]]
    #
    # Then the effective admittance Y = (n0*m11 + m21) / (m11*n0 + m22) ... 
    # Actually, let me use the standard formula:
    # Y = (n0 * m11 + m21) / (n0 * m12 + m22) ... no
    #
    # The correct formula for normal incidence:
    # The ratio of the electric field amplitudes gives:
    # r = (n0 * m11 + m21 - n0 * m12 - m22) / (n0 * m11 + m21 + n0 * m12 + m22)
    # Wait, let me be more careful.
    
    # Standard result: For a multilayer stack, if M = [[A, B], [C, D]] is the total
    # characteristic matrix (product of layer matrices), then the reflectance coefficient is:
    # r = (n0*A + C - n0*B - D) / (n0*A + C + n0*B + D)
    # Wait, I need to be careful about the convention.
    
    # Let me use a well-known formulation. The characteristic matrix for layer j is:
    # M_j = [[cos(delta_j), i*sin(delta_j)/n_j],
    #        [i*n_j*sin(delta_j), cos(delta_j)]]
    #
    # The total matrix M = M_1 * M_2 * ... * M_N
    #
    # Then:
    # Y = (C + n0*A) / (D + n0*B)  ... this is the effective admittance
    # Actually, I recall:
    # The reflection coefficient is:
    # r = (Y - n0) / (Y + n0)  where Y is the input admittance
    # And Y = (n0*m11 + m21) / (n0*m12 + m22) ... no, that's not right either.
    
    # Let me use the direct approach from thin-film optics:
    # If M = [[m11, m12], [m21, m22]] is the product of all layer matrices,
    # then the reflection coefficient at normal incidence is:
    # r = (m11 + m21/n0 - m12/n0 - m22) / (m11 + m21/n0 + m12/n0 + m22)
    # Hmm, I'm getting confused. Let me use a different, well-established formula.
    
    # Actually, the cleanest way: 
    # The characteristic matrix relates the fields:
    # [E]   = M * [E']
    # [H]       [H']
    # where the prime is at the substrate side.
    
    # At the substrate, H' = n_s * E' (for normal incidence, with appropriate convention)
    # At the air side, we have E = E_inc + E_ref, H = n0*(E_inc - E_ref)
    
    # This is getting complex. Let me just implement it numerically with a loop over layers
    # for each wavelength, which is straightforward.
    
    nw = len(wavelengths)
    nl = len(thicknesses)
    
    reflectance = np.zeros(nw)
    
    for i in range(nw):
        lam = wavelengths[i]
        # Compute the total matrix
        # Start with identity
        m11 = 1.0 + 0.0j
        m12 = 0.0 + 0.0j
        m21 = 0.0 + 0.0j
        m22 = 1.0 + 0.0j
        
        for j in range(nl):
            n_j = indices[j + 1]  # indices[0] is ambient, indices[-1] is substrate
            d_j = thicknesses[j]
            delta = 2.0 * np.pi * n_j * d_j / lam
            
            c = np.cos(delta)
            s = np.sin(delta)
            
            # Layer matrix
            L11 = c
            L12 = 1j * s / n_j
            L21 = 1j * n_j * s
            L22 = c
            
            # Multiply: M = M * L
            new_m11 = m11 * L11 + m12 * L21
            new_m12 = m11 * L12 + m12 * L22
            new_m21 = m21 * L11 + m22 * L21
            new_m22 = m21 * L12 + m22 * L22
            
            m11 = new_m11
            m12 = new_m12
            m21 = new_m21
            m22 = new_m22
        
        # Compute reflectance
        # Using the standard formula:
        # r = (m11 + m21/ambient - m12/ambient - m22) / (m11 + m21/ambient + m12/ambient + m22)
        # Wait, I need to verify this.
        
        # Actually, let me use a different approach. The input admittance Y is:
        # Y = (m21 + ambient * m11) / (m22 + ambient * m12)
        # Then r = (Y - ambient) / (Y + ambient)
        # Hmm, I'm not confident. Let me use a well-tested formula.
        
        # From Macleod "Thin-Film Optical Filters":
        # The characteristic matrix M = [[A, B], [C, D]]
        # The reflection coefficient is:
        # r = (n0*A + C - n0*B - D) / (n0*A + C + n0*B + D)
        # Wait, that doesn't look right dimensionally if B and D have different units.
        
        # Let me just use the transfer matrix method properly.
        # The key relation is:
        # [E]   = M * [E_s]
        # [H]       [H_s]
        # where E_s and H_s are at the substrate interface.
        
        # At the substrate: H_s = n_s * E_s (taking the wave traveling into substrate)
        # So: [E] = M * [1, n_s]^T * E_s
        # E = (m11 + m12*n_s) * E_s
        # H = (m21 + m22*n_s) * E_s
        
        # At the air side:
        # E = E_i + E_r
        # H = n0*(E_i - E_r)
        
        # So:
        # E_i + E_r = (m11 + m12*n_s) * E_s
        # n0*(E_i - E_r) = (m21 + m22*n_s) * E_s
        
        # Let a = m11 + m12*n_s
        # Let b = m21 + m22*n_s
        
        # E_i + E_r = a * E_s
        # n0*E_i - n0*E_r = b * E_s
        
        # Solving:
        # E_i = (a*E_s + b*E_s/n0) / 2 = E_s*(a + b/n0)/2
        # E_r = (a*E_s - b*E_s/n0) / 2 = E_s*(a - b/n0)/2
        
        # r = E_r / E_i = (a - b/n0) / (a + b/n0)
        
        a = m11 + m12 * substrate
        b = m21 + m22 * substrate
        
        r = (a - b / ambient) / (a + b / ambient)
        reflectance[i] = np.abs(r) ** 2 * 100.0
    
    return reflectance


def _objective(params, problem):
    """
    Objective function for optimization.
    params: flat array of [n_layers, thickness1, ..., thickness_n]
    where n_layers is the number of layers (integer part)
    and the rest are thicknesses.
    
    Actually, let's make it simpler: fixed number of layers, optimize thicknesses.
    """
    n_layers = params[0].astype(int)
    if n_layers < 1 or n_layers > problem['max_layers']:
        return 100.0
    
    thicknesses = params[1:1 + n_layers]
    
    # Project to feasible region
    thicknesses = np.clip(thicknesses, 0, problem['max_layer_nm'])
    
    total = np.sum(thicknesses)
    if total > problem['max_total_nm']:
        # Scale down
        if total > 0:
            thicknesses = thicknesses * (problem['max_total_nm'] / total)
        else:
            return 100.0
    
    # Assign materials: alternate L, H, L, H, ...
    materials = []
    for i in range(n_layers):
        materials.append('L' if i % 2 == 0 else 'H')
    
    # Get indices
    indices = [problem['ambient_index']]
    for mat in materials:
        indices.append(problem['materials'][mat])
    indices.append(problem['substrate_index'])
    
    # Compute reflectance
    wavelengths = np.linspace(problem['band_nm'][0], problem['band_nm'][1], 401)
    refl = _compute_reflectance(wavelengths, indices, thicknesses, 
                                 problem['ambient_index'], problem['substrate_index'])
    
    return np.mean(refl)


def design(problem):
    """Design an antireflection coating using optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    ambient = problem['ambient_index']
    substrate = problem['substrate_index']
    materials = problem['materials']
    band = problem['band_nm']
    max_layers = problem['max_layers']
    max_total = problem['max_total_nm']
    max_layer = problem['max_layer_nm']
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Start with a good initial guess: quarter-wave stack
    centre = 0.5 * (band[0] + band[1])
    nL = materials['L']
    nH = materials['H']
    
    # Try a few different starting points and optimization strategies
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # Initial guess: 2-layer quarter-wave (L, H) or (H, L)
    # Quarter-wave thickness: lambda_centre / (4 * n)
    qw_L = centre / (4.0 * nL)
    qw_H = centre / (4.0 * nH)
    
    # Try different layer counts and patterns
    for n_layers in [2, 4, 6, 8]:
        if n_layers > max_layers:
            break
        
        # Pattern: start with L, alternate
        materials_list = []
        for i in range(n_layers):
            materials_list.append('L' if i % 2 == 0 else 'H')
        
        # Initial thicknesses: quarter-wave
        init_thicknesses = []
        for i in range(n_layers):
            if materials_list[i] == 'L':
                init_thicknesses.append(qw_L)
            else:
                init_thicknesses.append(qw_H)
        
        # Check total thickness
        total = sum(init_thicknesses)
        if total > max_total:
            scale = max_total / total
            init_thicknesses = [t * scale for t in init_thicknesses]
        
        # Evaluate initial guess
        indices = [ambient] + [materials[m] for m in materials_list] + [substrate]
        refl = _compute_reflectance(wavelengths, indices, init_thicknesses, ambient, substrate)
        score = np.mean(refl)
        
        if score < best_score:
            best_score = score
            best_materials = materials_list[:]
            best_thicknesses = init_thicknesses[:]
        
        # Try Nelder-Mead optimization
        if time.time() > deadline - 20:
            break
        
        x0 = np.array(init_thicknesses)
        bounds = [(0, max_layer)] * n_layers
        
        def obj(x):
            x = np.clip(x, 0, max_layer)
            if np.sum(x) > max_total:
                x = x * (max_total / np.sum(x))
            
            indices = [ambient] + [materials[m] for m in materials_list] + [substrate]
            refl = _compute_reflectance(wavelengths, indices, x, ambient, substrate)
            return np.mean(refl)
        
        try:
            result = minimize(obj, x0, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 0.1, 'fatol': 0.01,
                                     'adaptive': True})
            
            if result.success or result.fun < best_score:
                x_opt = np.clip(result.x, 0, max_layer)
                if np.sum(x_opt) > max_total:
                    x_opt = x_opt * (max_total / np.sum(x_opt))
                
                indices = [ambient] + [materials[m] for m in materials_list] + [substrate]
                refl = _compute_reflectance(wavelengths, indices, x_opt, ambient, substrate)
                score = np.mean(refl)
                
                if score < best_score:
                    best_score = score
                    best_materials = materials_list[:]
                    best_thicknesses = list(x_opt)
        except Exception:
            pass
    
    # Try differential evolution for a larger search space
    if time.time() < deadline - 30 and max_layers >= 6:
        n_layers = min(6, max_layers)
        materials_list = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        
        def de_obj(x):
            x = np.clip(x, 0, max_layer)
            if np.sum(x) > max_total:
                x = x * (max_total / np.sum(x))
            
            indices = [ambient] + [materials[m] for m in materials_list] + [substrate]
            refl = _compute_reflectance(wavelengths, indices, x, ambient, substrate)
            return np.mean(refl)
        
        bounds = [(0, max_layer)] * n_layers
        try:
            result = differential_evolution(de_obj, bounds, seed=42, maxiter=100,
                                            tol=1e-4, polish=True, workers=1)
            if result.fun < best_score:
                x_opt = np.clip(result.x, 0, max_layer)
                if np.sum(x_opt) > max_total:
                    x_opt = x_opt * (max_total / np.sum(x_opt))
                
                indices = [ambient] + [materials[m] for m in materials_list] + [substrate]
                refl = _compute_reflectance(wavelengths, indices, x_opt, ambient, substrate)
                score = np.mean(refl)
                
                if score < best_score:
                    best_score = score
                    best_materials = materials_list[:]
                    best_thicknesses = list(x_opt)
        except Exception:
            pass
    
    # Ensure we have a valid result
    if best_materials is None:
        # Fallback to simple quarter-wave layer
        best_materials = ['L']
        best_thicknesses = [qw_L]
    
    # Final validation
    total = sum(best_thicknesses)
    if total > max_total:
        scale = max_total / total
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    for i in range(len(best_thicknesses)):
        best_thicknesses[i] = max(0, min(max_layer, best_thicknesses[i]))
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
