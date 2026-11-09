"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _compute_mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (%) over the band using characteristic matrix method."""
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    band_nm = problem["band_nm"]
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    # Build the stack matrices
    # Each layer has a characteristic matrix M_j = [[cos(d_j), i*sin(d_j)/n_j], [i*n_j*sin(d_j), cos(d_j)]]
    # where d_j = 2*pi*n_j*t_j/lambda
    
    # We'll compute the product of all layer matrices, then multiply by substrate impedance
    # The total characteristic matrix M = M_1 * M_2 * ... * M_N
    # Then B = M_11 + M_12, C = M_21 + M_22
    # R = |(C - B*n_s)/(C + B*n_s)|^2  ... wait, let me be careful.
    
    # Standard thin-film theory:
    # The characteristic matrix for a single layer is:
    # M_j = [[cos(delta_j), (i/n_j)sin(delta_j)], [i*n_j*sin(delta_j), cos(delta_j)]]
    # where delta_j = 2*pi*n_j*t_j/lambda
    # 
    # For N layers, the total matrix is M = M_1 * M_2 * ... * M_N
    # Let M = [[M11, M12], [M21, M22]]
    # 
    # The admittance Y = C/B where B = M11 + M12, C = M21 + M22
    # Actually, the reflectance is:
    # r = (C - B*n_s) / (C + B*n_s)  ... no wait
    # 
    # Let me recall: for normal incidence, the reflection coefficient is:
    # r = (n_0 * Y - n_s) / (n_0 * Y + n_s)  where Y is the admittance of the coating
    # 
    # The admittance Y = C/B where B and C are from the total characteristic matrix.
    # Actually, the standard formula is:
    # B = M11 + M12, C = M21 + M22
    # Y = C/B
    # r = (n_0*Y - n_s)/(n_0*Y + n_s)
    # R = |r|^2
    # 
    # But wait, I need to be more careful. Let me use the standard approach:
    # The characteristic matrix of the multilayer is M = M_1 M_2 ... M_N
    # where each M_j is a 2x2 matrix.
    # 
    # The reflection coefficient at normal incidence is:
    # r = (M21 - M22*n_s) / (M11 - M12*n_s) ... hmm, this varies by convention.
    # 
    # Let me use the most standard convention from Heavens or Macleod:
    # For a stack of layers on a substrate, the total characteristic matrix is:
    # M = [[M11, M12], [M21, M22]]
    # 
    # The complex amplitude reflection coefficient is:
    # r = (C - B*n_s) / (C + B*n_s)
    # where B = M11 + M12, C = M21 + M22
    # 
    # Wait, I think I'm confusing myself. Let me derive it properly.
    # 
    # The characteristic matrix for layer j is:
    # M_j = [[cos(delta_j), (i/n_j)sin(delta_j)], [i*n_j*sin(delta_j), cos(delta_j)]]
    # 
    # The total matrix M = M_1 * M_2 * ... * M_N
    # 
    # The relationship between the electric and magnetic field components at the front
    # (air side) and the back (substrate side) is:
    # [E_0]   = M * [E_s]
    # [H_0]         [H_s]
    # 
    # At the substrate, H_s = (n_s/c) * E_s (for normal incidence, lossless)
    # So [E_s, H_s]^T = E_s * [1, n_s/c]^T
    # 
    # At the front, the total field is the sum of incident and reflected:
    # E_0 = E_inc + E_ref
    # H_0 = (n_0/c)(E_inc - E_ref)
    # 
    # From M:
    # E_0 = M11*E_s + M12*H_s = E_s*(M11 + M12*n_s/c)
    # H_0 = M21*E_s + M22*H_s = E_s*(M21 + M22*n_s/c)
    # 
    # Let B = M11 + M12*n_s/c, C = M21 + M22*n_s/c
    # 
    # Then E_0 = E_s * B, H_0 = E_s * C
    # 
    # Also E_0 = E_inc + E_ref, H_0 = (n_0/c)(E_inc - E_ref)
    # 
    # From these: E_inc = (E_0 + (c/n_0)*H_0)/2, E_ref = (E_0 - (c/n_0)*H_0)/2
    # 
    # r = E_ref/E_inc = (E_0 - (c/n_0)*H_0)/(E_0 + (c/n_0)*H_0)
    #   = (B - (c/n_0)*C)/(B + (c/n_0)*C)
    #   = (n_0*B - c*C)/(n_0*B + c*C)
    #   = (n_0*(M11 + M12*n_s/c) - c*(M21 + M22*n_s/c))/(n_0*(M11 + M12*n_s/c) + c*(M21 + M22*n_s/c))
    #   = (n_0*M11 + n_0*M12*n_s/c - c*M21 - M22*n_s)/(n_0*M11 + n_0*M12*n_s/c + c*M21 + M22*n_s)
    # 
    # This is getting messy. Let me just use the standard formula directly.
    # 
    # Standard result: r = (Y - n_s)/(Y + n_s) where Y = C/B and B = M11 + M12, C = M21 + M22
    # Wait, that's not right either.
    # 
    # Let me just use: the admittance of the coating as seen from the air is Y = C/B
    # where B = M11 + M12, C = M21 + M22, and the reflection coefficient is:
    # r = (n_0 - Y)/(n_0 + Y) ... no, that's for the case where Y is the effective index.
    # 
    # I think the cleanest way is:
    # Y = C/B
    # r = (n_0*Y - n_s)/(n_0*Y + n_s)
    # 
    # Let me verify with a single quarter-wave layer. If we have one layer with n_1 and
    # optical thickness n_1*t = lambda/4, then delta = pi/2, cos(delta)=0, sin(delta)=1.
    # M_1 = [[0, i/n_1], [i*n_1, 0]]
    # B = 0 + i/n_1 = i/n_1
    # C = i*n_1 + 0 = i*n_1
    # Y = C/B = n_1^2
    # r = (n_0*n_1^2 - n_s)/(n_0*n_1^2 + n_s)
    # For n_0=1, n_s=1.52, n_1=sqrt(1.52)=1.233, r = (1*1.52 - 1.52)/(1*1.52 + 1.52) = 0. Good.
    # 
    # OK so the formula is: r = (n_0*Y - n_s)/(n_0*Y + n_s) where Y = C/B.
    
    n0 = ambient_index
    ns = substrate_index
    
    reflectances = []
    for wl in wavelengths:
        # Compute each layer's matrix
        M_total = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]])
        for i, (mat, t) in enumerate(zip(materials, thicknesses)):
            n_j = n_L if mat == 'L' else n_H
            delta = 2.0 * np.pi * n_j * t / wl
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            M_j = np.array([[cos_d, 1j*sin_d/n_j],
                            [1j*n_j*sin_d, cos_d]])
            M_total = M_total @ M_j
        
        M11, M12 = M_total[0,0], M_total[0,1]
        M21, M22 = M_total[1,0], M_total[1,1]
        
        B = M11 + M12
        C = M21 + M22
        
        if abs(B) < 1e-15:
            reflectances.append(1.0)  # Should not happen
            continue
        
        Y = C / B
        r = (n0 * Y - ns) / (n0 * Y + ns)
        R = np.abs(r)**2
        reflectances.append(R * 100.0)
    
    return np.mean(reflectances)

def _build_stack(base_thickness, gradient_factor, num_layers):
    """Build a periodic alternating L/H stack with gradient.
    
    The stack alternates L, H, L, H, ... starting with L.
    Each layer's thickness is base_thickness * (1 + gradient_factor * i) where i is the layer index.
    We ensure all thicknesses are in [0, 400] and total <= 1000.
    """
    materials = []
    thicknesses = []
    for i in range(num_layers):
        mat = 'L' if i % 2 == 0 else 'H'
        t = base_thickness * (1.0 + gradient_factor * i)
        if t < 0:
            t = 0.0
        if t > 400.0:
            t = 400.0
        materials.append(mat)
        thicknesses.append(t)
    
    # Check total thickness constraint
    total = sum(thicknesses)
    if total > 1000.0:
        # Scale down proportionally
        scale = 1000.0 / total
        thicknesses = [t * scale for t in thicknesses]
    
    return materials, thicknesses

def _objective(params, num_layers, problem):
    """Objective function for differential evolution."""
    base_thickness, gradient_factor = params
    materials, thicknesses = _build_stack(base_thickness, gradient_factor, num_layers)
    
    # Check constraints
    if len(materials) > 20:
        return 100.0
    if any(t < 0 or t > 400 for t in thicknesses):
        return 100.0
    if sum(thicknesses) > 1000.0:
        return 100.0
    
    return _compute_mean_reflectance(materials, thicknesses, problem)

def design(problem):
    """Design a broadband antireflection coating using periodic L/H stack with gradient.
    
    Uses differential evolution to optimize base thickness and gradient factor
    for different numbers of layers.
    """
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # Try different numbers of layers (must be even for symmetric L/H alternating)
    for num_layers in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        # Bounds for differential evolution
        # base_thickness: [0, 400]
        # gradient_factor: [-0.5, 0.5] to keep thicknesses reasonable
        bounds = [(0.0, 400.0), (-0.5, 0.5)]
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                args=(num_layers, problem),
                maxiter=100,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False
            )
            
            if result.success and result.fun < best_score:
                base_thickness, gradient_factor = result.x
                materials, thicknesses = _build_stack(base_thickness, gradient_factor, num_layers)
                
                # Final constraint check
                if (len(materials) <= 20 and 
                    all(0 <= t <= 400 for t in thicknesses) and 
                    sum(thicknesses) <= 1000.0):
                    score = _compute_mean_reflectance(materials, thicknesses, problem)
                    if score < best_score:
                        best_score = score
                        best_materials = materials
                        best_thicknesses = thicknesses
                        
        except Exception:
            continue
    
    # Fallback to simple quarter-wave if optimization fails
    if best_materials is None:
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * problem["materials"]["L"])]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
