"""Optimized antireflection coating design using global + local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _compute_reflectance(materials, thicknesses, problem, wavelengths):
    """
    Compute mean reflectance (in percent) over given wavelengths using
    the characteristic-matrix method.
    """
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    
    # Determine which layer is L and which is H
    # materials list contains 'L' or 'H'
    n_layers = len(materials)
    
    # Precompute optical thicknesses in phase units
    # delta_i = 2*pi*n_i*d_i / lambda
    # We'll compute the characteristic matrix product for each wavelength
    
    total_reflectance = 0.0
    N = len(wavelengths)
    
    for wl in wavelengths:
        # Compute optical phase for each layer
        # M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
        # Product M = M_1 * M_2 * ... * M_N
        # Y = n_amb * (M_11 + M_12*n_sub) / (M_21 + M_22*n_sub)  -- wait, let me be careful
        
        # The characteristic matrix for layer i:
        # M_i = [[cos(delta_i), (i/n_i)*sin(delta_i)], [i*n_i*sin(delta_i), cos(delta_i)]]
        # where delta_i = 2*pi*n_i*d_i/lambda
        
        # The overall matrix M = M_1 * M_2 * ... * M_N
        # Then:
        # Y_0 = n_amb * (M_11 + M_12 * n_sub) / (M_21 + M_22 * n_sub)  -- no
        
        # Actually, the admittance at the input side:
        # B = M_11 + M_12 * n_sub
        # C = M_21 + M_22 * n_sub
        # Y = (n_amb * B) / C  -- no, that's not right either
        
        # Let me use the standard formula:
        # The characteristic matrix M relates the fields:
        # [E, H]^T at air side = M * [E, H]^T at substrate side
        # M = [[B, C], [D, E]] (using different notation)
        
        # Standard: M = [[M11, M12], [M21, M22]]
        # Y = (n_amb * (M11 + M12 * n_sub)) / (M21 + M22 * n_sub)  -- no
        
        # Actually the correct formula for the input admittance:
        # Y_in = (n_amb * B) / C where B and C come from... 
        
        # Let me just use the standard transfer matrix approach:
        # For each layer, the transfer matrix is:
        # T_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
        # The total transfer matrix T = T_1 * T_2 * ... * T_N
        # Then the reflection coefficient:
        # r = (T_11 + T_12*n_sub - T_21 - T_22*n_sub) / (T_11 + T_12*n_sub + T_21 + T_22*n_sub)
        # Wait, I need to be more careful.
        
        # The standard result: if the transfer matrix is M = [[m11, m12], [m21, m22]],
        # and the substrate has index n_s, ambient n_0, then:
        # r = (n_0*(m11 + m12*n_s) - (m21 + m22*n_s)) / (n_0*(m11 + m12*n_s) + (m21 + m22*n_s))
        # Hmm, let me think again.
        
        # The boundary condition at the substrate:
        # E_s + E_r_s = E_inc (but no reflection from substrate side in our setup)
        # Actually, the standard formula is:
        # Y = (m11*n_s + m12) / (m21*n_s + m22)  -- this is the effective admittance
        # Then r = (n_0 - Y) / (n_0 + Y)
        
        # Let me verify: for a single quarter-wave layer of index n_q on substrate n_s:
        # delta = pi/2, so cos=0, sin=1
        # T = [[0, i/n_q], [i*n_q, 0]]
        # Y = (0*n_s + i/n_q) / (i*n_q*n_s + 0) = (i/n_q)/(i*n_q*n_s) = 1/(n_q^2*n_s)
        # That doesn't seem right. Let me reconsider.
        
        # Actually, I think the correct formula involves:
        # B = m11 + m12 * n_s
        # C = m21 + m22 * n_s
        # Y = n_0 * B / C  -- no
        
        # Let me use a cleaner derivation. The transfer matrix relates:
        # [E_0, H_0]^T = M * [E_s, H_s]^T
        # where E_s, H_s are at the substrate interface.
        
        # At the substrate, for a plane wave incident from the coating side:
        # E_s = E_t (transmitted)
        # H_s = n_s * E_t / n_0  (in normalized units)
        
        # So H_s = (n_s/n_0) * E_s
        # [E_0, H_0]^T = M * [E_s, (n_s/n_0)*E_s]^T = E_s * M * [1, n_s/n_0]^T
        # E_0 = E_s * (m11 + m12*n_s/n_0)
        # H_0 = E_s * (m21 + m22*n_s/n_0)
        
        # The reflection coefficient r = (E_0 - H_0/n_0) / (E_0 + H_0/n_0)  -- in terms of E and H
        # Wait, in the ambient medium, E = E_i + E_r, H = (n_0)(E_i - E_r)
        # So E_i = (E + H/n_0)/2, E_r = (E - H/n_0)/2
        # r = E_r/E_i = (E - H/n_0)/(E + H/n_0)
        
        # So r = (E_0 - H_0/n_0) / (E_0 + H_0/n_0)
        # = (m11 + m12*n_s/n_0 - (m21 + m22*n_s/n_0)/n_0) / (m11 + m12*n_s/n_0 + (m21 + m22*n_s/n_0)/n_0)
        # = (n_0*m11 + m12*n_s - m21 - m22*n_s/n_0... )
        
        # Let me just compute numerically with a known case to verify.
        
        # For simplicity, let me compute the matrix product and then use:
        # B = m11 + m12 * n_s
        # C = m21 + m22 * n_s  
        # r = (n_0*B - C) / (n_0*B + C)
        
        # Test: bare glass (no layers), n_0=1, n_s=1.52
        # M = identity, B = 1, C = 1.52
        # r = (1 - 1.52)/(1 + 1.52) = -0.52/2.52 = -0.2063
        # R = |r|^2 = 0.0426 = 4.26% ✓
        
        # Good, so the formula is:
        # r = (n_0*B - C) / (n_0*B + C) where B = m11 + m12*n_s, C = m21 + m22*n_s
        
        # Now let's compute the product of transfer matrices.
        
        m11, m12 = 1.0+0.0j, 0.0+0.0j
        m21, m22 = 0.0+0.0j, 1.0+0.0j
        
        for i in range(n_layers):
            n_i = n_L if materials[i] == 'L' else n_H
            d_i = thicknesses[i]
            delta_i = 2.0 * np.pi * n_i * d_i / wl
            
            c = np.cos(delta_i)
            s = np.sin(delta_i)
            
            # T_i = [[c, i*s/n_i], [i*n_i*s, c]]
            t11 = c
            t12 = 1j * s / n_i
            t21 = 1j * n_i * s
            t22 = c
            
            # M_new = M * T_i
            new_m11 = m11 * t11 + m12 * t21
            new_m12 = m11 * t12 + m12 * t22
            new_m21 = m21 * t11 + m22 * t21
            new_m22 = m21 * t12 + m22 * t22
            
            m11, m12, m21, m22 = new_m11, new_m12, new_m21, new_m22
        
        B = m11 + m12 * n_sub
        C = m21 + m22 * n_sub
        r = (n_amb * B - C) / (n_amb * B + C)
        R = np.abs(r) ** 2
        total_reflectance += R * 100.0  # in percent
    
    return total_reflectance / N


def _compute_reflectance_fast(materials, thicknesses, problem, wavelengths):
    """Vectorized version for speed."""
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    
    n_layers = len(materials)
    n_wl = len(wavelengths)
    
    # Precompute indices
    indices = np.array([n_L if m == 'L' else n_H for m in materials])
    
    total_reflectance = 0.0
    
    for wl in wavelengths:
        m11 = 1.0 + 0.0j
        m12 = 0.0 + 0.0j
        m21 = 0.0 + 0.0j
        m22 = 1.0 + 0.0j
        
        for i in range(n_layers):
            n_i = indices[i]
            d_i = thicknesses[i]
            delta_i = 2.0 * np.pi * n_i * d_i / wl
            
            c = np.cos(delta_i)
            s = np.sin(delta_i)
            
            t11 = c
            t12 = 1j * s / n_i
            t21 = 1j * n_i * s
            t22 = c
            
            new_m11 = m11 * t11 + m12 * t21
            new_m12 = m11 * t12 + m12 * t22
            new_m21 = m21 * t11 + m22 * t21
            new_m22 = m21 * t12 + m22 * t22
            
            m11, m12, m21, m22 = new_m11, new_m12, new_m21, new_m22
        
        B = m11 + m12 * n_sub
        C = m21 + m22 * n_sub
        r = (n_amb * B - C) / (n_amb * B + C)
        R = np.abs(r) ** 2
        total_reflectance += R * 100.0
    
    return total_reflectance / n_wl


def design(problem):
    """
    Design an antireflection coating using global search (random restarts + local optimization).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_sub = problem["substrate_index"]
    n_amb = problem["ambient_index"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # Try different numbers of layers and patterns
    # Start with some good initial guesses
    
    def evaluate(materials, thicknesses):
        nonlocal best_score, best_materials, best_thicknesses
        # Check constraints
        if len(materials) != len(thicknesses):
            return float('inf')
        if len(materials) > max_layers:
            return float('inf')
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            return float('inf')
        if sum(thicknesses) > max_total_nm + 1e-6:
            return float('inf')
        
        score = _compute_reflectance(materials, thicknesses, problem, wavelengths)
        if score < best_score:
            best_score = score
            best_materials = materials[:]
            best_thicknesses = thicknesses[:]
        return score
    
    def local_optimize(materials, thicknesses):
        """Local optimization using Nelder-Mead on thicknesses."""
        nonlocal best_score, best_materials, best_thicknesses
        
        n = len(thicknesses)
        if n == 0:
            return
        
        # Objective function
        def objective(x):
            x = np.clip(x, 0, max_layer_nm)
            if np.sum(x) > max_total_nm:
                return 1000.0
            return _compute_reflectance(materials, list(x), problem, wavelengths)
        
        x0 = np.array(thicknesses, dtype=float)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 2000, 'xatol': 0.1, 'fatol': 1e-4})
            if result.success or result.fun < best_score:
                x_opt = np.clip(result.x, 0, max_layer_nm)
                if np.sum(x_opt) <= max_total_nm + 1e-6:
                    evaluate(materials, list(x_opt))
        except:
            pass
    
    # Strategy: Try various layer counts and patterns
    # Good patterns: alternating H/L starting with L (to match air better) or H
    
    patterns_to_try = []
    
    # Single layer
    patterns_to_try.append(["L"])
    patterns_to_try.append(["H"])
    
    # Two layers
    patterns_to_try.append(["L", "H"])
    patterns_to_try.append(["H", "L"])
    
    # Three layers
    patterns_to_try.append(["L", "H", "L"])
    patterns_to_try.append(["H", "L", "H"])
    
    # Four layers
    patterns_to_try.append(["L", "H", "L", "H"])
    patterns_to_try.append(["H", "L", "H", "L"])
    
    # Five layers
    patterns_to_try.append(["L", "H", "L", "H", "L"])
    patterns_to_try.append(["H", "L", "H", "L", "H"])
    
    # Six layers
    patterns_to_try.append(["L", "H", "L", "H", "L", "H"])
    patterns_to_try.append(["H", "L", "H", "L", "H", "L"])
    
    # Eight layers
    patterns_to_try.append(["L", "H", "L", "H", "L", "H", "L", "H"])
    patterns_to_try.append(["H", "L", "H", "L", "H", "L", "H", "L"])
    
    # Ten layers
    patterns_to_try.append(["L", "H", "L", "H", "L", "H", "L", "H", "L", "H"])
    patterns_to_try.append(["H", "L", "H", "L", "H", "L", "H", "L", "H", "L"])
    
    # Twelve layers
    patterns_to_try.append(["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"])
    patterns_to_try.append(["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"])
    
    # Fifteen layers
    patterns_to_try.append(["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"])
    patterns_to_try.append(["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"])
    
    # Twenty layers
    patterns_to_try.append(["L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H"])
    patterns_to_try.append(["H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L", "H", "L"])
    
    # For each pattern, try initial guesses based on quarter-wave at center wavelength
    centre_wl = 0.5 * (band[0] + band[1])
    
    for pattern in patterns_to_try:
        if time.time() > deadline - 10.0:
            break
        
        n = len(pattern)
        if n > max_layers:
            continue
        
        # Initial guess: quarter-wave at center wavelength
        init_thick = []
        for m in pattern:
            n_i = n_L if m == 'L' else n_H
            d = centre_wl / (4.0 * n_i)
            init_thick.append(d)
        
        # Check if within total thickness limit
        if sum(init_thick) > max_total_nm:
            # Scale down
            scale = max_total_nm / sum(init_thick)
            init_thick = [d * scale for d in init_thick]
        
        evaluate(pattern, init_thick)
        
        # Local optimization
        if time.time() < deadline - 5.0:
            local_optimize(pattern, init_thick)
        
        # Also try a few random perturbations
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            random_thick = [np.random.uniform(0, max_layer_nm) for _ in range(n)]
            if sum(random_thick) <= max_total_nm:
                evaluate(pattern, random_thick)
                if time.time() < deadline - 5.0:
                    local_optimize(pattern, random_thick)
    
    # Try some random patterns with varying layer counts
    for _ in range(50):
        if time.time() > deadline - 15.0:
            break
        
        n = np.random.randint(2, min(max_layers, 15) + 1)
        # Random pattern, but avoid too many consecutive same layers
        pattern = []
        for i in range(n):
            if i == 0:
                pattern.append(np.random.choice(["L", "H"]))
            else:
                if np.random.random() < 0.7:
                    # Alternate
                    pattern.append("H" if pattern[-1] == "L" else "L")
                else:
                    pattern.append(np.random.choice(["L", "H"]))
        
        init_thick = []
        for m in pattern:
            n_i = n_L if m == 'L' else n_H
            d = np.random.uniform(50, min(max_layer_nm, max_total_nm / n))
            init_thick.append(d)
        
        if sum(init_thick) <= max_total_nm:
            evaluate(pattern, init_thick)
            if time.time() < deadline - 5.0:
                local_optimize(pattern, init_thick)
    
    # Final check
    if best_materials is None:
        # Fallback: single quarter-wave L layer
        centre = 0.5 * (band[0] + band[1])
        return ["L"], [centre / (4.0 * n_L)]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
