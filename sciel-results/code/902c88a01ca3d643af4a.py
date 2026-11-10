"""Combined design program: analytical AR stack + local refinement with deadline guard."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    """Design a broadband AR coating using analytical quarter-wave stacks + local search."""
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation (401 points from 400 to 800 nm)
    lambdas = np.linspace(band[0], band[1], 401)
    lambda_c = 0.5 * (band[0] + band[1])
    
    def mean_reflectance(materials, thicknesses):
        """Compute mean reflectance in percent over the band using characteristic matrix method."""
        n_layers = [nL if m == 'L' else nH for m in materials]
        n = len(materials)
        if n == 0:
            # Bare substrate
            r = (n0 - ns) / (n0 + ns)
            return (np.abs(r) ** 2) * 100.0
        
        total_ref = 0.0
        for wl in lambdas:
            # Build total characteristic matrix
            m_total = np.array([[1.0+0j, 0.0+0j], [0.0+0j, 1.0+0j]])
            for i in range(n):
                ni = n_layers[i]
                di = thicknesses[i]
                delta = 2.0 * np.pi * ni * di / wl
                c = np.cos(delta)
                s = np.sin(delta)
                m_layer = np.array([
                    [c, 1j * s / ni],
                    [1j * ni * s, c]
                ])
                m_total = m_total @ m_layer
            
            A, B = m_total[0, 0], m_total[0, 1]
            C, D = m_total[1, 0], m_total[1, 1]
            
            # Admittance at input
            y_in = (C + D * ns) / (A + B * ns)
            r = (n0 - y_in) / (n0 + y_in)
            total_ref += np.abs(r) ** 2
        
        return (total_ref / len(lambdas)) * 100.0
    
    def is_valid(materials, thicknesses):
        """Check constraints."""
        if len(materials) != len(thicknesses):
            return False
        if len(materials) > max_layers:
            return False
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            return False
        if sum(thicknesses) > max_total_nm + 1e-9:
            return False
        return True
    
    # --- Generate candidate structures ---
    # For broadband AR, quarter-wave stacks are a good starting point.
    # We consider symmetric and asymmetric stacks.
    
    candidates = []
    
    # Single layer
    candidates.append(("L", [lambda_c / (4.0 * nL)]))
    candidates.append(("H", [lambda_c / (4.0 * nH)]))
    
    # Two layers
    candidates.append(("LH", [lambda_c / (4.0 * nL), lambda_c / (4.0 * nH)]))
    candidates.append(("HL", [lambda_c / (4.0 * nH), lambda_c / (4.0 * nL)]))
    
    # Three layers
    candidates.append(("LHL", [lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL)]))
    candidates.append(("HLH", [lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH)]))
    candidates.append(("LLH", [lambda_c / (4.0 * nL), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH)]))
    candidates.append(("HHL", [lambda_c / (4.0 * nH), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL)]))
    
    # Four layers
    candidates.append(("LHHL", [lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL)]))
    candidates.append(("HLHL", [lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH)]))
    candidates.append(("LHLH", [lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH)]))
    candidates.append(("HLHL", [lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL)]))
    
    # Five layers
    candidates.append(("LHLHL", [lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL)]))
    candidates.append(("HLHLH", [lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH)]))
    
    # Six layers
    candidates.append(("LHLHHL", [lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL)]))
    candidates.append(("HLHLHH", [lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nH)]))
    
    # Seven layers
    candidates.append(("LHLHLHL", [lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL)]))
    candidates.append(("HLHLHLH", [lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH)]))
    
    # Eight layers
    candidates.append(("LHLHLHHL", [lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL)]))
    candidates.append(("HLHLHLHH", [lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nH)]))
    
    # Nine layers
    candidates.append(("LHLHLHLHL", [lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL)]))
    candidates.append(("HLHLHLHLH", [lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH)]))
    
    # Ten layers
    candidates.append(("LHLHLHLHHL", [lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL)]))
    candidates.append(("HLHLHLHLHH", [lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nL), lambda_c / (4.0 * nH), lambda_c / (4.0 * nH)]))
    
    # Evaluate all candidates and find the best
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    for mat_str, thick in candidates:
        materials = list(mat_str)
        if not is_valid(materials, thick):
            continue
        score = mean_reflectance(materials, thick)
        if score < best_score:
            best_score = score
            best_materials = materials
            best_thicknesses = list(thick)
    
    if best_materials is None:
        # Fallback: single quarter-wave L layer
        return ["L"], [lambda_c / (4.0 * nL)]
    
    # --- Local refinement using coordinate descent ---
    current_materials = best_materials
    current_thicknesses = best_thicknesses
    current_score = best_score
    
    n_layers = len(current_materials)
    n_indices = [nL if m == 'L' else nH for m in current_materials]
    
    # Coordinate descent: perturb each thickness slightly and accept if better
    step_sizes = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1]  # nm
    improved = True
    iter_count = 0
    
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        for step in step_sizes:
            for i in range(n_layers):
                # Try increasing thickness
                new_thick = list(current_thicknesses)
                new_thick[i] += step
                if is_valid(current_materials, new_thick):
                    score = mean_reflectance(current_materials, new_thick)
                    if score < current_score - 1e-12:
                        current_thicknesses = new_thick
                        current_score = score
                        improved = True
                        iter_count += 1
                
                # Try decreasing thickness
                new_thick = list(current_thicknesses)
                new_thick[i] -= step
                if is_valid(current_materials, new_thick):
                    score = mean_reflectance(current_materials, new_thick)
                    if score < current_score - 1e-12:
                        current_thicknesses = new_thick
                        current_score = score
                        improved = True
                        iter_count += 1
                
                if (time.time() - start_time) > DEADLINE:
                    break
            if (time.time() - start_time) > DEADLINE:
                break
    
    # Final validation
    if not is_valid(current_materials, current_thicknesses):
        return best_materials, best_thicknesses
    
    return current_materials, current_thicknesses
# EVOLVE-BLOCK-END
