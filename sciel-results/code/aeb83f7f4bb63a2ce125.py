"""Broadband antireflection coating design using characteristic-matrix method and local search."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _char_matrix(n, d, lam, n0=1.0):
    """
    Compute the 2x2 characteristic matrix for a single layer.
    n: refractive index of the layer
    d: physical thickness in nm
    lam: wavelength in nm
    n0: ambient index (for normalization of optical admittance)
    """
    beta = 2.0 * np.pi * n * d / lam
    c = np.cos(beta)
    s = np.sin(beta)
    # For normal incidence:
    # M = [[cos(beta), i*sin(beta)/n], [i*n*sin(beta), cos(beta)]]
    # We work with the normalized admittance formulation.
    # Actually, the standard characteristic matrix for normal incidence is:
    # M = [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
    # where delta = 2*pi*n*d/lam
    return np.array([[c, 1j*s/n], [1j*n*s, c]], dtype=complex)


def _reflectance_for_stack(materials, thicknesses, lambdas, n_ambient=1.0, n_substrate=1.52):
    """
    Compute the reflectance spectrum for a given stack.
    materials: list of 'L' or 'H'
    thicknesses: list of physical thicknesses in nm
    lambdas: array of wavelengths in nm
    Returns: array of reflectance values (0 to 1) for each wavelength.
    """
    n_vals = []
    for m in materials:
        if m == 'L':
            n_vals.append(1.38)
        elif m == 'H':
            n_vals.append(2.35)
        else:
            n_vals.append(1.0)
    
    n_layers = len(materials)
    reflectances = np.zeros(len(lambdas))
    
    for i, lam in enumerate(lambdas):
        # Start with the identity matrix
        M_total = np.eye(2, dtype=complex)
        
        # Multiply by each layer's matrix from air side to substrate
        for j in range(n_layers):
            M_layer = _char_matrix(n_vals[j], thicknesses[j], lam, n_ambient)
            M_total = M_total @ M_layer
        
        # The admittance at the air side is:
        # Y = (M[0,0] + M[0,1]*n_sub) / (M[1,0] + M[1,1]*n_sub)
        # But we need to be careful with the normalization.
        # Actually, the standard formula is:
        # Y = (M11 + M12 * n_s) / (M21 + M22 * n_s)
        # where n_s is the substrate index, and the admittance is normalized to n_ambient.
        
        M11, M12 = M_total[0, 0], M_total[0, 1]
        M21, M22 = M_total[1, 0], M_total[1, 1]
        
        # The input admittance (normalized to ambient)
        Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
        
        # The reflectance is:
        # R = |(Y - 1)/(Y + 1)|^2  (when normalized to ambient index = 1)
        # More generally: R = |(Y - n_ambient)/(Y + n_ambient)|^2 / n_ambient^2
        # But if Y is already normalized to n_ambient, then:
        # R = |(Y - 1)/(Y + 1)|^2
        
        # Let's use the general formula:
        # The reflection coefficient r = (n_ambient - Y) / (n_ambient + Y)
        # R = |r|^2
        
        r = (n_ambient - Y) / (n_ambient + Y)
        reflectances[i] = np.abs(r)**2
    
    return reflectances


def _mean_reflectance(materials, thicknesses, problem):
    """
    Compute the mean reflectance in percent over the band.
    """
    band = problem["band_nm"]
    lambdas = np.linspace(band[0], band[1], 401)
    R = _reflectance_for_stack(materials, thicknesses, lambdas, 
                                n_ambient=problem["ambient_index"],
                                n_substrate=problem["substrate_index"])
    return 100.0 * np.mean(R)


def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if len(materials) == 0:
        return True
    
    total = sum(thicknesses)
    if total > max_total_nm + 1e-6:
        return False
    
    for t in thicknesses:
        if t < -1e-6 or t > max_layer_nm + 1e-6:
            return False
    
    for m in materials:
        if m not in ['L', 'H']:
            return False
    
    return True


def _reflectance_vectorized(materials, thicknesses, lambdas, n_ambient, n_substrate):
    """
    Vectorized reflectance computation for speed.
    """
    n_vals = np.array([1.38 if m == 'L' else 2.35 for m in materials])
    d_vals = np.array(thicknesses)
    
    n_layers = len(materials)
    reflectances = np.zeros(len(lambdas))
    
    for i, lam in enumerate(lambdas):
        M_total = np.eye(2, dtype=complex)
        for j in range(n_layers):
            beta = 2.0 * np.pi * n_vals[j] * d_vals[j] / lam
            c = np.cos(beta)
            s = np.sin(beta)
            M_layer = np.array([[c, 1j*s/n_vals[j]], [1j*n_vals[j]*s, c]], dtype=complex)
            M_total = M_total @ M_layer
        
        M11, M12 = M_total[0, 0], M_total[0, 1]
        M21, M22 = M_total[1, 0], M_total[1, 1]
        
        Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
        r = (n_ambient - Y) / (n_ambient + Y)
        reflectances[i] = np.abs(r)**2
    
    return reflectances


def design(problem):
    """
    Design a broadband antireflection coating.
    Uses a combination of analytical initialization and local search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lambdas = np.linspace(band[0], band[1], 401)
    
    # Helper to evaluate a design
    def evaluate(materials, thicknesses):
        if not _is_valid(materials, thicknesses, problem):
            return float('inf')
        R = _reflectance_vectorized(materials, thicknesses, lambdas, n_ambient, n_substrate)
        return 100.0 * np.mean(R)
    
    # Start with a reasonable initial design
    # A common approach: use a quarter-wave stack at the center wavelength
    center_lam = 0.5 * (band[0] + band[1])
    
    # Try different numbers of layers and patterns
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Strategy 1: Analytical quarter-wave stack
    # For a broadband AR coating, we can try a stack of alternating L and H layers
    # with optical thicknesses near lambda/4 at the center wavelength.
    
    # Let's try a few initial patterns
    patterns = [
        # Single layer
        (['L'], [center_lam / (4.0 * n_L)]),
        (['H'], [center_lam / (4.0 * n_H)]),
        
        # Two layers
        (['L', 'H'], [center_lam / (4.0 * n_L), center_lam / (4.0 * n_H)]),
        (['H', 'L'], [center_lam / (4.0 * n_H), center_lam / (4.0 * n_L)]),
        
        # Three layers
        (['L', 'H', 'L'], [center_lam / (4.0 * n_L), center_lam / (4.0 * n_H), center_lam / (4.0 * n_L)]),
        (['H', 'L', 'H'], [center_lam / (4.0 * n_H), center_lam / (4.0 * n_L), center_lam / (4.0 * n_H)]),
        
        # Four layers
        (['L', 'H', 'L', 'H'], [center_lam / (4.0 * n_L), center_lam / (4.0 * n_H), center_lam / (4.0 * n_L), center_lam / (4.0 * n_H)]),
        (['H', 'L', 'H', 'L'], [center_lam / (4.0 * n_H), center_lam / (4.0 * n_L), center_lam / (4.0 * n_H), center_lam / (4.0 * n_L)]),
        
        # Five layers
        (['L', 'H', 'L', 'H', 'L'], [center_lam / (4.0 * n_L), center_lam / (4.0 * n_H), center_lam / (4.0 * n_L), center_lam / (4.0 * n_H), center_lam / (4.0 * n_L)]),
        (['H', 'L', 'H', 'L', 'H'], [center_lam / (4.0 * n_H), center_lam / (4.0 * n_L), center_lam / (4.0 * n_H), center_lam / (4.0 * n_L), center_lam / (4.0 * n_H)]),
        
        # Six layers
        (['L', 'H', 'L', 'H', 'L', 'H'], [center_lam / (4.0 * n_L), center_lam / (4.0 * n_H), center_lam / (4.0 * n_L), center_lam / (4.0 * n_H), center_lam / (4.0 * n_L), center_lam / (4.0 * n_H)]),
        (['H', 'L', 'H', 'L', 'H', 'L'], [center_lam / (4.0 * n_H), center_lam / (4.0 * n_L), center_lam / (4.0 * n_H), center_lam / (4.0 * n_L), center_lam / (4.0 * n_H), center_lam / (4.0 * n_L)]),
    ]
    
    for mats, thicks in patterns:
        score = evaluate(mats, thicks)
        if score < best_score:
            best_score = score
            best_materials = mats[:]
            best_thicknesses = thicks[:]
    
    # Now do local search to improve the best design
    if best_materials is not None:
        # Local search: perturb thicknesses and try adding/removing layers
        current_materials = best_materials[:]
        current_thicknesses = best_thicknesses[:]
        current_score = best_score
        
        # Try to improve by optimizing thicknesses
        # Use a simple gradient-free local search
        
        step_size = 50.0  # nm
        for iteration in range(200):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            n_layers = len(current_materials)
            
            # Try perturbing each thickness
            for i in range(n_layers):
                for delta in [-step_size, step_size, -step_size/2, step_size/2]:
                    new_thicks = current_thicknesses[:]
                    new_thicks[i] = max(0.0, min(max_layer_nm, new_thicks[i] + delta))
                    
                    if not _is_valid(current_materials, new_thicks, problem):
                        continue
                    
                    score = evaluate(current_materials, new_thicks)
                    if score < current_score - 1e-8:
                        current_thicknesses = new_thicks
                        current_score = score
                        improved = True
            
            # Try adding a layer
            if len(current_materials) < max_layers:
                for pos in range(len(current_materials) + 1):
                    for new_mat in ['L', 'H']:
                        new_mats = current_materials[:]
                        new_mats.insert(pos, new_mat)
                        
                        # Try a few initial thicknesses for the new layer
                        for init_t in [50.0, 100.0, 150.0, 200.0]:
                            new_thicks = current_thicknesses[:]
                            new_thicks.insert(pos, init_t)
                            
                            if not _is_valid(new_mats, new_thicks, problem):
                                continue
                            
                            score = evaluate(new_mats, new_thicks)
                            if score < current_score - 1e-8:
                                current_materials = new_mats
                                current_thicknesses = new_thicks
                                current_score = score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Try removing a layer if it improves
            if len(current_materials) > 1:
                for i in range(len(current_materials)):
                    new_mats = current_materials[:]
                    new_mats.pop(i)
                    new_thicks = current_thicknesses[:]
                    new_thicks.pop(i)
                    
                    if not _is_valid(new_mats, new_thicks, problem):
                        continue
                    
                    score = evaluate(new_mats, new_thicks)
                    if score < current_score - 1e-8:
                        current_materials = new_mats
                        current_thicknesses = new_thicks
                        current_score = score
                        improved = True
                        break
            
            # Reduce step size if no improvement
            if not improved:
                step_size *= 0.8
                if step_size < 1.0:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_materials = current_materials[:]
                best_thicknesses = current_thicknesses[:]
        
        # Final high-precision polish
        if time.time() < deadline - 2.0:
            # Do a finer search around the best point
            for i in range(len(best_materials)):
                for delta in np.linspace(-20, 20, 21):
                    new_thicks = best_thicknesses[:]
                    new_thicks[i] = max(0.0, min(max_layer_nm, new_thicks[i] + delta))
                    if not _is_valid(best_materials, new_thicks, problem):
                        continue
                    score = evaluate(best_materials, new_thicks)
                    if score < best_score:
                        best_score = score
                        best_thicknesses = new_thicks
    
    # If we didn't find a good design, fall back to the simple quarter-wave layer
    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [center_lam / (4.0 * n_L)]
    
    # Ensure constraints are satisfied
    for i in range(len(best_thicknesses)):
        best_thicknesses[i] = max(0.0, min(max_layer_nm, best_thicknesses[i]))
    
    # Check total thickness
    total = sum(best_thicknesses)
    if total > max_total_nm:
        # Scale down proportionally
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
