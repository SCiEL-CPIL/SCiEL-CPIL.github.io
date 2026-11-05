"""Broadband antireflection coating design via multi-start local optimization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _char_matrix(n, d, wl):
    """Compute the characteristic matrix for a single layer."""
    beta = 2.0 * math.pi * n * d / wl
    cos_b = math.cos(beta)
    sin_b = math.sin(beta)
    # For s-polarization (normal incidence, same for p and s)
    # M = [[cos(beta), i*sin(beta)/n], [i*n*sin(beta), cos(beta)]]
    return np.array([[cos_b, 1j * sin_b / n],
                     [1j * n * sin_b, cos_b]], dtype=complex)

def _compute_reflectance(materials, thicknesses, wl, n_ambient, n_substrate):
    """Compute reflectance at a single wavelength using characteristic matrix method."""
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return r * r
    
    # Compute total characteristic matrix
    M11, M12 = 1.0 + 0.0j, 0.0 + 0.0j
    M21, M22 = 0.0 + 0.0j, 1.0 + 0.0j
    
    for i in range(n_layers):
        n = n_substrate if i == n_layers else None  # placeholder
        # Actually we need the material index for layer i
        pass
    
    # Let me redo this properly
    M11, M12 = 1.0 + 0.0j, 0.0 + 0.0j
    M21, M22 = 0.0 + 0.0j, 1.0 + 0.0j
    
    for i in range(n_layers):
        n = problem_materials[materials[i]]
        d = thicknesses[i]
        beta = 2.0 * math.pi * n * d / wl
        cos_b = math.cos(beta)
        sin_b = math.sin(beta)
        
        # Layer matrix
        L11 = cos_b
        L12 = 1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        # Multiply M = M * L
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Compute reflectance
    # Y = (M21 + M22 * n_substrate) / (M11 + M12 * n_substrate)
    # r = (n_ambient * Y - n_substrate) / (n_ambient * Y + n_substrate)
    # Wait, let me use the standard formula
    # B = M11 + M12 * n_substrate
    # C = M21 + M22 * n_substrate
    # Y = C / B (admittance)
    # r = (n_ambient - Y) / (n_ambient + Y)
    
    B = M11 + M12 * n_substrate
    C = M21 + M22 * n_substrate
    Y = C / B
    r = (n_ambient - Y) / (n_ambient + Y)
    return abs(r) ** 2

def _mean_reflectance_percent(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    total_r = 0.0
    for wl in wavelengths:
        r = _compute_reflectance_at(materials, thicknesses, wl, n_ambient, n_substrate, n_L, n_H)
        total_r += r
    
    return 100.0 * total_r / len(wavelengths)

def _compute_reflectance_at(materials, thicknesses, wl, n_ambient, n_substrate, n_L, n_H):
    """Compute reflectance at a single wavelength."""
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return r * r
    
    M11, M12 = 1.0 + 0.0j, 0.0 + 0.0j
    M21, M22 = 0.0 + 0.0j, 1.0 + 0.0j
    
    for i in range(n_layers):
        n = n_L if materials[i] == 'L' else n_H
        d = thicknesses[i]
        beta = 2.0 * math.pi * n * d / wl
        cos_b = math.cos(beta)
        sin_b = math.sin(beta)
        
        L11 = cos_b
        L12 = 1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    B = M11 + M12 * n_substrate
    C = M21 + M22 * n_substrate
    Y = C / B
    r = (n_ambient - Y) / (n_ambient + Y)
    return abs(r) ** 2

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True

def _evaluate(materials, thicknesses, problem):
    """Evaluate the mean reflectance percent for a design."""
    if not _is_valid(materials, thicknesses, problem):
        return float('inf')
    return _mean_reflectance_percent(materials, thicknesses, problem)

def _local_optimize(materials, thicknesses, problem, deadline, max_steps=200):
    """Simple coordinate descent local optimization."""
    best_materials = list(materials)
    best_thicknesses = list(thicknesses)
    best_score = _evaluate(best_materials, best_thicknesses, problem)
    
    if best_score == float('inf'):
        return best_materials, best_thicknesses, best_score
    
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    step = 10.0  # initial step in nm
    n_layers = len(best_thicknesses)
    
    for _ in range(max_steps):
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        for i in range(n_layers):
            # Try increasing
            new_t = list(best_thicknesses)
            new_t[i] = min(new_t[i] + step, max_layer_nm)
            if sum(new_t) <= max_total_nm:
                score = _evaluate(best_materials, new_t, problem)
                if score < best_score:
                    best_thicknesses = new_t
                    best_score = score
                    improved = True
            
            # Try decreasing
            new_t = list(best_thicknesses)
            new_t[i] = max(new_t[i] - step, 0.0)
            score = _evaluate(best_materials, new_t, problem)
            if score < best_score:
                best_thicknesses = new_t
                best_score = score
                improved = True
        
        if not improved:
            step *= 0.7
            if step < 0.5:
                break
    
    return best_materials, best_thicknesses, best_score

def _try_add_layer(materials, thicknesses, problem, deadline):
    """Try adding a layer and optimizing."""
    max_layers = problem.get("max_layers", 20)
    if len(materials) >= max_layers:
        return None
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    candidates = []
    # Try adding L or H at various positions
    for pos in range(len(materials) + 1):
        for mat in ['L', 'H']:
            n = n_L if mat == 'L' else n_H
            # Quarter-wave thickness at centre wavelength
            d = centre / (4.0 * n)
            new_materials = materials[:pos] + [mat] + materials[pos:]
            new_thicknesses = thicknesses[:pos] + [d] + thicknesses[pos:]
            
            if _is_valid(new_materials, new_thicknesses, problem):
                score = _evaluate(new_materials, new_thicknesses, problem)
                if score < float('inf'):
                    candidates.append((score, new_materials, new_thicknesses))
    
    if not candidates:
        return None
    
    candidates.sort(key=lambda x: x[0])
    best_score, best_materials, best_thicknesses = candidates[0]
    
    # Local optimize the best candidate
    best_materials, best_thicknesses, best_score = _local_optimize(
        best_materials, best_thicknesses, problem, deadline, max_steps=50)
    
    return best_materials, best_thicknesses, best_score

def _try_remove_layer(materials, thicknesses, problem, deadline):
    """Try removing a layer and optimizing."""
    if len(materials) <= 1:
        return None
    
    candidates = []
    for i in range(len(materials)):
        new_materials = materials[:i] + materials[i+1:]
        new_thicknesses = thicknesses[:i] + thicknesses[i+1:]
        
        if _is_valid(new_materials, new_thicknesses, problem):
            score = _evaluate(new_materials, new_thicknesses, problem)
            if score < float('inf'):
                candidates.append((score, new_materials, new_thicknesses))
    
    if not candidates:
        return None
    
    candidates.sort(key=lambda x: x[0])
    best_score, best_materials, best_thicknesses = candidates[0]
    
    best_materials, best_thicknesses, best_score = _local_optimize(
        best_materials, best_thicknesses, problem, deadline, max_steps=50)
    
    return best_materials, best_thicknesses, best_score

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    # Start with a simple quarter-wave design
    best_materials = ["L"]
    best_thicknesses = [centre / (4.0 * n_L)]
    best_score = _evaluate(best_materials, best_thicknesses, problem)
    
    # Iteratively improve
    for iteration in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Try adding a layer
        result = _try_add_layer(best_materials, best_thicknesses, problem, deadline)
        if result is not None:
            mat, th, score = result
            if score < best_score:
                best_materials, best_thicknesses, best_score = mat, th, score
        
        # Try removing a layer
        result = _try_remove_layer(best_materials, best_thicknesses, problem, deadline)
        if result is not None:
            mat, th, score = result
            if score < best_score:
                best_materials, best_thicknesses, best_score = mat, th, score
        
        # Local optimize current best
        if time.time() > deadline - 10.0:
            break
        best_materials, best_thicknesses, best_score = _local_optimize(
            best_materials, best_thicknesses, problem, deadline, max_steps=100)
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
