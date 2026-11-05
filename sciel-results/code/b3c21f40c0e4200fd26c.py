"""Broadband antireflection coating design via local optimization over fixed alternating stacks."""
import time
import numpy as np

def _fast_reflectance(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls):
    """
    Compute mean reflectance in percent for a single design across all wavelengths.
    Vectorized over wavelengths for speed.
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return (r * r) * 100.0
    
    # Initialize M for all wavelengths
    M11 = np.ones_like(wls, dtype=complex)
    M12 = np.zeros_like(wls, dtype=complex)
    M21 = np.zeros_like(wls, dtype=complex)
    M22 = np.ones_like(wls, dtype=complex)
    
    for i in range(n_layers):
        if materials[i] == 'H':
            n = n_H
        else:
            n = n_L
            
        d = thicknesses[i]
        beta = 2.0 * np.pi * n * d / wls
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Layer matrix components
        L11 = cos_b
        L12 = -1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        # M = M @ L
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Admittance Y
    Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = (r * r.conjugate()).real
    return np.mean(R) * 100.0

def design(problem):
    """Design a broadband antireflection coating using local optimization over alternating stacks."""
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    wl_min, wl_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 401 equally spaced wavelengths
    wls = np.linspace(wl_min, wl_max, 401)
    
    def is_valid(materials, thicknesses):
        if len(materials) > max_layers:
            return False
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            return False
        if sum(thicknesses) > max_total_nm:
            return False
        return True

    def evaluate(materials, thicknesses):
        if not is_valid(materials, thicknesses):
            return float('inf')
        return _fast_reflectance(thicknesses, materials, n_ambient, n_substrate, n_L, n_H, wls)
    
    # Strategy: Try different alternating stack patterns and optimize their thicknesses
    # This is a different approach from GA - we fix the material sequence and optimize thicknesses
    
    # Generate candidate material sequences (alternating patterns)
    candidate_patterns = []
    
    # Start with L (air side)
    for n_layers in range(1, max_layers + 1):
        pattern = []
        for i in range(n_layers):
            if i % 2 == 0:
                pattern.append('L')
            else:
                pattern.append('H')
        candidate_patterns.append(pattern)
        
        # Also try starting with H
        pattern_h = []
        for i in range(n_layers):
            if i % 2 == 0:
                pattern_h.append('H')
            else:
                pattern_h.append('L')
        if pattern_h not in candidate_patterns:
            candidate_patterns.append(pattern_h)
    
    # Also add some non-alternating patterns that might work well
    # H-L-H-L-H (5 layers)
    candidate_patterns.append(['H', 'L', 'H', 'L', 'H'])
    candidate_patterns.append(['L', 'H', 'L', 'H', 'L'])
    # 7 layer patterns
    candidate_patterns.append(['H', 'L', 'H', 'L', 'H', 'L', 'H'])
    candidate_patterns.append(['L', 'H', 'L', 'H', 'L', 'H', 'L'])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_patterns = []
    for p in candidate_patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    # For each pattern, do local optimization of thicknesses
    for pattern in unique_patterns:
        if time.time() > deadline:
            break
            
        n_layers = len(pattern)
        if n_layers > max_layers:
            continue
            
        # Initial thickness estimate: quarter-wave at center wavelength
        wl_center = 0.5 * (wl_min + wl_max)
        initial_ths = []
        for mat in pattern:
            if mat == 'H':
                initial_ths.append(0.5 * wl_center / (4.0 * n_H))
            else:
                initial_ths.append(0.5 * wl_center / (4.0 * n_L))
        
        # Local optimization using Nelder-Mead
        def objective(thickness_array):
            thicknesses = [max(0, min(max_layer_nm, t)) for t in thickness_array]
            if sum(thicknesses) > max_total_nm:
                return float('inf')
            return _fast_reflectance(thicknesses, pattern, n_ambient, n_substrate, n_L, n_H, wls)
        
        try:
            from scipy.optimize import minimize
            result = minimize(objective, initial_ths, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_thicknesses = [max(0, min(max_layer_nm, t)) for t in result.x]
                best_materials = pattern[:]
        except Exception:
            # If optimization fails, just use initial estimate
            score = objective(initial_ths)
            if score < best_score:
                best_score = score
                best_thicknesses = initial_ths[:]
                best_materials = pattern[:]
        
        # Check time
        if time.time() > deadline:
            break
    
    # If no valid result found, fallback to simple design
    if best_materials is None or best_score == float('inf'):
        wl_center = 0.5 * (wl_min + wl_max)
        return ["L"], [0.5 * wl_center / (4.0 * n_L)]
    
    # Ensure final validity
    if not is_valid(best_materials, best_thicknesses):
        wl_center = 0.5 * (wl_min + wl_max)
        return ["L"], [0.5 * wl_center / (4.0 * n_L)]
        
    return best_materials, best_thicknesses
