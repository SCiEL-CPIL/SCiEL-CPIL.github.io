"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution, minimize
import time

# EVOLVE-BLOCK-START

def _compute_reflectance_vec(wavelengths, materials, thicknesses, ambient_index, substrate_index):
    """Vectorized reflectance computation using characteristic matrix method."""
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wavelengths, r**2)
    
    # Precompute layer indices
    layer_indices = []
    for label in materials:
        if label == 'L':
            layer_indices.append(1.38)
        else:
            layer_indices.append(2.35)
    
    # Vectorized: compute phase for all wavelengths at once
    # For each layer j: phase_j(w) = 2*pi*n_j*d_j/w
    # We can't easily vectorize the matrix multiplication across wavelengths with different phases,
    # but we can use a chunked approach or just loop over wavelengths with numpy ops.
    
    # Alternative: use the fact that for a given set of thicknesses, we can compute
    # the product of 2x2 matrices. Let's do it per-wavelength but vectorize the math.
    
    reflectances = np.zeros(len(wavelengths))
    
    for i, wl in enumerate(wavelengths):
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        for j in range(n_layers):
            n_j = layer_indices[j]
            d_j = thicknesses[j]
            phase = 2.0 * np.pi * n_j * d_j / wl
            c = np.cos(phase)
            s = np.sin(phase)
            # Characteristic matrix
            C = np.array([[c, 1j * s / n_j],
                          [1j * n_j * s, c]], dtype=complex)
            M = M @ C
        
        # Admittance
        B = M[0, 0] * substrate_index + M[0, 1] * substrate_index**2
        C = M[1, 0] * substrate_index + M[1, 1] * substrate_index**2
        Y = C / B
        r = (ambient_index - Y) / (ambient_index + Y)
        reflectances[i] = np.abs(r)**2
    
    return reflectances

def _mean_reflectance_percent(wavelengths, materials, thicknesses, ambient_index, substrate_index):
    """Compute mean reflectance in percent."""
    refl = _compute_reflectance_vec(wavelengths, materials, thicknesses, ambient_index, substrate_index)
    return 100.0 * float(np.mean(refl))

def design(problem):
    """Design an antireflection coating using multi-start differential evolution + local refinement."""
    deadline = time.time() + 50.0  # Safety margin
    
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    centre = 0.5 * (band[0] + band[1])
    
    # Start with a good single-layer guess
    best_materials = ["L"]
    best_thicknesses = [centre / (4.0 * 1.38)]
    best_score = _mean_reflectance_percent(wavelengths, best_materials, best_thicknesses, ambient_index, substrate_index)
    
    # Refine single L layer
    def obj_single_L(t):
        if t < 0 or t > max_layer_nm:
            return 1e6
        return _mean_reflectance_percent(wavelengths, ["L"], [t], ambient_index, substrate_index)
    
    res = minimize_scalar(obj_single_L, bounds=(0, max_layer_nm), method='bounded',
                          options={'maxiter': 200, 'xatol': 0.001})
    if res.fun < best_score:
        best_materials = ["L"]
        best_thicknesses = [float(res.x)]
        best_score = float(res.fun)
    
    # Single H layer
    def obj_single_H(t):
        if t < 0 or t > max_layer_nm:
            return 1e6
        return _mean_reflectance_percent(wavelengths, ["H"], [t], ambient_index, substrate_index)
    
    res = minimize_scalar(obj_single_H, bounds=(0, max_layer_nm), method='bounded',
                          options={'maxiter': 200, 'xatol': 0.001})
    if res.fun < best_score:
        best_materials = ["H"]
        best_thicknesses = [float(res.x)]
        best_score = float(res.fun)
    
    # Helper to evaluate a given stack pattern (list of 'L'/'H')
    def evaluate_stack(pattern, thicknesses):
        materials = list(pattern)
        thicknesses = [float(t) for t in thicknesses]
        # Check constraints
        for t in thicknesses:
            if t < 0 or t > max_layer_nm:
                return 1e6
        total = sum(thicknesses)
        if total > max_total_nm:
            return 1e6
        return _mean_reflectance_percent(wavelengths, materials, thicknesses, ambient_index, substrate_index)
    
    # Helper for differential evolution with a fixed pattern
    def run_de(pattern, bounds_list, maxiter=100, tol=1e-8):
        n = len(pattern)
        
        def objective(x):
            return evaluate_stack(pattern, x)
        
        # Bounds for DE
        bounds = [(0.0, max_layer_nm) for _ in range(n)]
        # Additional constraint: sum <= max_total_nm
        # We'll handle this in the objective by returning 1e6
        
        try:
            res = differential_evolution(objective, bounds, maxiter=maxiter, tol=tol,
                                         seed=42, polish=True, init='sobol')
            if res.fun < best_score:
                return pattern, [float(t) for t in res.x], float(res.fun)
        except Exception:
            pass
        return None
    
    # Try two-layer stacks: LH, HL
    if time.time() < deadline - 30:
        for pattern in [["L", "H"], ["H", "L"]]:
            # Quarter-wave initial guesses
            t_guess = [centre / (4.0 * 1.38), centre / (4.0 * 2.35)]
            
            # First try local optimization from quarter-wave
            def obj_lh(x):
                return evaluate_stack(pattern, x)
            
            res = minimize(obj_lh, t_guess, method='Nelder-Mead',
                          options={'xatol': 0.1, 'fatol': 1e-6, 'maxiter': 500, 'adaptive': True})
            if res.fun < best_score:
                # Verify constraints
                tvals = [float(t) for t in res.x]
                if all(0 <= t <= max_layer_nm for t in tvals) and sum(tvals) <= max_total_nm:
                    best_materials = list(pattern)
                    best_thicknesses = tvals
                    best_score = float(res.fun)
            
            # DE search
            result = run_de(pattern, None, maxiter=80)
            if result is not None:
                pat, thicks, score = result
                best_materials = list(pat)
                best_thicknesses = thicks
                best_score = score
    
    # Try three-layer stacks: LHL, HLH
    if time.time() < deadline - 20:
        for pattern in [["L", "H", "L"], ["H", "L", "H"]]:
            # Quarter-wave guesses
            t_guess = [centre/(4*1.38), centre/(4*2.35), centre/(4*1.38)]
            
            def obj_3(x):
                return evaluate_stack(pattern, x)
            
            res = minimize(obj_3, t_guess, method='Nelder-Mead',
                          options={'xatol': 0.1, 'fatol': 1e-6, 'maxiter': 800, 'adaptive': True})
            if res.fun < best_score:
                tvals = [float(t) for t in res.x]
                if all(0 <= t <= max_layer_nm for t in tvals) and sum(tvals) <= max_total_nm:
                    best_materials = list(pattern)
                    best_thicknesses = tvals
                    best_score = float(res.fun)
            
            result = run_de(pattern, None, maxiter=60)
            if result is not None:
                pat, thicks, score = result
                best_materials = list(pat)
                best_thicknesses = thicks
                best_score = score
    
    # Try four-layer stacks: LHHL, HLHL, etc.
    if time.time() < deadline - 10:
        patterns_4 = [["L", "H", "H", "L"], ["H", "L", "L", "H"], ["L", "H", "L", "H"]]
        for pattern in patterns_4:
            # Quarter-wave guesses
            nL = pattern.count('L')
            nH = pattern.count('H')
            t_guess = []
            for p in pattern:
                if p == 'L':
                    t_guess.append(centre / (4 * 1.38))
                else:
                    t_guess.append(centre / (4 * 2.35))
            
            def obj_4(x):
                return evaluate_stack(pattern, x)
            
            res = minimize(obj_4, t_guess, method='Nelder-Mead',
                          options={'xatol': 0.1, 'fatol': 1e-6, 'maxiter': 1000, 'adaptive': True})
            if res.fun < best_score:
                tvals = [float(t) for t in res.x]
                if all(0 <= t <= max_layer_nm for t in tvals) and sum(tvals) <= max_total_nm:
                    best_materials = list(pattern)
                    best_thicknesses = tvals
                    best_score = float(res.fun)
    
    # Final polish: try a few more patterns if time permits
    if time.time() < deadline - 5:
        more_patterns = [
            ["L", "H", "L", "H", "L"],
            ["H", "L", "H", "L", "H"],
            ["L", "H", "H", "L", "H", "L"],
            ["H", "L", "L", "H", "L", "H"],
        ]
        for pattern in more_patterns:
            n = len(pattern)
            t_guess = []
            for p in pattern:
                if p == 'L':
                    t_guess.append(centre / (4 * 1.38))
                else:
                    t_guess.append(centre / (4 * 2.35))
            
            def obj_more(x):
                return evaluate_stack(pattern, x)
            
            try:
                res = minimize(obj_more, t_guess, method='Nelder-Mead',
                              options={'xatol': 0.1, 'fatol': 1e-6, 'maxiter': 800, 'adaptive': True})
                if res.fun < best_score:
                    tvals = [float(t) for t in res.x]
                    if all(0 <= t <= max_layer_nm for t in tvals) and sum(tvals) <= max_total_nm:
                        best_materials = list(pattern)
                        best_thicknesses = tvals
                        best_score = float(res.fun)
            except Exception:
                pass
    
    # Final constraint enforcement
    total = sum(best_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    best_thicknesses = [max(0.0, min(float(t), float(max_layer_nm))) for t in best_thicknesses]
    
    return list(best_materials), best_thicknesses

# EVOLVE-BLOCK-END
