"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _mean_reflectance(n_vals, d_vals, wavelengths, n_amb, n_sub):
    """Compute mean reflectance using vectorized characteristic matrix method."""
    n_layers = len(n_vals)
    n_waves = len(wavelengths)
    
    # Precompute phase factors for all layers and wavelengths
    # delta[j, i] = 2*pi*n_j*d_j/lambda_i
    lam = wavelengths.reshape(1, -1)  # (1, n_waves)
    nd = n_vals * d_vals  # (n_layers,)
    delta = 2.0 * np.pi * nd[:, None] / lam  # (n_layers, n_waves)
    
    # Compute cos and sin for all layers and wavelengths
    cd = np.cos(delta)  # (n_layers, n_waves)
    sd = np.sin(delta)  # (n_layers, n_waves)
    
    # Initialize the total matrix as identity for all wavelengths
    M11 = np.ones(n_waves, dtype=complex)
    M12 = np.zeros(n_waves, dtype=complex)
    M21 = np.zeros(n_waves, dtype=complex)
    M22 = np.ones(n_waves, dtype=complex)
    
    for j in range(n_layers):
        # Layer matrix elements for all wavelengths
        n_j = n_vals[j]
        # Handle small sin values
        safe_sd = np.where(np.abs(sd[j]) < 1e-15, 1e-15, sd[j])
        inv_n_sd = 1.0 / (n_j * safe_sd)
        B = n_j * sd[j]
        
        # Multiply current total matrix by this layer's matrix
        # New M = M @ layer_M
        # layer_M = [[cd[j], 1j*inv_n_sd[j]], [1j*B[j], cd[j]]]
        new_M11 = M11 * cd[j] + M12 * 1j * B[j]
        new_M12 = M11 * 1j * inv_n_sd[j] + M12 * cd[j]
        new_M21 = M21 * cd[j] + M22 * 1j * B[j]
        new_M22 = M21 * 1j * inv_n_sd[j] + M22 * cd[j]
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Input admittance
    Y_in = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
    
    # Reflection coefficient
    r = (n_amb - Y_in) / (n_amb + Y_in)
    R = np.abs(r) ** 2 * 100.0
    
    return np.mean(R)

def _nelder_mead(objective, x0, maxiter=500, tol=1e-10):
    """Nelder-Mead implementation."""
    n = len(x0)
    
    # Simplex initialization
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        point = np.array(x0, dtype=float)
        step = 0.1 * abs(point[i]) + 0.01
        point[i] += step
        simplex.append(point)
    
    # Evaluate initial simplex
    values = [objective(p) for p in simplex]
    
    for it in range(maxiter):
        # Sort simplex
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[0] - values[-1]) < tol:
            break
        
        # Centroid (excluding worst point)
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < values[0]:
            # Expansion
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            # Accept reflection
            simplex[-1] = xr
            values[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    values[i] = objective(simplex[i])
    
    # Return best point
    best_idx = np.argmin(values)
    return np.array(simplex[best_idx]), values[best_idx]

def _make_evaluator(materials, problem, wavelengths):
    """Create a fast objective function for a fixed material pattern."""
    n_vals = np.array([problem["materials"][m] for m in materials])
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    
    def objective(d_vals):
        d = np.asarray(d_vals, dtype=float)
        if np.any(d < 0) or np.any(d > problem["max_layer_nm"]):
            return 1e6
        if np.sum(d) > problem["max_total_nm"]:
            return 1e6
        return _mean_reflectance(n_vals, d, wavelengths, n_amb, n_sub)
    
    return objective

def _nm_fallback(problem):
    """Return a valid fallback solution."""
    band = problem["band_nm"]
    n_L = problem["materials"]["L"]
    return ["L"], [band[0] / (4.0 * n_L)]

def design(problem):
    """Design an antireflection coating using multi-start optimization."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    
    # Precompute wavelengths for evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    
    best_score = 1e6
    best_materials, best_thicknesses = _nm_fallback(problem)
    
    # Define candidate patterns for various layer counts
    patterns_to_try = []
    
    # Single layer
    patterns_to_try.append((["L"], 1))
    patterns_to_try.append((["H"], 1))
    
    # Two layers
    patterns_to_try.append((["L", "H"], 2))
    patterns_to_try.append((["H", "L"], 2))
    
    # Three layers
    patterns_to_try.append((["L", "H", "L"], 3))
    patterns_to_try.append((["H", "L", "H"], 3))
    
    # Four layers
    patterns_to_try.append((["L", "H", "H", "L"], 4))
    patterns_to_try.append((["H", "L", "L", "H"], 4))
    patterns_to_try.append((["L", "H", "L", "H"], 4))
    patterns_to_try.append((["H", "L", "H", "L"], 4))
    
    # Five layers
    patterns_to_try.append((["L", "H", "H", "H", "L"], 5))
    patterns_to_try.append((["H", "L", "L", "L", "H"], 5))
    patterns_to_try.append((["L", "H", "L", "H", "L"], 5))
    patterns_to_try.append((["H", "L", "H", "L", "H"], 5))
    
    # Six layers
    patterns_to_try.append((["L", "H", "H", "H", "H", "L"], 6))
    patterns_to_try.append((["H", "L", "L", "L", "L", "H"], 6))
    patterns_to_try.append((["L", "H", "L", "H", "L", "H"], 6))
    patterns_to_try.append((["H", "L", "H", "L", "H", "L"], 6))
    
    # Seven layers
    patterns_to_try.append((["L", "H", "H", "H", "H", "H", "L"], 7))
    patterns_to_try.append((["H", "L", "L", "L", "L", "L", "H"], 7))
    patterns_to_try.append((["L", "H", "L", "H", "L", "H", "L"], 7))
    patterns_to_try.append((["H", "L", "H", "L", "H", "L", "H"], 7))
    
    # Eight layers
    patterns_to_try.append((["L", "H", "H", "H", "H", "H", "H", "L"], 8))
    patterns_to_try.append((["H", "L", "L", "L", "L", "L", "L", "H"], 8))
    patterns_to_try.append((["L", "H", "L", "H", "L", "H", "L", "H"], 8))
    patterns_to_try.append((["H", "L", "H", "L", "H", "L", "H", "L"], 8))
    
    # Nine layers
    patterns_to_try.append((["L", "H", "H", "H", "H", "H", "H", "H", "L"], 9))
    patterns_to_try.append((["H", "L", "L", "L", "L", "L", "L", "L", "H"], 9))
    patterns_to_try.append((["L", "H", "L", "H", "L", "H", "L", "H", "L"], 9))
    patterns_to_try.append((["H", "L", "H", "L", "H", "L", "H", "L", "H"], 9))
    
    # Ten layers
    patterns_to_try.append((["L", "H", "H", "H", "H", "H", "H", "H", "H", "L"], 10))
    patterns_to_try.append((["H", "L", "L", "L", "L", "L", "L", "L", "L", "H"], 10))
    patterns_to_try.append((["L", "H", "L", "H", "L", "H", "L", "H", "L", "H"], 10))
    patterns_to_try.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L"], 10))
    
    # Twelve layers
    for n in [12, 14, 16, 18, 20]:
        if n <= max_layers:
            # Alternating starting with L
            pat = ["L" if i % 2 == 0 else "H" for i in range(n)]
            patterns_to_try.append((pat, n))
            # Alternating starting with H
            pat = ["H" if i % 2 == 0 else "L" for i in range(n)]
            patterns_to_try.append((pat, n))
            # Symmetric with more H in the middle
            if n >= 4:
                half = n // 2
                pat = []
                for i in range(n):
                    if i < half - 1:
                        pat.append("L" if i % 2 == 0 else "H")
                    elif i >= half - 1 and i <= half + 1:
                        pat.append("H")
                    else:
                        pat.append("L" if (n - 1 - i) % 2 == 0 else "H")
                patterns_to_try.append((pat, n))
    
    # Sort patterns by layer count (fewer layers first for quick wins)
    patterns_to_try.sort(key=lambda x: x[1])
    
    # Remove duplicates
    seen = set()
    unique_patterns = []
    for pat, n in patterns_to_try:
        key = tuple(pat)
        if key not in seen:
            seen.add(key)
            unique_patterns.append((pat, n))
    
    # For each pattern, try multiple starting points
    n_starts_by_size = {1: 3, 2: 4, 3: 4, 4: 5, 5: 5, 6: 4, 7: 4, 8: 3, 9: 3, 10: 3, 12: 2, 14: 2, 16: 1, 18: 1, 20: 1}
    
    for materials, n_layers in unique_patterns:
        if time.time() > deadline - 5:
            break
        
        evaluator = _make_evaluator(materials, problem, wavelengths)
        
        # Generate starting points
        start_points = []
        
        # Quarter-wave at center wavelength
        x0_qw = [centre / (4.0 * problem["materials"][m]) for m in materials]
        if sum(x0_qw) <= max_total_nm:
            start_points.append(np.array(x0_qw))
        
        # Scaled quarter-wave
        scale_factors = [0.5, 0.75, 1.25, 1.5, 2.0]
        for sf in scale_factors:
            x0 = [d * sf for d in x0_qw]
            if sum(x0) <= max_total_nm and all(0 <= d <= max_layer_nm for d in x0):
                start_points.append(np.array(x0))
        
        # Random starts (limited)
        n_random = min(2, n_starts_by_size.get(n_layers, 3))
        for _ in range(n_random):
            x0 = np.random.uniform(0, max_total_nm / n_layers, n_layers)
            if np.sum(x0) <= max_total_nm and np.all(x0 <= max_layer_nm):
                start_points.append(x0)
        
        if not start_points:
            # Fallback: uniform distribution of total thickness
            x0 = np.full(n_layers, max_total_nm / n_layers)
            start_points.append(x0)
        
        # Limit number of starts
        max_starts = n_starts_by_size.get(n_layers, 3)
        start_points = start_points[:max_starts]
        
        # Try each starting point
        for x0 in start_points:
            if time.time() > deadline - 5:
                break
            
            try:
                result_x, result_f = _nelder_mead(
                    evaluator,
                    x0,
                    maxiter=200 if n_layers <= 6 else 150
                )
                
                # Verify constraints
                if np.all(result_x >= 0) and np.all(result_x <= max_layer_nm) and np.sum(result_x) <= max_total_nm:
                    if result_f < best_score:
                        best_score = result_f
                        best_materials = materials
                        best_thicknesses = list(result_x)
            except Exception:
                pass
        
        # If we found a good solution, do a final polish with more iterations
        if best_score < 5.0 and time.time() < deadline - 10:
            try:
                evaluator_best = _make_evaluator(best_materials, problem, wavelengths)
                result_x, result_f = _nelder_mead(
                    evaluator_best,
                    np.array(best_thicknesses),
                    maxiter=300
                )
                if np.all(result_x >= 0) and np.all(result_x <= max_layer_nm) and np.sum(result_x) <= max_total_nm:
                    if result_f < best_score:
                        best_score = result_f
                        best_thicknesses = list(result_x)
            except Exception:
                pass
    
    # Ensure final result is valid
    if len(best_materials) != len(best_thicknesses):
        best_materials, best_thicknesses = _nm_fallback(problem)
    
    if not all(0 <= t <= max_layer_nm for t in best_thicknesses):
        best_thicknesses = [min(max(t, 0), max_layer_nm) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
