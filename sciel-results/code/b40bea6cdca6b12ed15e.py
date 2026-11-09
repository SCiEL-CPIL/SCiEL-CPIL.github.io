"""
Broadband antireflection coating design using symmetric tapered stacks.
Strategy:
1. Parameterize the stack as a symmetric sequence of layers.
2. Use a geometric progression for thicknesses (tapered stack) to reduce parameters.
3. Optimize the taper base thickness and ratio using Nelder-Mead.
4. Also try full symmetric optimization for the best tapered candidates.
"""
import time
import numpy as np

def _compute_reflectance(wavelengths, materials_list, thicknesses, n_ambient, n_substrate, n_materials):
    """
    Compute reflectance spectrum using the characteristic matrix method (vectorized).
    Returns reflectance (N,) as fraction (0-1).
    """
    N = len(wavelengths)
    n_layers = len(materials_list)
    
    if n_layers == 0:
        # Bare substrate
        n_s = n_substrate
        Y = n_s
        R = np.abs((n_ambient - Y) / (n_ambient + Y)) ** 2
        return R

    # k = 2*pi/lambda
    k = 2.0 * np.pi / wavelengths  # (N,)
    
    # Initialize identity matrix components
    M00 = np.ones(N, dtype=np.complex128)
    M01 = np.zeros(N, dtype=np.complex128)
    M10 = np.zeros(N, dtype=np.complex128)
    M11 = np.ones(N, dtype=np.complex128)
    
    for j in range(n_layers):
        n_j = n_materials[materials_list[j]]
        d_j = thicknesses[j]
        delta_j = k * n_j * d_j  # (N,)
        
        cos_d = np.cos(delta_j)
        sin_d = np.sin(delta_j)
        
        # Layer matrix
        L00 = cos_d
        L01 = 1j * sin_d / n_j
        L10 = 1j * n_j * sin_d
        L11 = cos_d
        
        # M_new = M_old * L
        new_M00 = M00 * L00 + M01 * L10
        new_M01 = M00 * L01 + M01 * L11
        new_M10 = M10 * L00 + M11 * L10
        new_M11 = M10 * L01 + M11 * L11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    # Admittance at front
    Y = M01 / M00
    R = np.abs((n_ambient - Y) / (n_ambient + Y)) ** 2
    
    return R

def _mean_reflectance_pct(materials_list, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_materials = problem["materials"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    R = _compute_reflectance(wavelengths, materials_list, thicknesses, n_ambient, n_substrate, n_materials)
    
    return float(np.mean(R) * 100.0)

def _is_valid(materials_list, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials_list) != len(thicknesses):
        return False
    if len(materials_list) > max_layers:
        return False
    if len(materials_list) == 0:
        return False
    
    for t in thicknesses:
        if t < -1e-6 or t > max_layer_nm + 1e-6:
            return False
    
    if sum(thicknesses) > max_total_nm + 1e-6:
        return False
    
    for m in materials_list:
        if m not in ('L', 'H'):
            return False
    
    return True

def _nelder_mead(objective, x0, bounds, max_iter=500, tol=1e-8, time_limit=10.0):
    """
    Simple Nelder-Mead optimizer with bounds and time limit.
    x0: initial guess
    bounds: tuple of (lower_bounds, upper_bounds) arrays
    """
    start_time = time.time()
    n = len(x0)
    x = np.array(x0, dtype=float)
    
    # Create simplex
    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        # Step size relative to bound or value
        lb, ub = bounds
        step = 0.1 * (ub[i] - lb[i]) if (ub[i] - lb[i]) > 0 else 0.1
        # If x is at boundary, try moving inward
        if xi[i] >= ub[i] - step:
            xi[i] -= step
        elif xi[i] <= lb[i] + step:
            xi[i] += step
        else:
            xi[i] += step
        # Clamp
        xi[i] = np.clip(xi[i], lb[i], ub[i])
        simplex.append(xi)
    
    # Evaluate
    fvals = [objective(s) for s in simplex]
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        # Clamp to bounds
        xr = np.clip(xr, bounds[0], bounds[1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + 2.0 * (centroid - simplex[-1])
            xe = np.clip(xe, bounds[0], bounds[1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            xc = np.clip(xc, bounds[0], bounds[1])
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
                    fvals[i] = objective(simplex[i])
    
    return simplex[0], fvals[0]

def _optimize_tapered_stack(problem, n_layers, start_d0, start_r, time_limit=5.0):
    """
    Optimize a symmetric tapered stack.
    Stack structure: L H L ... (alternating)
    Thicknesses: symmetric, geometric progression.
    """
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Generate material pattern (alternating L/H, starting with L)
    materials = []
    for i in range(n_layers):
        if i % 2 == 0:
            materials.append('L')
        else:
            materials.append('H')
    
    # Symmetric indices: 0, 1, ..., mid, ..., 1, 0
    half = n_layers // 2
    # Thicknesses are determined by d0 and r
    # d_k = d0 * r^k for k=0..half
    # The stack thicknesses are: d0, d0*r, ..., d0*r^half, ..., d0*r, d0
    
    def objective(params):
        d0, r = params
        if d0 <= 0 or r <= 0:
            return 1000.0
        
        # Construct thicknesses
        thicks = []
        for i in range(half + 1):
            d = d0 * (r ** i)
            thicks.append(d)
        
        # Symmetric expansion
        full_thicks = thicks + thicks[-2::-1]
        
        # Check constraints
        if any(t > max_layer_nm for t in full_thicks):
            return 1000.0
        if sum(full_thicks) > max_total_nm:
            return 1000.0
        
        return _mean_reflectance_pct(materials, full_thicks, problem)
    
    # Bounds for d0 and r
    # d0: reasonable range, e.g., 20nm to 200nm
    # r: reasonable range, e.g., 0.5 to 1.5
    lb = np.array([10.0, 0.5])
    ub = np.array([200.0, 1.5])
    
    x0 = np.array([start_d0, start_r])
    x0 = np.clip(x0, lb, ub)
    
    x_opt, f_opt = _nelder_mead(objective, x0, (lb, ub), max_iter=200, time_limit=time_limit)
    
    if f_opt >= 1000.0:
        return None, float('inf')
        
    d0, r = x_opt
    half = n_layers // 2
    thicks = []
    for i in range(half + 1):
        thicks.append(d0 * (r ** i))
    full_thicks = thicks + thicks[-2::-1]
    
    return materials, full_thicks

def _optimize_symmetric_full(problem, materials, initial_thicks, time_limit=5.0):
    """
    Optimize thicknesses of a symmetric stack directly.
    """
    n_layers = len(materials)
    half = n_layers // 2
    # Only optimize the first half + center
    n_params = half + 1
    
    lb = np.zeros(n_params)
    ub = np.full(n_params, problem["max_layer_nm"])
    
    # Initial guess
    x0 = np.array(initial_thicks[:n_params])
    x0 = np.clip(x0, lb, ub)
    
    def objective(params):
        # Symmetric expansion
        thicks = list(params) + list(params[-2::-1])
        
        if sum(thicks) > problem["max_total_nm"]:
            return 1000.0
            
        return _mean_reflectance_pct(materials, thicks, problem)
    
    x_opt, f_opt = _nelder_mead(objective, x0, (lb, ub), max_iter=200, time_limit=time_limit)
    
    if f_opt >= 1000.0:
        return None, float('inf')
        
    thicks = list(x_opt) + list(x_opt[-2::-1])
    return materials, thicks

def design(problem):
    """Design a broadband antireflection coating."""
    deadline = time.time() + 140.0 # Safety margin
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    
    # Quarter wave thicknesses
    d_L = centre / (4.0 * n_L)
    d_H = centre / (4.0 * n_H)
    
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Try different layer counts
    # Symmetric stacks must have odd number of layers
    layer_counts = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    
    # For each layer count, try optimizing a tapered stack
    for n_layers in layer_counts:
        if time.time() > deadline - 10.0:
            break
            
        # Initial guess for taper
        # Base thickness roughly quarter wave
        start_d0 = d_L
        start_r = 1.0
        
        # Try a few different starting points for r
        start_rs = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        for r0 in start_rs:
            if time.time() > deadline - 10.0:
                break
                
            mats, thicks = _optimize_tapered_stack(problem, n_layers, start_d0, r0, time_limit=3.0)
            
            if mats is not None and _is_valid(mats, thicks, problem):
                score = _mean_reflectance_pct(mats, thicks, problem)
                if score < best_score:
                    best_score = score
                    best_materials = mats
                    best_thicknesses = thicks
    
    # If we have time, refine the best symmetric stack with full parameter optimization
    if best_materials is not None and time.time() < deadline - 5.0:
        # Use the best tapered stack as initial guess for full symmetric optimization
        n_layers = len(best_materials)
        # Ensure it's symmetric
        half = n_layers // 2
        init_half = best_thicknesses[:half+1]
        
        mats, thicks = _optimize_symmetric_full(problem, best_materials, init_half, time_limit=5.0)
        
        if mats is not None and _is_valid(mats, thicks, problem):
            score = _mean_reflectance_pct(mats, thicks, problem)
            if score < best_score:
                best_score = score
                best_materials = mats
                best_thicknesses = thicks

    # Fallback if no valid design found or to ensure we return something
    if best_materials is None:
        best_materials = ["L"]
        best_thicknesses = [d_L]
        
    # Final validation
    if not _is_valid(best_materials, best_thicknesses, problem):
        # Force valid fallback
        best_materials = ["L"]
        best_thicknesses = [d_L]
        
    return best_materials, best_thicknesses
