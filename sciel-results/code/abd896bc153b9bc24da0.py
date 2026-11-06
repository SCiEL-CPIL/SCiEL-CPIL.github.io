"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _compute_reflectance(wavelengths, materials, thicknesses, ambient_index, substrate_index):
    """Compute normal-incidence reflectance using characteristic matrix method."""
    n_layers = len(materials)
    n_wl = len(wavelengths)
    reflectance = np.zeros(n_wl)
    
    for i, wl in enumerate(wavelengths):
        # Build characteristic matrix
        M = np.array([[1.0, 0.0]], dtype=complex)
        
        for j in range(n_layers):
            n_j = materials[j]
            d_j = thicknesses[j]
            k_j = 2.0 * np.pi * n_j / wl  # phase factor
            cos_kj = np.cos(k_j)
            sin_kj = np.sin(k_j)
            
            # Layer matrix for normal incidence
            if n_j > 0:
                # Diagonal matrix for absorbing or non-absorbing
                B = np.array([[cos_kj, 1j * sin_kj / n_j],
                              [1j * n_j * sin_kj, cos_kj]], dtype=complex)
            else:
                B = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
            
            M = M @ B
        
        # Input and substrate indices
        n0 = ambient_index
        ns = substrate_index
        
        # Reflectance
        r = (M[0,0] + M[0,1]*ns) / (M[0,0] + M[0,1]*ns)  # This is wrong, let me fix
        
        # Actually, the reflection coefficient is:
        # r = (B11 + B12*ns - B21 - B22*ns) / (B11 + B12*ns + B21 + B22*ns)
        # Wait, let me use the standard formula
        
        # The characteristic matrix M = product of all layer matrices
        # Then: r = (M11 + M12*ns - M21 - M22*ns) / (M11 + M12*ns + M21 + M22*ns)
        # But actually for normal incidence with real indices:
        # The admittance approach: Y = M21/M11 (or similar)
        
        # Let me use the standard thin-film formula:
        # For a stack, the reflection coefficient r is given by:
        # r = (M11 + M12*ns - M21 - M22*ns) / (M11 + M12*ns + M21 + M22*ns)
        # where M is the total characteristic matrix
        
        # Actually, the correct formula uses:
        # Y = (M21 + M22*ns) / (M11 + M12*ns)  # admittance
        # r = (n0 - Y) / (n0 + Y)
        
        M11, M12 = M[0,0], M[0,1]
        M21, M22 = M[1,0], M[1,1]
        
        Y = (M21 + M22 * ns) / (M11 + M12 * ns)
        r = (n0 - Y) / (n0 + Y)
        reflectance[i] = np.abs(r) ** 2
    
    return reflectance


def _mean_reflectance(wavelengths, materials, thicknesses, ambient_index, substrate_index):
    """Compute mean reflectance in percent."""
    R = _compute_reflectance(wavelengths, materials, thicknesses, ambient_index, substrate_index)
    return 100.0 * np.mean(R)


def _nelder_mead(objective, x0, bounds, max_iter=200, tol=1e-6, deadline=None):
    """Custom Nelder-Mead simplex optimizer with deadline guard."""
    n = len(x0)
    
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        x = np.array(x0, dtype=float)
        if x[i] != 0:
            x[i] *= 1.05
        else:
            x[i] = 0.05
        simplex.append(x)
    
    # Evaluate initial simplex
    values = [objective(x) for x in simplex]
    
    # Sort simplex by objective value
    order = np.argsort(values)
    simplex = [simplex[i] for i in order]
    values = [values[i] for i in order]
    
    alpha, gamma, rho = 1.0, 2.0, 0.5  # Reflection, expansion, contraction
    start_time = time.time()
    
    for iteration in range(max_iter):
        # Check deadline
        if deadline is not None and time.time() > deadline:
            break
        
        # Centroid of all points except worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        # Clip to bounds
        for i in range(n):
            xr[i] = np.clip(xr[i], bounds[i][0], bounds[i][1])
        fr = objective(xr)
        
        if fr < values[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            for i in range(n):
                xe[i] = np.clip(xe[i], bounds[i][0], bounds[i][1])
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
            if fr < values[-1]:
                # Outside contraction
                xc = centroid + rho * (simplex[-1] - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (simplex[-1] - centroid)
            for i in range(n):
                xc[i] = np.clip(xc[i], bounds[i][0], bounds[i][1])
            fc = objective(xc)
            
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    values[i] = objective(simplex[i])
        
        # Sort simplex
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[-1] - values[0]) < tol:
            break
    
    return simplex[0], values[0]


def _optimize_stack(problem):
    """Optimize the coating stack."""
    band = problem["band_nm"]
    wl_min, wl_max = band[0], band[1]
    n_wl = 401
    wavelengths = np.linspace(wl_min, wl_max, n_wl)
    
    materials_dict = problem["materials"]
    nL = materials_dict["L"]
    nH = materials_dict["H"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    # Start with a reasonable initial guess: alternating L/H layers
    # Use quarter-wave thicknesses centered at 600nm
    center_wl = 0.5 * (wl_min + wl_max)
    
    # Try different numbers of layers
    best_result = None
    best_score = float('inf')
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Try various configurations
    for n_layers in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if n_layers > max_layers:
            break
        
        if time.time() > deadline:
            break
        
        # Initialize with alternating L/H
        mat_labels = []
        for i in range(n_layers):
            if i % 2 == 0:
                mat_labels.append('L')
            else:
                mat_labels.append('H')
        
        # Initial thicknesses: quarter-wave at center wavelength
        init_thicknesses = []
        for i, m in enumerate(mat_labels):
            n_mat = nL if m == 'L' else nH
            d = center_wl / (4.0 * n_mat)
            init_thicknesses.append(d)
        
        # Define objective function
        def objective(x):
            # x contains thicknesses
            thicknesses = list(x)
            
            # Check constraints
            if any(t < 0 or t > max_layer for t in thicknesses):
                return 100.0  # Penalize
            if sum(thicknesses) > max_total:
                return 100.0  # Penalize
            
            mats = [nL if m == 'L' else nH for m in mat_labels]
            return _mean_reflectance(wavelengths, mats, thicknesses, n0, ns)
        
        # Bounds for thicknesses
        bounds = [(0.0, max_layer) for _ in range(n_layers)]
        
        # Run Nelder-Mead
        try:
            x_opt, f_opt = _nelder_mead(objective, init_thicknesses, bounds, 
                                          max_iter=100, tol=1e-5, deadline=deadline)
        except Exception:
            continue
        
        if f_opt < best_score:
            best_score = f_opt
            best_result = (mat_labels, list(x_opt))
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: single quarter-wave layer
        d = center_wl / (4.0 * nL)
        return ["L"], [d]


def design(problem):
    """Design an antireflection coating stack."""
    start_time = time.time()
    
    try:
        result = _optimize_stack(problem)
    except Exception as e:
        # Fallback to simple quarter-wave layer
        center = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        d = center / (4.0 * problem["materials"]["L"])
        result = ["L"], [d]
    
    # Validate result
    materials, thicknesses = result
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    # Ensure constraints are met
    if len(materials) > max_layers:
        materials = materials[:max_layers]
        thicknesses = thicknesses[:max_layers]
    
    # Clip thicknesses
    thicknesses = [max(0.0, min(t, max_layer)) for t in thicknesses]
    
    # If total exceeds limit, scale down
    total = sum(thicknesses)
    if total > max_total:
        scale = max_total / total
        thicknesses = [t * scale for t in thicknesses]
    
    return materials, thicknesses
# EVOLVE-BLOCK-END
