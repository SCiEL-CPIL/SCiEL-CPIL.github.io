"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _characteristic_matrix(n, d, wl):
    """Compute the characteristic matrix for a single layer.
    n: refractive index (scalar)
    d: thickness (scalar)
    wl: wavelength (array)
    Returns 4 arrays: m00, m01, m10, m11 each of shape (len(wl),)
    """
    if d < 1e-10:
        # Zero-thickness layer: identity matrix
        m00 = np.ones_like(wl, dtype=complex)
        m01 = np.zeros_like(wl, dtype=complex)
        m10 = np.zeros_like(wl, dtype=complex)
        m11 = np.ones_like(wl, dtype=complex)
        return m00, m01, m10, m11
    
    k = 2.0 * np.pi * n * d / wl
    c = np.cos(k)
    s = np.sin(k)
    
    # Handle potential division by zero (shouldn't happen for d > 0, but be safe)
    s_safe = np.where(np.abs(s) < 1e-15, 1e-15, s)
    
    m00 = c.astype(complex)
    m01 = 1j / s_safe
    m10 = 1j * n * s.astype(complex)
    m11 = c.astype(complex)
    
    return m00, m01, m10, m11

def _reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    wl = np.linspace(band[0], band[1], 401)
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mat_dict = problem["materials"]
    
    # Start with identity matrix components
    m11_t = np.ones_like(wl, dtype=complex)
    m12_t = np.zeros_like(wl, dtype=complex)
    m21_t = np.zeros_like(wl, dtype=complex)
    m22_t = np.ones_like(wl, dtype=complex)
    
    for mat, d in zip(materials, thicknesses):
        n = mat_dict[mat]
        m00, m01, m10, m11 = _characteristic_matrix(n, d, wl)
        
        # M_total = M_total @ M_layer
        new_m11 = m11_t * m00 + m12_t * m10
        new_m12 = m11_t * m01 + m12_t * m11
        new_m21 = m21_t * m00 + m22_t * m10
        new_m22 = m21_t * m01 + m22_t * m11
        
        m11_t = new_m11
        m12_t = new_m12
        m21_t = new_m21
        m22_t = new_m22
    
    # Substrate admittance
    Ys = ns
    Y0 = n0
    
    # r = (Y0*(m11 + m12*Ys) - (m21 + m22*Ys)) / (Y0*(m11 + m12*Ys) + (m21 + m22*Ys))
    num = Y0 * (m11_t + m12_t * Ys) - (m21_t + m22_t * Ys)
    den = Y0 * (m11_t + m12_t * Ys) + (m21_t + m22_t * Ys)
    
    # Handle potential division by zero
    den_safe = np.where(np.abs(den) < 1e-15, 1e-15 + 0j, den)
    r = num / den_safe
    R = np.abs(r) ** 2
    
    return np.mean(R) * 100.0

def _nelder_mead(f, x0, bounds, max_iter=500, tol=1e-8):
    """Nelder-Mead optimization with bounds."""
    n = len(x0)
    # Initialize simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        step = 0.05 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.05
        simplex[i + 1] = x0
        simplex[i + 1, i] += step
    
    # Compute initial function values
    fvals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5   # contraction
    sigma = 0.5 # shrink
    
    for _ in range(max_iter):
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        # Check convergence
        if np.max(fvals) - np.min(fvals) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        # Clamp to bounds
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, [b[0] for b in bounds], [b[1] for b in bounds])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            # Accept reflection
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (xr - centroid)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = f(xc)
            
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = f(simplex[i])
    
    # Return best point
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s safety margin
    
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    mat_dict = problem["materials"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    nL = mat_dict["L"]
    nH = mat_dict["H"]
    
    # Initial best: quarter-wave L layer
    best_materials = ["L"]
    best_thicknesses = [centre / (4.0 * nL)]
    try:
        best_score = _reflectance(best_materials, best_thicknesses, problem)
    except Exception:
        best_score = 100.0  # fallback if computation fails
    
    # Try different numbers of layers
    # We'll search over 2-10 layers with alternating or optimized materials
    # For each layer count, we'll use Nelder-Mead on thicknesses
    
    # Precompute wavelength grid
    wl = np.linspace(band[0], band[1], 401)
    
    def fast_reflectance(thicknesses, materials_list):
        """Fast reflectance computation for given thicknesses and materials."""
        n0 = problem["ambient_index"]
        ns = problem["substrate_index"]
        
        # Start with identity matrix components
        m11_t = np.ones_like(wl, dtype=complex)
        m12_t = np.zeros_like(wl, dtype=complex)
        m21_t = np.zeros_like(wl, dtype=complex)
        m22_t = np.ones_like(wl, dtype=complex)
        
        for mat, d in zip(materials_list, thicknesses):
            n = mat_dict[mat]
            if d < 1e-10:
                # Zero-thickness layer: identity, skip
                continue
            k = 2.0 * np.pi * n * d / wl
            c = np.cos(k)
            s = np.sin(k)
            
            # Handle potential division by zero
            s_safe = np.where(np.abs(s) < 1e-15, 1e-15, s)
            
            m00 = c.astype(complex)
            m01 = 1j / s_safe
            m10 = 1j * n * s.astype(complex)
            m11 = c.astype(complex)
            
            # M_total = M_total @ M_layer
            new_m11 = m11_t * m00 + m12_t * m10
            new_m12 = m11_t * m01 + m12_t * m11
            new_m21 = m21_t * m00 + m22_t * m10
            new_m22 = m21_t * m01 + m22_t * m11
            
            m11_t = new_m11
            m12_t = new_m12
            m21_t = new_m21
            m22_t = new_m22
        
        Ys = ns
        Y0 = n0
        
        num = Y0 * (m11_t + m12_t * Ys) - (m21_t + m22_t * Ys)
        den = Y0 * (m11_t + m12_t * Ys) + (m21_t + m22_t * Ys)
        
        # Handle potential division by zero
        den_safe = np.where(np.abs(den) < 1e-15, 1e-15 + 0j, den)
        r = num / den_safe
        R = np.abs(r) ** 2
        
        return np.mean(R) * 100.0
    
    # Search strategy: try different layer counts and material sequences
    # For each, optimize thicknesses with Nelder-Mead
    
    # Material sequences to try
    # Start with simple alternating patterns
    material_patterns = []
    
    # Quarter-wave stack patterns
    for nl in range(1, min(max_layers, 8) + 1):
        # Alternating starting with L
        pattern = []
        for i in range(nl):
            if i % 2 == 0:
                pattern.append("L")
            else:
                pattern.append("H")
        material_patterns.append(pattern)
        
        # Alternating starting with H
        pattern = []
        for i in range(nl):
            if i % 2 == 0:
                pattern.append("H")
            else:
                pattern.append("L")
        material_patterns.append(pattern)
    
    # Also try some non-alternating patterns
    material_patterns.append(["L", "H", "L", "H", "L"])
    material_patterns.append(["H", "L", "H", "L", "H"])
    material_patterns.append(["L", "H", "H", "L"])
    material_patterns.append(["H", "L", "L", "H"])
    
    # Remove duplicates
    seen = set()
    unique_patterns = []
    for p in material_patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    
    # For each pattern, optimize thicknesses
    for pattern in unique_patterns:
        if time.time() > deadline:
            break
        
        nl = len(pattern)
        if nl > max_layers:
            continue
        
        # Initial guess: quarter-wave thicknesses
        x0 = np.array([centre / (4.0 * mat_dict[mat]) for mat in pattern])
        
        # Bounds
        bounds = [(0.0, max_layer_nm) for _ in range(nl)]
        
        # Additional constraint: total thickness <= max_total_nm
        # We'll handle this by checking in the objective function
        
        def objective(x):
            if time.time() > deadline:
                return 100.0
            # Check total thickness constraint
            if np.sum(x) > max_total_nm:
                return 100.0 + (np.sum(x) - max_total_nm) * 0.1
            if np.any(x < 0) or np.any(x > max_layer_nm):
                return 100.0
            return fast_reflectance(list(x), pattern)
        
        # Run Nelder-Mead
        try:
            x_opt, f_opt = _nelder_mead(objective, x0, bounds, max_iter=300, tol=1e-6)
            
            if f_opt < best_score:
                # Check constraints
                if np.sum(x_opt) <= max_total_nm and np.all(x_opt >= 0) and np.all(x_opt <= max_layer_nm):
                    best_materials = list(pattern)
                    best_thicknesses = list(x_opt)
                    best_score = f_opt
        except Exception:
            pass
    
    # Try a few more random restarts if time permits
    if time.time() < deadline - 30.0:
        np.random.seed(42)
        for _ in range(5):
            if time.time() > deadline:
                break
            # Random material pattern
            nl = np.random.randint(3, min(max_layers, 10) + 1)
            pattern = []
            for i in range(nl):
                if np.random.rand() > 0.5:
                    pattern.append("L")
                else:
                    pattern.append("H")
            
            # Random initial thicknesses
            x0 = np.random.uniform(0.1 * max_layer_nm, 0.9 * max_layer_nm, nl)
            
            bounds = [(0.0, max_layer_nm) for _ in range(nl)]
            
            def objective(x):
                if time.time() > deadline:
                    return 100.0
                if np.sum(x) > max_total_nm:
                    return 100.0 + (np.sum(x) - max_total_nm) * 0.1
                if np.any(x < 0) or np.any(x > max_layer_nm):
                    return 100.0
                return fast_reflectance(list(x), pattern)
            
            try:
                x_opt, f_opt = _nelder_mead(objective, x0, bounds, max_iter=200, tol=1e-6)
                
                if f_opt < best_score:
                    if np.sum(x_opt) <= max_total_nm and np.all(x_opt >= 0) and np.all(x_opt <= max_layer_nm):
                        best_materials = list(pattern)
                        best_thicknesses = list(x_opt)
                        best_score = f_opt
            except Exception:
                pass
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
