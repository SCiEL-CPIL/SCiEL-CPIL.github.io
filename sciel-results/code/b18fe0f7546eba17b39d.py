"""Broadband antireflection coating design using characteristic-matrix method."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _char_matrix(n, d, wl):
    """
    Characteristic matrix for a single lossless layer.
    n: refractive index (real)
    d: physical thickness (nm)
    wl: wavelength (nm)
    Returns 2x2 numpy array.
    """
    beta = 2.0 * np.pi * n * d / wl
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    # M = [[cos(beta), (1/(i*n))*sin(beta)], [i*n*sin(beta), cos(beta)]]
    # Since 1/i = -i, (1/(i*n))*sin(beta) = -i*sin(beta)/n
    # i*n*sin(beta) = i*n*sin(beta)
    # For real-valued computation, we can work with the matrix in a way that
    # allows us to compute the reflectance. The standard approach uses complex
    # matrices, but since everything is real (lossless), we can use the fact that
    # the reflectance depends on the ratio of the matrix elements.
    # Let's use the complex form directly.
    M = np.array([
        [cos_b, -1j * sin_b / n],
        [1j * n * sin_b, cos_b]
    ], dtype=complex)
    return M


def _compute_reflectance(materials, thicknesses, wl, n_ambient, n_substrate):
    """
    Compute reflectance (fraction, 0-1) at a single wavelength using the
    characteristic-matrix method.
    materials: list of 'L'/'H'
    thicknesses: list of physical thicknesses in nm
    wl: wavelength in nm
    n_ambient: ambient refractive index
    n_substrate: substrate refractive index
    """
    if len(materials) == 0:
        # Bare substrate
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return abs(r) ** 2
    
    # Start with the substrate matrix (identity for the substrate side)
    # Actually, the total matrix is the product of all layer matrices.
    # The reflectance is computed from the total matrix M_total = M1 * M2 * ... * M_N
    # where M1 is the first layer (air side) and MN is the last layer (substrate side).
    # The boundary conditions give:
    # E_ambient = M11 * E_substrate + M12 * H_substrate
    # H_ambient = M21 * E_substrate + M22 * H_substrate
    # For the substrate, H_substrate = (n_substrate / n_ambient) * E_substrate / (something)...
    # Let me use the standard formula:
    # Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    # r = (n_ambient - Y) / (n_ambient + Y)
    # R = |r|^2
    
    M_total = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    for mat, d in zip(materials, thicknesses):
        n = problem_materials[mat]  # This will be set in design()
        M_layer = _char_matrix(n, d, wl)
        M_total = M_total @ M_layer
    
    # Now compute the reflectance
    M11, M12 = M_total[0, 0], M_total[0, 1]
    M21, M22 = M_total[1, 0], M_total[1, 1]
    
    Y = (M11 * n_substrate + M12) / (M21 * n_substrate + M22)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = abs(r) ** 2
    return float(R)


def _mean_reflectance_percent(materials, thicknesses, problem):
    """
    Compute the mean reflectance in percent over the band.
    """
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    band_nm = problem["band_nm"]
    wl_min, wl_max = band_nm[0], band_nm[1]
    
    # 401 equally spaced wavelengths
    wls = np.linspace(wl_min, wl_max, 401)
    
    total_R = 0.0
    for wl in wls:
        R = _compute_reflectance(materials, thicknesses, wl, n_ambient, n_substrate)
        total_R += R
    
    mean_R = total_R / 401.0
    return mean_R * 100.0  # Convert to percent


def _is_valid(materials, thicknesses, problem):
    """
    Check if the design is valid.
    """
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if any(t < 0 or t > max_layer_nm for t in thicknesses):
        return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True


# Global variable to hold the problem's materials dict for use in _compute_reflectance
problem_materials = {}


def design(problem):
    """
    Design a broadband antireflection coating.
    Uses a combination of analytical initialization and local optimization.
    """
    global problem_materials
    problem_materials = problem["materials"]
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    band_nm = problem["band_nm"]
    wl_min, wl_max = band_nm[0], band_nm[1]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Wavelengths for evaluation
    wls = np.linspace(wl_min, wl_max, 401)
    
    def score(materials, thicknesses):
        """Compute mean reflectance in percent."""
        if not _is_valid(materials, thicknesses, problem):
            return float('inf')
        total_R = 0.0
        for wl in wls:
            R = _compute_reflectance(materials, thicknesses, wl, n_ambient, n_substrate)
            total_R += R
        return (total_R / 401.0) * 100.0
    
    # Initialize with a multi-layer quarter-wave stack
    # A common approach: alternate L and H layers, each quarter-wave at the center wavelength
    wl_center = 0.5 * (wl_min + wl_max)
    
    # Start with a simple 3-layer stack: H-L-H or L-H-L
    # Let's try several initial designs and pick the best
    
    initial_designs = []
    
    # Design 1: Single quarter-wave L layer
    initial_designs.append((["L"], [wl_center / (4.0 * n_L)]))
    
    # Design 2: Single quarter-wave H layer
    initial_designs.append((["H"], [wl_center / (4.0 * n_H)]))
    
    # Design 3: 3-layer H-L-H quarter-wave stack
    d_L = wl_center / (4.0 * n_L)
    d_H = wl_center / (4.0 * n_H)
    initial_designs.append((["H", "L", "H"], [d_H, d_L, d_H]))
    
    # Design 4: 3-layer L-H-L quarter-wave stack
    initial_designs.append((["L", "H", "L"], [d_L, d_H, d_L]))
    
    # Design 5: 5-layer H-L-H-L-H
    initial_designs.append((["H", "L", "H", "L", "H"], [d_H, d_L, d_H, d_L, d_H]))
    
    # Design 6: 5-layer L-H-L-H-L
    initial_designs.append((["L", "H", "L", "H", "L"], [d_L, d_H, d_L, d_H, d_L]))
    
    # Design 7: 7-layer alternating starting with H
    mats7 = ["H", "L", "H", "L", "H", "L", "H"]
    th7 = [d_H, d_L, d_H, d_L, d_H, d_L, d_H]
    initial_designs.append((mats7, th7))
    
    # Design 8: 7-layer alternating starting with L
    mats7b = ["L", "H", "L", "H", "L", "H", "L"]
    th7b = [d_L, d_H, d_L, d_H, d_L, d_H, d_L]
    initial_designs.append((mats7b, th7b))
    
    # Design 9: 9-layer alternating starting with H
    mats9 = ["H", "L", "H", "L", "H", "L", "H", "L", "H"]
    th9 = [d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H]
    initial_designs.append((mats9, th9))
    
    # Design 10: 9-layer alternating starting with L
    mats9b = ["L", "H", "L", "H", "L", "H", "L", "H", "L"]
    th9b = [d_L, d_H, d_L, d_H, d_L, d_H, d_L, d_H, d_L]
    initial_designs.append((mats9b, th9b))
    
    # Evaluate all initial designs
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    for mats, ths in initial_designs:
        if not _is_valid(mats, ths, problem):
            continue
        s = score(mats, ths)
        if s < best_score:
            best_score = s
            best_materials = mats[:]
            best_thicknesses = ths[:]
    
    if best_materials is None:
        # Fallback: single L layer
        return ["L"], [wl_center / (4.0 * n_L)]
    
    # Local optimization using Nelder-Mead on the thicknesses
    # The materials are fixed; we optimize the thicknesses
    
    def objective(thicknesses):
        """Objective function for optimization."""
        if not _is_valid(best_materials, list(thicknesses), problem):
            return 1e10
        total_R = 0.0
        for wl in wls:
            R = _compute_reflectance(best_materials, list(thicknesses), wl, n_ambient, n_substrate)
            total_R += R
        return (total_R / 401.0) * 100.0
    
    # Use scipy.optimize.minimize with Nelder-Mead
    try:
        from scipy.optimize import minimize
    except ImportError:
        minimize = None
    
    if minimize is not None:
        x0 = np.array(best_thicknesses, dtype=float)
        bounds = [(0.0, max_layer_nm)] * len(x0)
        
        # Nelder-Mead doesn't support bounds directly, so we use a penalty approach
        # or use L-BFGS-B
        # Let's try L-BFGS-B first
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < best_score:
            new_ths = [max(0.0, min(max_layer_nm, t)) for t in result.x]
            if _is_valid(best_materials, new_ths, problem):
                best_thicknesses = new_ths
                best_score = result.fun
        
        # Try Nelder-Mead as well for robustness
        elapsed = time.time() - start_time
        if elapsed < deadline - 10.0:
            result_nm = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result_nm.fun < best_score:
                new_ths = [max(0.0, min(max_layer_nm, t)) for t in result_nm.x]
                if _is_valid(best_materials, new_ths, problem):
                    best_thicknesses = new_ths
                    best_score = result_nm.fun
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed >= deadline - 5.0:
        return best_materials, best_thicknesses
    
    # Try different material sequences if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 20.0:
        # Try a few more material patterns
        additional_patterns = []
        
        # 11-layer patterns
        for start_char in ['H', 'L']:
            mats = []
            ths = []
            for i in range(11):
                c = start_char if i % 2 == 0 else ('L' if start_char == 'H' else 'H')
                mats.append(c)
                ths.append(d_H if c == 'H' else d_L)
            if _is_valid(mats, ths, problem):
                additional_patterns.append((mats, ths))
        
        # 13-layer patterns
        for start_char in ['H', 'L']:
            mats = []
            ths = []
            for i in range(13):
                c = start_char if i % 2 == 0 else ('L' if start_char == 'H' else 'H')
                mats.append(c)
                ths.append(d_H if c == 'H' else d_L)
            if _is_valid(mats, ths, problem):
                additional_patterns.append((mats, ths))
        
        # 15-layer patterns
        for start_char in ['H', 'L']:
            mats = []
            ths = []
            for i in range(15):
                c = start_char if i % 2 == 0 else ('L' if start_char == 'H' else 'H')
                mats.append(c)
                ths.append(d_H if c == 'H' else d_L)
            if _is_valid(mats, ths, problem):
                additional_patterns.append((mats, ths))
        
        # 17-layer patterns
        for start_char in ['H', 'L']:
            mats = []
            ths = []
            for i in range(17):
                c = start_char if i % 2 == 0 else ('L' if start_char == 'H' else 'H')
                mats.append(c)
                ths.append(d_H if c == 'H' else d_L)
            if _is_valid(mats, ths, problem):
                additional_patterns.append((mats, ths))
        
        # 19-layer patterns
        for start_char in ['H', 'L']:
            mats = []
            ths = []
            for i in range(19):
                c = start_char if i % 2 == 0 else ('L' if start_char == 'H' else 'H')
                mats.append(c)
                ths.append(d_H if c == 'H' else d_L)
            if _is_valid(mats, ths, problem):
                additional_patterns.append((mats, ths))
        
        for mats, ths in additional_patterns:
            if not _is_valid(mats, ths, problem):
                continue
            s = score(mats, ths)
            if s < best_score:
                best_score = s
                best_materials = mats[:]
                best_thicknesses = ths[:]
                
                # Optimize this new best
                elapsed = time.time() - start_time
                if elapsed < deadline - 10.0 and minimize is not None:
                    x0 = np.array(best_thicknesses, dtype=float)
                    bounds = [(0.0, max_layer_nm)] * len(x0)
                    result = minimize(
                        objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if result.fun < best_score:
                        new_ths = [max(0.0, min(max_layer_nm, t)) for t in result.x]
                        if _is_valid(best_materials, new_ths, problem):
                            best_thicknesses = new_ths
                            best_score = result.fun
    
    # Final check
    if not _is_valid(best_materials, best_thicknesses, problem):
        return ["L"], [wl_center / (4.0 * n_L)]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
