"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _characteristic_matrix(n, d, wavelen):
    """Compute the 2x2 characteristic matrix for a single layer.
    n: refractive index (scalar or array)
    d: physical thickness (scalar or array)
    wavelen: wavelength in nm (scalar or array)
    """
    beta = 2.0 * np.pi * n * d / wavelen
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    # M = [[cos_beta, i*sin_beta/n], [i*n*sin_beta, cos_beta]]
    # We'll work with the matrix elements directly
    m11 = cos_beta
    m12 = 1j * sin_beta / n
    m21 = 1j * n * sin_beta
    m22 = cos_beta
    return m11, m12, m21, m22


def _reflectance(materials_list, thicknesses, wavelens, ambient_index, substrate_index, nmat):
    """Compute reflectance spectrum using characteristic matrix method.
    materials_list: list of 'L'/'H'
    thicknesses: list of thicknesses in nm
    wavelens: array of wavelengths
    Returns array of reflectance values (0 to 1).
    """
    n_layers = len(materials_list)
    if n_layers == 0:
        # Bare substrate
        r0 = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full_like(wavelens, r0 * r0, dtype=float)
    
    # Build total characteristic matrix by multiplying layer matrices
    # Start with identity
    total_m11 = np.ones_like(wavelens, dtype=complex)
    total_m12 = np.zeros_like(wavelens, dtype=complex)
    total_m21 = np.zeros_like(wavelens, dtype=complex)
    total_m22 = np.ones_like(wavelens, dtype=complex)
    
    for i in range(n_layers):
        n_i = nmat[materials_list[i]]
        d_i = thicknesses[i]
        m11, m12, m21, m22 = _characteristic_matrix(n_i, d_i, wavelens)
        
        # total = total @ M_i
        new_m11 = total_m11 * m11 + total_m12 * m21
        new_m12 = total_m11 * m12 + total_m12 * m22
        new_m21 = total_m21 * m11 + total_m22 * m21
        new_m22 = total_m21 * m12 + total_m22 * m22
        
        total_m11 = new_m11
        total_m12 = new_m12
        total_m21 = new_m21
        total_m22 = new_m22
    
    # Admittance at air side
    y0 = ambient_index
    ys = substrate_index
    
    # B = M11 * ys + M12 * ys^2 ... wait, let me recall the formula
    # For normal incidence:
    # B = M11 * n_s + M12 * n_s^2  -- no, that's not right
    # Actually: B = M11 + M12 * n_s / n_s ... 
    # Let me use the standard formula:
    # B = M11 * n_s + M12 * n_s^2 / n_s ... 
    # 
    # Standard: 
    # B = M11 * n_s + M12 * n_s^2  -- no
    # 
    # The characteristic matrix relates:
    # [E, H]_air = M_total * [E, H]_substrate
    # At substrate: H/E = n_s (for forward wave)
    # So: [E, H]_air = M_total * [1, n_s]^T (up to a phase)
    # E_air = M11 * 1 + M12 * n_s
    # H_air = M21 * 1 + M22 * n_s
    # Admittance Y = H_air / E_air
    # Reflection coefficient r = (Y0 - Y) / (Y0 + Y)
    # where Y0 = n_ambient
    
    B = total_m11 * 1.0 + total_m12 * ys
    C = total_m21 * 1.0 + total_m22 * ys
    
    Y = C / B  # admittance
    
    r = (y0 - Y) / (y0 + Y)
    R = np.abs(r) ** 2
    
    # Clip to [0, 1] for numerical safety
    R = np.clip(R, 0.0, 1.0)
    return R


def _mean_reflectance(materials_list, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    wavelens = np.linspace(band[0], band[1], 401)
    R = _reflectance(materials_list, thicknesses, wavelens, 
                     problem["ambient_index"], problem["substrate_index"], 
                     problem["materials"])
    return np.mean(R) * 100.0


def _is_valid(materials_list, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    if len(materials_list) != len(thicknesses):
        return False
    if len(materials_list) > problem["max_layers"]:
        return False
    total = sum(thicknesses)
    if total > problem["max_total_nm"] + 1e-9:
        return False
    for t in thicknesses:
        if t < -1e-9 or t > problem["max_layer_nm"] + 1e-9:
            return False
    for m in materials_list:
        if m not in problem["materials"]:
            return False
    return True


def _evaluate(materials_list, thicknesses, problem):
    """Evaluate a design. Returns (mean_reflectance_percent, is_valid)."""
    if not _is_valid(materials_list, thicknesses, problem):
        return float('inf'), False
    mr = _mean_reflectance(materials_list, thicknesses, problem)
    return mr, True


def _random_design(problem, rng):
    """Generate a random valid design."""
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    materials = list(problem["materials"].keys())
    
    n_layers = rng.integers(1, max_layers + 1)
    materials_list = [materials[rng.integers(0, len(materials))] for _ in range(n_layers)]
    
    # Random thicknesses that sum to <= max_total
    # Use Dirichlet-like approach
    raw = rng.random(n_layers)
    raw += 1e-10
    raw /= raw.sum()
    thicknesses = [raw[i] * max_total * 0.8 for i in range(n_layers)]
    
    # Clamp individual thicknesses
    thicknesses = [min(t, max_layer) for t in thicknesses]
    
    # If sum exceeds max_total, scale down
    total = sum(thicknesses)
    if total > max_total:
        scale = max_total / total
        thicknesses = [t * scale for t in thicknesses]
    
    return materials_list, thicknesses


def _coordinate_descent(materials_list, thicknesses, problem, max_iter=500, tol=1e-10, deadline=None):
    """Coordinate descent on thicknesses, keeping material sequence fixed."""
    current_mr, _ = _evaluate(materials_list, thicknesses, problem)
    if not _is_valid(materials_list, thicknesses, problem):
        return materials_list, thicknesses, current_mr
    
    best_materials = list(materials_list)
    best_thicknesses = list(thicknesses)
    best_mr = current_mr
    
    n_layers = len(materials_list)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n_layers):
            # Try to optimize thickness i
            # Use a simple line search / golden section
            t_best = thicknesses[i]
            mr_best = best_mr
            
            # Try a few perturbations
            for delta in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
                for sign in [-1, 1]:
                    new_t = t_best + sign * delta
                    if new_t < 0 or new_t > problem["max_layer_nm"]:
                        continue
                    new_thicknesses = list(thicknesses)
                    new_thicknesses[i] = new_t
                    if sum(new_thicknesses) > problem["max_total_nm"] + 1e-9:
                        continue
                    mr, valid = _evaluate(materials_list, new_thicknesses, problem)
                    if valid and mr < mr_best - 1e-12:
                        mr_best = mr
                        t_best = new_t
                        improved = True
            thicknesses[i] = t_best
            best_mr = mr_best
        
        if not improved:
            break
        
        best_materials = list(materials_list)
        best_thicknesses = list(thicknesses)
        best_mr = current_mr
    
    return best_materials, best_thicknesses, best_mr


def _try_add_layer(materials_list, thicknesses, problem, deadline=None):
    """Try adding a layer and optimize."""
    max_layers = problem["max_layers"]
    if len(materials_list) >= max_layers:
        return materials_list, thicknesses, _evaluate(materials_list, thicknesses, problem)[0]
    
    materials = list(problem["materials"].keys())
    best_mr = _evaluate(materials_list, thicknesses, problem)[0]
    best_mats = list(materials_list)
    best_thick = list(thicknesses)
    
    for pos in range(len(materials_list) + 1):
        for mat in materials:
            new_mats = list(materials_list)
            new_mats.insert(pos, mat)
            new_thick = list(thicknesses)
            new_thick.insert(pos, 100.0)  # initial guess
            
            # Quick check
            if sum(new_thick) > problem["max_total_nm"]:
                continue
            
            # Optimize the new thickness
            m_opt, t_opt, mr_opt = _coordinate_descent(new_mats, new_thick, problem, max_iter=100, deadline=deadline)
            
            if mr_opt < best_mr - 1e-12:
                best_mr = mr_opt
                best_mats = m_opt
                best_thick = t_opt
    
    return best_mats, best_thick, best_mr


def _try_remove_layer(materials_list, thicknesses, problem, deadline=None):
    """Try removing a layer and see if it improves."""
    if len(materials_list) <= 1:
        return materials_list, thicknesses, _evaluate(materials_list, thicknesses, problem)[0]
    
    best_mr = _evaluate(materials_list, thicknesses, problem)[0]
    best_mats = list(materials_list)
    best_thick = list(thicknesses)
    
    for pos in range(len(materials_list)):
        new_mats = list(materials_list)
        new_mats.pop(pos)
        new_thick = list(thicknesses)
        new_thick.pop(pos)
        
        if not new_mats:
            continue
            
        m_opt, t_opt, mr_opt = _coordinate_descent(new_mats, new_thick, problem, max_iter=100, deadline=deadline)
        
        if mr_opt < best_mr - 1e-12:
            best_mr = mr_opt
            best_mats = m_opt
            best_thick = t_opt
    
    return best_mats, best_thick, best_mr


def _try_swap_materials(materials_list, thicknesses, problem, deadline=None):
    """Try swapping material of a layer."""
    if len(materials_list) == 0:
        return materials_list, thicknesses, _evaluate(materials_list, thicknesses, problem)[0]
    
    materials = list(problem["materials"].keys())
    best_mr = _evaluate(materials_list, thicknesses, problem)[0]
    best_mats = list(materials_list)
    best_thick = list(thicknesses)
    
    for pos in range(len(materials_list)):
        current_mat = materials_list[pos]
        for mat in materials:
            if mat == current_mat:
                continue
            new_mats = list(materials_list)
            new_mats[pos] = mat
            m_opt, t_opt, mr_opt = _coordinate_descent(new_mats, thicknesses, problem, max_iter=100, deadline=deadline)
            
            if mr_opt < best_mr - 1e-12:
                best_mr = mr_opt
                best_mats = m_opt
                best_thick = t_opt
    
    return best_mats, best_thick, best_mr


def design(problem):
    """Design an antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Start with a reasonable initial design
    # Quarter-wave stack at center wavelength
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    
    # Initial: single L quarter-wave
    init_mats = ["L"]
    init_thick = [centre / (4.0 * nL)]
    
    best_mats = list(init_mats)
    best_thick = list(init_thick)
    best_mr, _ = _evaluate(best_mats, best_thick, problem)
    
    # Coordinate descent on initial design
    best_mats, best_thick, best_mr = _coordinate_descent(best_mats, best_thick, problem, 
                                                         max_iter=200, deadline=deadline)
    
    # Try adding layers
    for _ in range(10):
        if time.time() > deadline:
            break
        new_mats, new_thick, new_mr = _try_add_layer(best_mats, best_thick, problem, deadline)
        if new_mr < best_mr - 1e-12:
            best_mats, best_thick, best_mr = new_mats, new_thick, new_mr
            # Further optimize
            best_mats, best_thick, best_mr = _coordinate_descent(best_mats, best_thick, problem,
                                                                 max_iter=200, deadline=deadline)
        else:
            break
    
    # Try removing layers
    for _ in range(5):
        if time.time() > deadline:
            break
        new_mats, new_thick, new_mr = _try_remove_layer(best_mats, best_thick, problem, deadline)
        if new_mr < best_mr - 1e-12:
            best_mats, best_thick, best_mr = new_mats, new_thick, new_mr
            best_mats, best_thick, best_mr = _coordinate_descent(best_mats, best_thick, problem,
                                                                 max_iter=200, deadline=deadline)
        else:
            break
    
    # Try swapping materials
    for _ in range(5):
        if time.time() > deadline:
            break
        new_mats, new_thick, new_mr = _try_swap_materials(best_mats, best_thick, problem, deadline)
        if new_mr < best_mr - 1e-12:
            best_mats, best_thick, best_mr = new_mats, new_thick, new_mr
            best_mats, best_thick, best_mr = _coordinate_descent(best_mats, best_thick, problem,
                                                                 max_iter=200, deadline=deadline)
        else:
            break
    
    # Multi-seed random restarts
    n_restarts = 20
    for seed in range(n_restarts):
        if time.time() > deadline:
            break
        rng = np.random.default_rng(seed * 12345 + 67890)
        mats, thick = _random_design(problem, rng)
        mats, thick, mr = _coordinate_descent(mats, thick, problem, max_iter=150, deadline=deadline)
        
        if mr < best_mr - 1e-12:
            best_mats, best_thick, best_mr = mats, thick, mr
            # Try to improve further
            best_mats, best_thick, best_mr = _coordinate_descent(best_mats, best_thick, problem,
                                                                 max_iter=200, deadline=deadline)
            if time.time() < deadline:
                new_mats, new_thick, new_mr = _try_add_layer(best_mats, best_thick, problem, deadline)
                if new_mr < best_mr - 1e-12:
                    best_mats, best_thick, best_mr = new_mats, new_thick, new_mr
                    best_mats, best_thick, best_mr = _coordinate_descent(best_mats, best_thick, problem,
                                                                         max_iter=200, deadline=deadline)
    
    # Final polish with more iterations
    if time.time() < deadline - 5.0:
        best_mats, best_thick, best_mr = _coordinate_descent(best_mats, best_thick, problem,
                                                             max_iter=500, deadline=deadline)
    
    # Ensure validity
    if not _is_valid(best_mats, best_thick, problem):
        # Fallback to simple quarter-wave
        best_mats = ["L"]
        best_thick = [centre / (4.0 * nL)]
    
    # Clamp thicknesses to valid range
    best_thick = [max(0.0, min(t, problem["max_layer_nm"])) for t in best_thick]
    
    # Ensure total doesn't exceed max
    total = sum(best_thick)
    if total > problem["max_total_nm"]:
        scale = problem["max_total_nm"] / total
        best_thick = [t * scale for t in best_thick]
    
    return best_mats, best_thick
# EVOLVE-BLOCK-END
