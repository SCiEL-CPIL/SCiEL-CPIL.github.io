"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

# Deadline guard: 150s wall-clock, keep 10s safety margin
_DEADLINE = 140.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _remaining():
    return _DEADLINE - _elapsed()

def _characteristic_matrix(wavelengths, n_layers, thicknesses, indices, n_ambient, n_substrate):
    """
    Compute reflectance using the characteristic matrix method.
    wavelengths: array of wavelengths in nm
    n_layers: number of layers
    thicknesses: array of physical thicknesses in nm
    indices: array of refractive indices for each layer
    n_ambient: ambient refractive index
    n_substrate: substrate refractive index
    Returns: array of reflectance values (0 to 1)
    """
    if n_layers == 0:
        # Bare substrate
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full_like(wavelengths, r * r, dtype=np.float64)
    
    # Optical thickness in radians: delta = 2*pi*n*d/lambda
    # For vectorized computation over wavelengths
    # M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
    
    # Build the total characteristic matrix
    # Start with identity
    M = np.ones((2, 2, len(wavelengths)), dtype=np.complex128)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses[i]
        delta = 2.0 * np.pi * n_i * d_i / wavelengths  # shape (len(wavelengths),)
        
        cos_d = np.cos(delta)  # real
        sin_d = np.sin(delta)  # real
        
        # M_i matrix for each wavelength
        # M_i = [[cos_d, 1j * sin_d / n_i],
        #         [1j * n_i * sin_d, cos_d]]
        
        M_i = np.ones((2, 2, len(wavelengths)), dtype=np.complex128)
        M_i[0, 0, :] = cos_d
        M_i[0, 1, :] = 1j * sin_d / n_i
        M_i[1, 0, :] = 1j * n_i * sin_d
        M_i[1, 1, :] = cos_d
        
        # M = M @ M_i
        # Matrix multiplication: (2,2) @ (2,2,N) -> (2,2,N)
        M = np.einsum('ab,bcn->an', M, M_i)
    
    # Admittance: Y = M[1,0] / M[0,0]
    # Reflectance: R = | (Y - n_ambient) / (Y + n_ambient) |^2
    # But we need to account for the substrate
    # Actually, the characteristic matrix gives us the relation:
    # [E_0, H_0] = M_total @ [E_N, H_N]
    # At the substrate: E_N/H_N = n_substrate (for s-polarization, normal incidence)
    # So Y = H_0/E_0 = (M[1,0]*n_sub + M[1,1]) / (M[0,0]*n_sub + M[0,1])
    # Wait, let me be more careful.
    
    # The characteristic matrix M relates the fields at the first interface to the last:
    # [E_0]   [M00 M01] [E_N]
    # [H_0] = [M10 M11] [H_N]
    
    # At the substrate, H_N/E_N = n_substrate (for normal incidence, s-polarization)
    # So E_N = E_N, H_N = n_sub * E_N
    # [E_0] = [M00 + M01*n_sub] E_N
    # [H_0] = [M10 + M11*n_sub] E_N
    # Y = H_0/E_0 = (M10 + M11*n_sub) / (M00 + M01*n_sub)
    
    Y = (M[1, 0, :] + M[1, 1, :] * n_substrate) / (M[0, 0, :] + M[0, 1, :] * n_substrate)
    
    # Reflectance: R = | (Y - n_ambient) / (Y + n_ambient) |^2
    r = (Y - n_ambient) / (Y + n_ambient)
    R = np.abs(r) ** 2
    
    return R

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    mat_indices = problem["materials"]
    
    n_layers = len(materials)
    if n_layers == 0:
        return 4.26  # bare glass
    
    indices = np.array([mat_indices[m] for m in materials], dtype=np.float64)
    thicknesses_arr = np.array(thicknesses, dtype=np.float64)
    
    R = _characteristic_matrix(wavelengths, n_layers, thicknesses_arr, indices, n_ambient, n_substrate)
    return float(np.mean(R) * 100.0)

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if len(materials) == 0:
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    for m in materials:
        if m not in ("L", "H"):
            return False
    return True

def _random_initial_design(problem):
    """Generate a random valid design."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_layers = np.random.randint(2, max_layers + 1)
    materials = list(np.random.choice(["L", "H"], size=n_layers))
    
    # Random thicknesses that sum to <= max_total_nm
    # Use Dirichlet-like distribution
    weights = np.random.dirichlet(np.ones(n_layers))
    total = np.random.uniform(max_total_nm * 0.5, max_total_nm)
    thicknesses = weights * total
    # Ensure each thickness is <= max_layer_nm
    thicknesses = np.minimum(thicknesses, max_layer_nm)
    
    return materials, thicknesses.tolist()

def _coordinate_descent(materials, thicknesses, problem, max_iter=500, tol=1e-8):
    """Perform coordinate descent optimization on thicknesses."""
    current_score = _mean_reflectance(materials, thicknesses, problem)
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    for iteration in range(max_iter):
        if _remaining() < 2.0:
            break
        
        improved = False
        n_layers = len(materials)
        
        for i in range(n_layers):
            # Try to optimize thickness[i]
            best_t = thicknesses[i]
            best_score = current_score
            
            # Golden section search or simple grid
            # Try a few candidate values
            candidates = [
                0.0,
                max_layer_nm,
                thicknesses[i] * 0.5,
                thicknesses[i] * 1.5,
                thicknesses[i] + 10.0,
                thicknesses[i] - 10.0,
                thicknesses[i] + 50.0,
                thicknesses[i] - 50.0,
            ]
            # Also try quarter-wave thicknesses for the center wavelength
            centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
            n_layer = problem["materials"][materials[i]]
            qw = centre / (4.0 * n_layer)
            candidates.extend([qw, qw * 0.8, qw * 1.2, qw * 0.5, qw * 2.0])
            
            # Filter valid candidates
            for t in candidates:
                if t < 0 or t > max_layer_nm:
                    continue
                # Check total thickness constraint
                other_sum = sum(thicknesses) - thicknesses[i]
                if other_sum + t > max_total_nm:
                    continue
                
                new_thicknesses = thicknesses.copy()
                new_thicknesses[i] = t
                new_score = _mean_reflectance(materials, new_thicknesses, problem)
                
                if new_score < best_score - tol:
                    best_score = new_score
                    best_t = t
                    improved = True
        
        thicknesses[i] = best_t
        current_score = best_score
    
    return materials, thicknesses, current_score

def _perturb_sequence(materials, problem):
    """Perturb the layer sequence."""
    n_layers = len(materials)
    if n_layers == 0:
        return materials
    
    new_materials = materials.copy()
    action = np.random.choice(["flip", "swap", "insert", "remove"])
    
    if action == "flip" and n_layers > 0:
        idx = np.random.randint(n_layers)
        new_materials[idx] = "H" if new_materials[idx] == "L" else "L"
    elif action == "swap" and n_layers > 1:
        i, j = np.random.choice(n_layers, 2, replace=False)
        new_materials[i], new_materials[j] = new_materials[j], new_materials[i]
    elif action == "insert" and n_layers < problem["max_layers"]:
        idx = np.random.randint(n_layers + 1)
        new_mat = "H" if np.random.random() < 0.5 else "L"
        new_materials.insert(idx, new_mat)
    elif action == "remove" and n_layers > 2:
        idx = np.random.randint(n_layers)
        del new_materials[idx]
    
    return new_materials

def design(problem):
    """Design an antireflection coating stack."""
    global _start_time
    _start_time = time.time()
    
    np.random.seed(42)
    
    best_materials = ["L"]
    best_thicknesses = [0.5 * (problem["band_nm"][0] + problem["band_nm"][1]) / (4.0 * problem["materials"]["L"])]
    best_score = _mean_reflectance(best_materials, best_thicknesses, problem)
    
    # Multi-seed initialization
    n_seeds = 10
    seeds = []
    
    for s in range(n_seeds):
        if _remaining() < 5.0:
            break
        
        mats, thick = _random_initial_design(problem)
        if not _is_valid(mats, thick, problem):
            continue
        score = _mean_reflectance(mats, thick, problem)
        seeds.append((mats, thick, score))
        
        if score < best_score:
            best_score = score
            best_materials = mats.copy()
            best_thicknesses = thick.copy()
    
    # Also try some structured designs
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    qw_L = centre / (4.0 * n_L)
    qw_H = centre / (4.0 * n_H)
    
    structured_designs = [
        (["L"], [qw_L]),
        (["H"], [qw_H]),
        (["L", "H"], [qw_L, qw_H]),
        (["H", "L"], [qw_H, qw_L]),
        (["L", "H", "L"], [qw_L, qw_H, qw_L]),
        (["H", "L", "H"], [qw_H, qw_L, qw_H]),
        (["L", "H", "L", "H"], [qw_L, qw_H, qw_L, qw_H]),
        (["H", "L", "H", "L"], [qw_H, qw_L, qw_H, qw_L]),
        (["L", "H", "L", "H", "L"], [qw_L, qw_H, qw_L, qw_H, qw_L]),
        (["H", "L", "H", "L", "H"], [qw_H, qw_L, qw_H, qw_L, qw_H]),
    ]
    
    for mats, thick in structured_designs:
        if _remaining() < 5.0:
            break
        if not _is_valid(mats, thick, problem):
            continue
        score = _mean_reflectance(mats, thick, problem)
        if score < best_score:
            best_score = score
            best_materials = mats.copy()
            best_thicknesses = thick.copy()
        seeds.append((mats.copy(), thick.copy(), score))
    
    # Coordinate descent on the best few seeds
    # Sort seeds by score
    seeds.sort(key=lambda x: x[2])
    
    # Optimize top seeds
    n_optimize = min(5, len(seeds))
    for i in range(n_optimize):
        if _remaining() < 10.0:
            break
        
        mats, thick, score = seeds[i]
        if not _is_valid(mats, thick, problem):
            continue
        
        # Coordinate descent
        mats_opt, thick_opt, score_opt = _coordinate_descent(mats, thick, problem, max_iter=200)
        
        if _is_valid(mats_opt, thick_opt, problem):
            if score_opt < best_score:
                best_score = score_opt
                best_materials = mats_opt.copy()
                best_thicknesses = thick_opt.copy()
    
    # Try some sequence perturbations on the best design
    n_perturb = 10
    for p in range(n_perturb):
        if _remaining() < 15.0:
            break
        
        mats_new = _perturb_sequence(best_materials, problem)
        
        # Generate corresponding thicknesses
        if len(mats_new) != len(best_thicknesses):
            # Need to generate new thicknesses
            n_layers = len(mats_new)
            if n_layers == 0:
                continue
            weights = np.random.dirichlet(np.ones(n_layers))
            total = np.random.uniform(max_total_nm * 0.5, max_total_nm) if 'max_total_nm' in dir() else 500
            max_total_nm = problem["max_total_nm"]
            total = np.random.uniform(max_total_nm * 0.5, max_total_nm)
            thick_new = (weights * total).tolist()
            thick_new = [min(t, problem["max_layer_nm"]) for t in thick_new]
        else:
            thick_new = best_thicknesses.copy()
        
        if not _is_valid(mats_new, thick_new, problem):
            continue
        
        # Coordinate descent
        mats_opt, thick_opt, score_opt = _coordinate_descent(mats_new, thick_new, problem, max_iter=100)
        
        if _is_valid(mats_opt, thick_opt, problem):
            if score_opt < best_score:
                best_score = score_opt
                best_materials = mats_opt.copy()
                best_thicknesses = thick_opt.copy()
    
    # Final check
    if not _is_valid(best_materials, best_thicknesses, problem):
        # Fallback to a simple valid design
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
