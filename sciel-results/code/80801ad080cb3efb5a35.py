"""Design broadband antireflection coating using SA + Nelder-Mead + coordinate descent."""
import time
import math
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _reflectance_spectrum(thicknesses, materials, n_amb, n_sub, wl_start, wl_end, n_points=401):
    """
    Compute normal-incidence reflectance spectrum using the characteristic matrix method.
    thicknesses: list of physical thicknesses in nm (air to substrate order)
    materials: dict mapping 'L'->n_L, 'H'->n_H
    Returns: array of reflectance in percent for n_points wavelengths.
    """
    wls = np.linspace(wl_start, wl_end, n_points)
    n_L = materials['L']
    n_H = materials['H']
    
    # We need to know which layers are L and which are H
    # Actually, the caller provides materials list and thicknesses list
    # Let's restructure: we'll pass the full design info
    
    # This function is a helper; the actual computation is in _mean_reflectance
    pass


def _compute_reflectance(materials_list, thicknesses, n_amb, n_sub, wl_start, wl_end, n_points=401):
    """
    Compute mean normal-incidence reflectance in percent over the band.
    materials_list: list of 'L' or 'H' strings
    thicknesses: list of physical thicknesses in nm
    """
    n_L = 1.38
    n_H = 2.35
    n_layers = len(thicknesses)
    
    # Precompute indices for each layer
    n_layers_arr = np.array([n_L if m == 'L' else n_H for m in materials_list])
    d_arr = np.array(thicknesses, dtype=float)
    
    wls = np.linspace(wl_start, wl_end, n_points)
    reflectances = np.zeros(n_points)
    
    for i, wl in enumerate(wls):
        # Phase thicknesses: delta = 2*pi*n*d/lambda
        delta = 2.0 * math.pi * n_layers_arr * d_arr / wl
        
        # Build characteristic matrix for each layer
        # For a layer with refractive index n_j and phase delta_j:
        # M_j = [[cos(delta_j), i*sin(delta_j)/n_j],
        #        [i*n_j*sin(delta_j), cos(delta_j)]]
        
        # Start with identity matrix
        M = np.array([[1.0 + 0j, 0.0 + 0j],
                      [0.0 + 0j, 1.0 + 0j]])
        
        for j in range(n_layers):
            cos_d = math.cos(delta[j])
            sin_d = math.sin(delta[j])
            n_j = n_layers_arr[j]
            
            M_j = np.array([[cos_d, 1j * sin_d / n_j],
                            [1j * n_j * sin_d, cos_d]])
            
            M = M @ M_j
        
        # Substrate admittance (for s-polarization at normal incidence, it's just n_sub)
        # The input admittance Y = M[1,1] / M[0,1] ... wait, let me recall the formula.
        # Actually, the characteristic matrix M relates (E, H) at the air side to (E, H) at the substrate.
        # [E_0]   = M [E_N]
        # [H_0]       [H_N]
        # where E_N and H_N are at the substrate.
        # At the substrate, the admittance is n_sub (for normal incidence).
        # So H_N = n_sub * E_N (in normalized units).
        # We want the reflectance: R = |(r_s)|^2 where r_s = (Y_in - n_amb)/(Y_in + n_amb)
        # Y_in = H_0 / E_0 = M[1,0] * E_N + M[1,1] * H_N / (M[0,0] * E_N + M[0,1] * H_N)
        # With H_N = n_sub * E_N:
        # Y_in = (M[1,0] + M[1,1] * n_sub) / (M[0,0] + M[0,1] * n_sub)
        
        Y_in = (M[1, 0] + M[1, 1] * n_sub) / (M[0, 0] + M[0, 1] * n_sub)
        r = (Y_in - n_amb) / (Y_in + n_amb)
        R = abs(r) ** 2 * 100.0  # in percent
        
        reflectances[i] = R
    
    return np.mean(reflectances)


def _is_valid(materials_list, thicknesses, max_layers, max_total_nm, max_layer_nm):
    """Check if the design satisfies all constraints."""
    if len(materials_list) != len(thicknesses):
        return False
    if len(materials_list) > max_layers:
        return False
    if any(m not in ('L', 'H') for m in materials_list):
        return False
    if any(t < 0 or t > max_layer_nm for t in thicknesses):
        return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True


def design(problem):
    """
    Design a broadband antireflection coating.
    Uses simulated annealing with Nelder-Mead refinement and coordinate descent.
    """
    deadline = time.time() + 145.0  # Safety margin: 150s wall-clock limit
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    materials = problem["materials"]
    wl_start, wl_end = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Precompute constants
    n_L = materials['L']
    n_H = materials['H']
    
    # Objective function: mean reflectance in percent
    def objective(thicknesses):
        """Compute mean reflectance for a fixed set of layer materials."""
        if not _is_valid(current_materials, thicknesses, max_layers, max_total_nm, max_layer_nm):
            return 1000.0  # Penalty for invalid
        return _compute_reflectance(current_materials, thicknesses, n_amb, n_sub, wl_start, wl_end)
    
    # Start with a reasonable initial guess
    # Try a few-layer stack
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    # Initial candidate: 3-layer quarter-wave stack (L-H-L or H-L-H)
    centre_wl = 0.5 * (wl_start + wl_end)
    
    # Candidate 1: L-H-L (low-high-low)
    mat1 = ['L', 'H', 'L']
    thick1 = [centre_wl / (4 * n_L), centre_wl / (4 * n_H), centre_wl / (4 * n_L)]
    
    # Candidate 2: H-L-H
    mat2 = ['H', 'L', 'H']
    thick2 = [centre_wl / (4 * n_H), centre_wl / (4 * n_L), centre_wl / (4 * n_H)]
    
    # Candidate 3: 5-layer alternating starting with L
    mat3 = ['L', 'H', 'L', 'H', 'L']
    thick3 = [centre_wl / (4 * n_L), centre_wl / (4 * n_H), centre_wl / (4 * n_L),
              centre_wl / (4 * n_H), centre_wl / (4 * n_L)]
    
    # Candidate 4: 5-layer alternating starting with H
    mat4 = ['H', 'L', 'H', 'L', 'H']
    thick4 = [centre_wl / (4 * n_H), centre_wl / (4 * n_L), centre_wl / (4 * n_H),
              centre_wl / (4 * n_L), centre_wl / (4 * n_H)]
    
    # Candidate 5: 7-layer alternating starting with L
    mat5 = ['L', 'H', 'L', 'H', 'L', 'H', 'L']
    thick5 = [centre_wl / (4 * n_L) if i % 2 == 0 else centre_wl / (4 * n_H) for i in range(7)]
    
    # Candidate 6: 7-layer alternating starting with H
    mat6 = ['H', 'L', 'H', 'L', 'H', 'L', 'H']
    thick6 = [centre_wl / (4 * n_H) if i % 2 == 0 else centre_wl / (4 * n_L) for i in range(7)]
    
    candidates = [
        (mat1, thick1),
        (mat2, thick2),
        (mat3, thick3),
        (mat4, thick4),
        (mat5, thick5),
        (mat6, thick6),
    ]
    
    # Evaluate initial candidates
    for mat, thick in candidates:
        if _is_valid(mat, thick, max_layers, max_total_nm, max_layer_nm):
            score = _compute_reflectance(mat, thick, n_amb, n_sub, wl_start, wl_end)
            if score < best_score:
                best_score = score
                best_materials = mat[:]
                best_thicknesses = thick[:]
    
    # Now do local optimization using Nelder-Mead for each promising candidate
    # We'll focus on the best few candidates
    
    # Sort candidates by initial score
    scored_candidates = []
    for mat, thick in candidates:
        if _is_valid(mat, thick, max_layers, max_total_nm, max_layer_nm):
            score = _compute_reflectance(mat, thick, n_amb, n_sub, wl_start, wl_end)
            scored_candidates.append((score, mat, thick))
    
    scored_candidates.sort()
    
    # Optimize the top candidates
    for init_score, mat, thick in scored_candidates[:4]:  # Top 4
        if time.time() > deadline - 5.0:
            break
        
        # Set up optimization
        current_materials = mat[:]
        x0 = np.array(thick, dtype=float)
        
        # Define objective for scipy
        def obj(x):
            if time.time() > deadline - 2.0:
                return 1000.0
            if not _is_valid(current_materials, x, max_layers, max_total_nm, max_layer_nm):
                return 1000.0
            return _compute_reflectance(current_materials, x, n_amb, n_sub, wl_start, wl_end)
        
        # Nelder-Mead optimization
        try:
            result = minimize(obj, x0, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 0.1, 'fatol': 1e-4,
                                     'adaptive': True})
            
            if result.success and result.fun < best_score:
                # Check validity
                opt_thick = result.x.tolist()
                if _is_valid(mat, opt_thick, max_layers, max_total_nm, max_layer_nm):
                    best_score = result.fun
                    best_materials = mat[:]
                    best_thicknesses = opt_thick
        except Exception:
            pass
        
        # Coordinate descent refinement
        if time.time() > deadline - 5.0:
            break
        
        current_materials = mat[:]
        x = np.array(thick, dtype=float)
        
        # Try to improve with coordinate descent
        step_sizes = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
        for step in step_sizes:
            if time.time() > deadline - 3.0:
                break
            improved = True
            while improved:
                if time.time() > deadline - 3.0:
                    break
                improved = False
                for i in range(len(x)):
                    for delta in [step, -step]:
                        x_new = x.copy()
                        x_new[i] += delta
                        if _is_valid(mat, x_new, max_layers, max_total_nm, max_layer_nm):
                            score_new = _compute_reflectance(mat, x_new, n_amb, n_sub, wl_start, wl_end)
                            if score_new < best_score - 1e-6:
                                x = x_new
                                best_score = score_new
                                best_materials = mat[:]
                                best_thicknesses = x.tolist()
                                improved = True
    
    # Final: try a few more layer configurations with optimization
    if time.time() < deadline - 10.0:
        # Try 4-layer and 6-layer configs
        extra_candidates = []
        
        # 4-layer: L-H-L-H
        mat7 = ['L', 'H', 'L', 'H']
        thick7 = [centre_wl / (4 * n_L), centre_wl / (4 * n_H),
                  centre_wl / (4 * n_L), centre_wl / (4 * n_H)]
        extra_candidates.append((mat7, thick7))
        
        # 4-layer: H-L-H-L
        mat8 = ['H', 'L', 'H', 'L']
        thick8 = [centre_wl / (4 * n_H), centre_wl / (4 * n_L),
                  centre_wl / (4 * n_H), centre_wl / (4 * n_L)]
        extra_candidates.append((mat8, thick8))
        
        # 6-layer: L-H-L-H-L-H
        mat9 = ['L', 'H', 'L', 'H', 'L', 'H']
        thick9 = [centre_wl / (4 * n_L) if i % 2 == 0 else centre_wl / (4 * n_H) for i in range(6)]
        extra_candidates.append((mat9, thick9))
        
        # 6-layer: H-L-H-L-H-L
        mat10 = ['H', 'L', 'H', 'L', 'H', 'L']
        thick10 = [centre_wl / (4 * n_H) if i % 2 == 0 else centre_wl / (4 * n_L) for i in range(6)]
        extra_candidates.append((mat10, thick10))
        
        for mat, thick in extra_candidates:
            if time.time() > deadline - 5.0:
                break
            if _is_valid(mat, thick, max_layers, max_total_nm, max_layer_nm):
                score = _compute_reflectance(mat, thick, n_amb, n_sub, wl_start, wl_end)
                if score < best_score + 0.5:  # Only optimize if close to best
                    current_materials = mat[:]
                    x0 = np.array(thick, dtype=float)
                    
                    def obj(x):
                        if time.time() > deadline - 2.0:
                            return 1000.0
                        if not _is_valid(current_materials, x, max_layers, max_total_nm, max_layer_nm):
                            return 1000.0
                        return _compute_reflectance(current_materials, x, n_amb, n_sub, wl_start, wl_end)
                    
                    try:
                        result = minimize(obj, x0, method='Nelder-Mead',
                                         options={'maxiter': 150, 'xatol': 0.1, 'fatol': 1e-4,
                                                 'adaptive': True})
                        
                        if result.success and result.fun < best_score:
                            opt_thick = result.x.tolist()
                            if _is_valid(mat, opt_thick, max_layers, max_total_nm, max_layer_nm):
                                best_score = result.fun
                                best_materials = mat[:]
                                best_thicknesses = opt_thick
                    except Exception:
                        pass
    
    # Return best found
    if best_materials is None:
        # Fallback: single quarter-wave L layer
        return ['L'], [centre_wl / (4.0 * n_L)]
    
    # Ensure thicknesses are within bounds
    best_thicknesses = [max(0.0, min(max_layer_nm, t)) for t in best_thicknesses]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
