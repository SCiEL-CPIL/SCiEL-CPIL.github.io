"""Broadband antireflection coating design with vectorized characteristic-matrix method."""
import time
import numpy as np

# EVOLVE-BLOCK-START


def _reflectance_spectrum(n_vals, d_vals, lambdas, n_ambient, n_substrate):
    """
    Vectorized reflectance computation for a stack of layers.
    n_vals: array of refractive indices (length N)
    d_vals: array of thicknesses in nm (length N)
    lambdas: array of wavelengths in nm (length W)
    Returns: array of reflectance (0 to 1) for each wavelength.
    """
    N = len(n_vals)
    W = len(lambdas)
    
    # Initialize total matrix as identity for each wavelength
    # M_total[w] = [[1, 0], [0, 1]]
    M11 = np.ones(W, dtype=complex)
    M12 = np.zeros(W, dtype=complex)
    M21 = np.zeros(W, dtype=complex)
    M22 = np.ones(W, dtype=complex)
    
    for j in range(N):
        n = n_vals[j]
        d = d_vals[j]
        # beta = 2*pi*n*d / lambda for each wavelength
        beta = 2.0 * np.pi * n * d / lambdas
        c = np.cos(beta)
        s = np.sin(beta)
        
        # Layer matrix: [[c, i*s/n], [i*n*s, c]]
        L11 = c
        L12 = 1j * s / n
        L21 = 1j * n * s
        L22 = c
        
        # M_total = M_total @ M_layer
        # New M11 = M11*L11 + M12*L21
        # New M12 = M11*L12 + M12*L22
        # New M21 = M21*L11 + M22*L21
        # New M22 = M21*L12 + M22*L22
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
    
    # Admittance Y = (M11 + M12 * n_sub) / (M21 + M22 * n_sub)
    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    
    # Reflection coefficient r = (n_ambient - Y) / (n_ambient + Y)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    
    return R


def _evaluate(n_vals, d_vals, lambdas, n_ambient, n_substrate):
    """Compute mean reflectance in percent."""
    R = _reflectance_spectrum(n_vals, d_vals, lambdas, n_ambient, n_substrate)
    return 100.0 * np.mean(R)


def _is_valid(d_vals, max_total_nm, max_layer_nm):
    """Check thickness constraints."""
    if len(d_vals) == 0:
        return True
    total = sum(d_vals)
    if total > max_total_nm + 1e-6:
        return False
    for t in d_vals:
        if t < -1e-6 or t > max_layer_nm + 1e-6:
            return False
    return True


def _n_from_materials(materials, n_L, n_H):
    """Convert material labels to refractive index array."""
    return np.array([n_L if m == 'L' else n_H for m in materials])


def design(problem):
    """
    Design a broadband antireflection coating using vectorized evaluation
    and a multi-start local search with adaptive steps.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lambdas = np.linspace(band[0], band[1], 401)
    center_lam = 0.5 * (band[0] + band[1])
    
    def time_ok():
        return time.time() < deadline - 2.0
    
    def eval_design(materials, thicknesses):
        """Evaluate a design, return mean reflectance in percent or inf if invalid."""
        if len(materials) > max_layers:
            return float('inf')
        d = np.array(thicknesses, dtype=float)
        if not _is_valid(d, max_total_nm, max_layer_nm):
            return float('inf')
        n_vals = _n_from_materials(materials, n_L, n_H)
        return _evaluate(n_vals, d, lambdas, n_ambient, n_substrate)
    
    # Generate initial candidate designs
    candidates = []
    
    # Quarter-wave stacks at center wavelength
    for n_layers in range(1, 9):
        for start_mat in ['L', 'H']:
            materials = []
            thicknesses = []
            for i in range(n_layers):
                if (i % 2 == 0):
                    mat = start_mat
                else:
                    mat = 'H' if start_mat == 'L' else 'L'
                materials.append(mat)
                n = n_L if mat == 'L' else n_H
                thicknesses.append(center_lam / (4.0 * n))
            candidates.append((materials[:], thicknesses[:]))
    
    # Also try some non-quarter-wave initializations
    # Single layer optimized
    for n_layers in [1, 2, 3, 4, 5]:
        for start_mat in ['L', 'H']:
            materials = []
            thicknesses = []
            for i in range(n_layers):
                if (i % 2 == 0):
                    mat = start_mat
                else:
                    mat = 'H' if start_mat == 'L' else 'L'
                materials.append(mat)
                n = n_L if mat == 'L' else n_H
                # Try different optical thicknesses
                thicknesses.append(center_lam / (4.0 * n) * 0.8)
            candidates.append((materials[:], thicknesses[:]))
    
    # Evaluate all candidates and find the best
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    
    for mats, thicks in candidates:
        if not time_ok():
            break
        score = eval_design(mats, thicks)
        if score < best_score:
            best_score = score
            best_materials = mats[:]
            best_thicknesses = thicks[:]
    
    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [center_lam / (4.0 * n_L)]
        best_score = eval_design(best_materials, best_thicknesses)
    
    # Local search with adaptive step size
    current_materials = best_materials[:]
    current_thicknesses = best_thicknesses[:]
    current_score = best_score
    
    # Phase 1: Coarse local search
    step_sizes = [100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78]
    
    for step in step_sizes:
        if not time_ok():
            break
        
        improved = True
        while improved and time_ok():
            improved = False
            n_layers = len(current_materials)
            
            # Try perturbing each thickness
            for i in range(n_layers):
                for delta in [-step, step]:
                    new_thicks = current_thicknesses[:]
                    new_thicks[i] = max(0.0, min(max_layer_nm, new_thicks[i] + delta))
                    
                    if not _is_valid(new_thicks, max_total_nm, max_layer_nm):
                        continue
                    
                    score = eval_design(current_materials, new_thicks)
                    if score < current_score - 1e-10:
                        current_thicknesses = new_thicks
                        current_score = score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_materials = current_materials[:]
                            best_thicknesses = current_thicknesses[:]
            
            # Try adding a layer
            if len(current_materials) < max_layers and time_ok():
                for pos in range(len(current_materials) + 1):
                    for new_mat in ['L', 'H']:
                        new_mats = current_materials[:]
                        new_mats.insert(pos, new_mat)
                        
                        # Try a few initial thicknesses
                        for init_t in [30.0, 60.0, 100.0, 150.0]:
                            new_thicks = current_thicknesses[:]
                            new_thicks.insert(pos, init_t)
                            
                            if not _is_valid(new_thicks, max_total_nm, max_layer_nm):
                                continue
                            
                            score = eval_design(new_mats, new_thicks)
                            if score < current_score - 1e-10:
                                current_materials = new_mats
                                current_thicknesses = new_thicks
                                current_score = score
                                improved = True
                                if current_score < best_score:
                                    best_score = current_score
                                    best_materials = current_materials[:]
                                    best_thicknesses = current_thicknesses[:]
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Try removing a layer
            if len(current_materials) > 1 and time_ok():
                for i in range(len(current_materials)):
                    new_mats = current_materials[:]
                    new_mats.pop(i)
                    new_thicks = current_thicknesses[:]
                    new_thicks.pop(i)
                    
                    if not _is_valid(new_thicks, max_total_nm, max_layer_nm):
                        continue
                    
                    score = eval_design(new_mats, new_thicks)
                    if score < current_score - 1e-10:
                        current_materials = new_mats
                        current_thicknesses = new_thicks
                        current_score = score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_materials = current_materials[:]
                            best_thicknesses = current_thicknesses[:]
                        break
    
    # Phase 2: Fine-tuning with smaller steps
    if time_ok():
        for step in [0.5, 0.25, 0.1, 0.05, 0.02]:
            if not time_ok():
                break
            
            improved = True
            while improved and time_ok():
                improved = False
                n_layers = len(current_materials)
                
                for i in range(n_layers):
                    for delta in [-step, step]:
                        new_thicks = current_thicknesses[:]
                        new_thicks[i] = max(0.0, min(max_layer_nm, new_thicks[i] + delta))
                        
                        if not _is_valid(new_thicks, max_total_nm, max_layer_nm):
                            continue
                        
                        score = eval_design(current_materials, new_thicks)
                        if score < current_score - 1e-12:
                            current_thicknesses = new_thicks
                            current_score = score
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_materials = current_materials[:]
                                best_thicknesses = current_thicknesses[:]
    
    # Final validation and clamping
    if best_materials is None:
        best_materials = ['L']
        best_thicknesses = [center_lam / (4.0 * n_L)]
    
    # Clamp thicknesses
    for i in range(len(best_thicknesses)):
        best_thicknesses[i] = max(0.0, min(max_layer_nm, best_thicknesses[i]))
    
    # Check total thickness and scale if needed
    total = sum(best_thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
