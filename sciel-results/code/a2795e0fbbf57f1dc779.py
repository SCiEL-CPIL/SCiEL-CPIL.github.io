"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _char_matrix_method(wavelengths, thicknesses, indices, ambient_n, substrate_n):
    """Compute reflectance using characteristic matrix method."""
    n_layers = len(thicknesses)
    reflectance = np.zeros(len(wavelengths))
    
    for i, wl in enumerate(wavelengths):
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            n_j = indices[j]
            d_j = thicknesses[j]
            delta_j = 2.0 * np.pi * n_j * d_j / wl
            
            cos_d = np.cos(delta_j)
            sin_d = np.sin(delta_j)
            
            if abs(sin_d) < 1e-15:
                sin_d = 1e-15 if sin_d >= 0 else -1e-15
            
            M_j = np.array([
                [cos_d, 1j * sin_d / n_j],
                [1j * n_j * sin_d, cos_d]
            ], dtype=complex)
            
            M = M @ M_j
        
        n_s = substrate_n
        n_0 = ambient_n
        
        B = M[0, 0] + M[0, 1] * n_s
        C = M[1, 0] * n_0 + M[1, 1] * n_s
        
        if abs(C + B) < 1e-15:
            reflectance[i] = 1.0
        else:
            r = (C - B) / (C + B)
            reflectance[i] = np.abs(r) ** 2
    
    return reflectance

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    wavelengths = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    indices = [problem["materials"][m] for m in materials]
    refl = _char_matrix_method(wavelengths, np.array(thicknesses), np.array(indices),
                               problem["ambient_index"], problem["substrate_index"])
    return 100.0 * np.mean(refl)

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if any(t < 0 or t > problem["max_layer_nm"] for t in thicknesses):
        return False
    if sum(thicknesses) > problem["max_total_nm"]:
        return False
    if any(m not in problem["materials"] for m in materials):
        return False
    return True

def _nelder_mead(materials, thicknesses, problem, time_limit):
    """Optimize thicknesses using Nelder-Mead with analytic gradients."""
    start_time = time.time()
    t = np.array(thicknesses, dtype=float)
    n = len(t)
    
    # Initial simplex
    simplex = [t.copy()]
    step = 10.0
    for i in range(n):
        t_i = t.copy()
        t_i[i] += step
        simplex.append(t_i.copy())
    
    # Evaluate initial simplex
    scores = [_mean_reflectance(materials, list(s), problem) for s in simplex]
    
    # Nelder-Mead iterations
    max_iter = 2000
    for _ in range(max_iter):
        if time.time() - start_time > time_limit - 3:
            break
        
        # Sort by score
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Check convergence
        if abs(scores[0] - scores[-1]) < 1e-12:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - simplex[-1])
        # Clip to bounds
        x_r = np.clip(x_r, 0, problem["max_layer_nm"])
        if np.sum(x_r) > problem["max_total_nm"]:
            x_r = simplex[-1]  # Skip if invalid
            scores_r = scores[-1]
        else:
            scores_r = _mean_reflectance(materials, list(x_r), problem)
        
        if scores[0] <= scores_r < scores[-1]:
            simplex[-1] = x_r
            scores[-1] = scores_r
            continue
        
        # Expansion
        if scores_r < scores[0]:
            gamma = 2.0
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, 0, problem["max_layer_nm"])
            if np.sum(x_e) > problem["max_total_nm"]:
                x_e = x_r
                scores_e = scores_r
            else:
                scores_e = _mean_reflectance(materials, list(x_e), problem)
            
            if scores_e < scores_r:
                simplex[-1] = x_e
                scores[-1] = scores_e
            else:
                simplex[-1] = x_r
                scores[-1] = scores_r
            continue
        
        # Contraction
        rho = 0.5
        x_c = centroid + rho * (simplex[-1] - centroid)
        x_c = np.clip(x_c, 0, problem["max_layer_nm"])
        if np.sum(x_c) > problem["max_total_nm"]:
            x_c = simplex[-1]
            scores_c = scores[-1]
        else:
            scores_c = _mean_reflectance(materials, list(x_c), problem)
        
        if scores_c < scores[-1]:
            simplex[-1] = x_c
            scores[-1] = scores_c
            continue
        
        # Shrink
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
            simplex[i] = np.clip(simplex[i], 0, problem["max_layer_nm"])
            if np.sum(simplex[i]) > problem["max_total_nm"]:
                simplex[i] = simplex[0]
            scores[i] = _mean_reflectance(materials, list(simplex[i]), problem)
    
    # Best point
    order = np.argsort(scores)
    best_idx = order[0]
    best_t = simplex[best_idx]
    
    # Ensure validity
    best_t = np.clip(best_t, 0, problem["max_layer_nm"])
    if np.sum(best_t) > problem["max_total_nm"]:
        best_t = np.array(thicknesses)
    
    return best_t

def _coordinate_descent(materials, thicknesses, problem, time_limit):
    """Optimize thicknesses using coordinate descent."""
    start_time = time.time()
    t = list(thicknesses)
    current_score = _mean_reflectance(materials, t, problem)
    best_t = list(t)
    best_score = current_score
    
    n_layers = len(t)
    if n_layers == 0:
        return best_t
    
    step_sizes = [20.0, 10.0, 5.0, 2.0, 1.0, 0.5]
    
    for step in step_sizes:
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3:
                break
            improved = False
            for i in range(n_layers):
                if time.time() - start_time > time_limit - 3:
                    break
                for delta in [step, -step]:
                    new_t = list(t)
                    new_t[i] += delta
                    if new_t[i] < 0 or new_t[i] > problem["max_layer_nm"]:
                        continue
                    if sum(new_t) > problem["max_total_nm"]:
                        continue
                    
                    new_score = _mean_reflectance(materials, new_t, problem)
                    if new_score < current_score - 1e-12:
                        current_score = new_score
                        t = new_t
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_t = list(t)
    
    return best_t

def design(problem):
    """Design antireflection coating using Nelder-Mead and coordinate descent."""
    start_time = time.time()
    time_limit = 145.0  # Leave 5s safety margin
    
    L_index = problem["materials"]["L"]
    H_index = problem["materials"]["H"]
    center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    # Generate candidate material sequences
    candidates = []
    
    # Quarter-wave stack: H L H L ... or L H L H ...
    for n_layers in range(4, 21, 2):
        # Start with H
        mat_h = []
        for i in range(n_layers):
            mat_h.append("H" if i % 2 == 0 else "L")
        candidates.append(mat_h)
        
        # Start with L
        mat_l = []
        for i in range(n_layers):
            mat_l.append("L" if i % 2 == 0 else "H")
        candidates.append(mat_l)
    
    # Also try odd layer counts
    for n_layers in [3, 5, 7, 9, 11, 13, 15, 17, 19]:
        mat_h = []
        for i in range(n_layers):
            mat_h.append("H" if i % 2 == 0 else "L")
        candidates.append(mat_h)
        
        mat_l = []
        for i in range(n_layers):
            mat_l.append("L" if i % 2 == 0 else "H")
        candidates.append(mat_l)
    
    # Evaluate all candidates
    best_materials = None
    best_thicknesses = None
    best_score = float("inf")
    
    for mat in candidates:
        if len(mat) > problem["max_layers"]:
            continue
        
        # Initial thicknesses: quarter-wave
        thicknesses = []
        for m in mat:
            n_idx = problem["materials"][m]
            d = center_wl / (4.0 * n_idx)
            thicknesses.append(d)
        
        if sum(thicknesses) > problem["max_total_nm"]:
            continue
        
        score = _mean_reflectance(mat, thicknesses, problem)
        if score < best_score:
            best_score = score
            best_materials = list(mat)
            best_thicknesses = list(thicknesses)
    
    if best_materials is None:
        # Fallback
        best_materials = ["L"]
        best_thicknesses = [center_wl / (4.0 * L_index)]
        best_score = _mean_reflectance(best_materials, best_thicknesses, problem)
    
    # Optimize the best candidate
    if time.time() - start_time < time_limit - 10:
        t_opt = _nelder_mead(best_materials, best_thicknesses, problem, time_limit - 10)
        if _is_valid(best_materials, list(t_opt), problem):
            score_opt = _mean_reflectance(best_materials, list(t_opt), problem)
            if score_opt < best_score:
                best_score = score_opt
                best_thicknesses = list(t_opt)
    
    # Final coordinate descent polish
    if time.time() - start_time < time_limit - 5:
        t_cd = _coordinate_descent(best_materials, best_thicknesses, problem, time_limit - 5)
        if _is_valid(best_materials, t_cd, problem):
            score_cd = _mean_reflectance(best_materials, t_cd, problem)
            if score_cd < best_score:
                best_score = score_cd
                best_thicknesses = t_cd
    
    # Ensure validity
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [center_wl / (4.0 * L_index)]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
