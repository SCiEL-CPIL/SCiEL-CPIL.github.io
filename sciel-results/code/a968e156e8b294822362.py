"""Design broadband antireflection coating using characteristic-matrix method."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _compute_reflectance_vectorized(materials, thicknesses, wavelengths, n0, ns, nL, nH):
    """
    Compute reflectance at all wavelengths simultaneously using vectorized ops.
    Returns array of reflectance fractions.
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full(len(wavelengths), abs(r)**2)
    
    # M11, M12, M21, M22 as complex arrays of length len(wavelengths)
    M11 = np.ones(len(wavelengths), dtype=complex)
    M12 = np.zeros(len(wavelengths), dtype=complex)
    M21 = np.zeros(len(wavelengths), dtype=complex)
    M22 = np.ones(len(wavelengths), dtype=complex)
    
    for i, (mat, d) in enumerate(zip(materials, thicknesses)):
        n = nL if mat == 'L' else nH
        beta = 2.0 * np.pi * n * d / wavelengths
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        L11 = cos_b
        L12 = 1j * sin_b / n
        L21 = 1j * n * sin_b
        L22 = cos_b
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    B = M11 + M12 * ns
    C = M21 + M22 * ns
    Y = C / B
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r)**2
    return np.real(R)


def _mean_reflectance_percent(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band using vectorized computation."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band0, band1 = problem["band_nm"]
    
    wavelengths = np.linspace(band0, band1, 401)
    R_array = _compute_reflectance_vectorized(materials, thicknesses, wavelengths, n0, ns, nL, nH)
    mean_R = np.mean(R_array)
    return float(mean_R * 100.0)


def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if len(materials) == 0:
        return True
    
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    
    if sum(thicknesses) > max_total_nm:
        return False
    
    for m in materials:
        if m not in ('L', 'H'):
            return False
    
    return True


def _evaluate_design(materials, thicknesses, problem):
    """Evaluate a design, returning (score, valid)."""
    if not _is_valid(materials, thicknesses, problem):
        return float('inf'), False
    score = _mean_reflectance_percent(materials, thicknesses, problem)
    return score, True


def _local_optimize(materials, thicknesses, problem, deadline, rng):
    """Local optimization using coordinate descent with adaptive steps."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    current_mats = materials[:]
    current_thick = thicknesses[:]
    current_score = _mean_reflectance_percent(current_mats, current_thick, problem)
    
    if not _is_valid(current_mats, current_thick, problem):
        return materials, thicknesses, float('inf')
    
    # Adaptive step sizes
    step_sizes = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.1]
    
    for step in step_sizes:
        if time.time() > deadline - 5:
            break
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            for i in range(len(current_thick)):
                if time.time() > deadline - 5:
                    break
                
                orig = current_thick[i]
                # Try both directions
                for sign in [1, -1]:
                    new_thick = current_thick[:]
                    new_thick[i] = orig + sign * step
                    
                    if new_thick[i] < 0 or new_thick[i] > max_layer_nm:
                        continue
                    if sum(new_thick) > max_total_nm:
                        continue
                    
                    trial_score = _mean_reflectance_percent(current_mats, new_thick, problem)
                    if trial_score < current_score - 1e-12:
                        current_thick = new_thick
                        current_score = trial_score
                        improved = True
                        break
    
    return current_mats, current_thick, current_score


def _nelder_mead_optimize(materials, thicknesses, problem, deadline, rng, max_iter=200):
    """
    Nelder-Mead simplex optimization on thicknesses.
    Materials are fixed; only thicknesses are optimized.
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        return materials, thicknesses, float('inf')
    
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    # Initialize simplex
    initial_thick = np.array(thicknesses, dtype=float)
    
    # Create simplex: initial point plus perturbations
    simplex = [initial_thick.copy()]
    for i in range(n_layers):
        perturbed = initial_thick.copy()
        perturbed[i] = initial_thick[i] + 10.0  # 10nm perturbation
        if perturbed[i] > max_layer_nm:
            perturbed[i] = max_layer_nm
        if perturbed[i] < 0:
            perturbed[i] = 0.0
        if sum(perturbed) <= max_total_nm:
            simplex.append(perturbed)
        else:
            # Try smaller perturbation
            perturbed[i] = initial_thick[i] + 5.0
            if perturbed[i] > max_layer_nm:
                perturbed[i] = max_layer_nm
            if sum(perturbed) <= max_total_nm:
                simplex.append(perturbed)
            else:
                # Use a different layer for perturbation
                for j in range(n_layers):
                    if j != i:
                        perturbed2 = initial_thick.copy()
                        perturbed2[j] = initial_thick[j] + 5.0
                        if 0 <= perturbed2[j] <= max_layer_nm and sum(perturbed2) <= max_total_nm:
                            simplex.append(perturbed2)
                            break
                else:
                    simplex.append(initial_thick.copy())
    
    # Evaluate all simplex points
    def objective(thick_arr):
        if not _is_valid(materials, list(thick_arr), problem):
            return float('inf')
        return _mean_reflectance_percent(materials, list(thick_arr), problem)
    
    simplex_scores = [objective(pt) for pt in simplex]
    
    # Nelder-Mead parameters
    alpha = 0.5  # reflection
    gamma = 2.0  # expansion
    rho = 0.5    # contraction
    sigma = 0.5  # shrink
    
    for iteration in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        # Sort simplex by objective value
        sorted_pairs = sorted(zip(simplex_scores, simplex), key=lambda x: x[0])
        simplex = [pt for _, pt in sorted_pairs]
        simplex_scores = [s for s, _ in sorted_pairs]
        
        # If converged
        if abs(simplex_scores[0] - simplex_scores[-1]) < 1e-10:
            break
        
        worst = simplex[-1]
        worst_score = simplex_scores[-1]
        best = simplex[0]
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        reflected = centroid + alpha * (centroid - worst)
        reflected = np.clip(reflected, 0, max_layer_nm)
        if sum(reflected) > max_total_nm:
            # Scale down to fit
            scale = max_total_nm / sum(reflected)
            reflected = reflected * scale
            reflected = np.clip(reflected, 0, max_layer_nm)
        
        reflected_score = objective(reflected)
        
        if reflected_score < simplex_scores[0]:
            # Expansion
            expanded = centroid + gamma * (reflected - centroid)
            expanded = np.clip(expanded, 0, max_layer_nm)
            if sum(expanded) > max_total_nm:
                scale = max_total_nm / sum(expanded)
                expanded = expanded * scale
                expanded = np.clip(expanded, 0, max_layer_nm)
            
            expanded_score = objective(expanded)
            if expanded_score < reflected_score:
                simplex[-1] = expanded
                simplex_scores[-1] = expanded_score
            else:
                simplex[-1] = reflected
                simplex_scores[-1] = reflected_score
        elif reflected_score < simplex_scores[-2]:
            # Accept reflection
            simplex[-1] = reflected
            simplex_scores[-1] = reflected_score
        else:
            # Contraction
            contracted = centroid + rho * (worst - centroid)
            contracted = np.clip(contracted, 0, max_layer_nm)
            if sum(contracted) > max_total_nm:
                scale = max_total_nm / sum(contracted)
                contracted = contracted * scale
                contracted = np.clip(contracted, 0, max_layer_nm)
            
            contracted_score = objective(contracted)
            if contracted_score < worst_score:
                simplex[-1] = contracted
                simplex_scores[-1] = contracted_score
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = best + sigma * (simplex[i] - best)
                    simplex[i] = np.clip(simplex[i], 0, max_layer_nm)
                    if sum(simplex[i]) > max_total_nm:
                        scale = max_total_nm / sum(simplex[i])
                        simplex[i] = simplex[i] * scale
                        simplex[i] = np.clip(simplex[i], 0, max_layer_nm)
                    simplex_scores[i] = objective(simplex[i])
    
    # Return best point
    best_idx = np.argmin(simplex_scores)
    best_thick = list(simplex[best_idx])
    best_score = simplex_scores[best_idx]
    
    return materials, best_thick, best_score


def _try_layer_addition(materials, thicknesses, score, problem, deadline, rng):
    """Try adding layers at various positions."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = 1.38
    nH = 2.35
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    best_mats = materials[:]
    best_thick = thicknesses[:]
    best_score = score
    
    if len(materials) >= max_layers:
        return best_mats, best_thick, best_score
    
    # Try inserting layers at different positions
    for insert_pos in range(len(materials) + 1):
        if time.time() > deadline - 10:
            break
        
        for mat_char in ['L', 'H']:
            if mat_char == 'L':
                n_try = nL
            else:
                n_try = nH
            
            # Try several thicknesses
            for d_try in [centre_wl / (4.0 * n_try), centre_wl / (6.0 * n_try), centre_wl / (3.0 * n_try)]:
                if d_try > max_layer_nm or d_try < 0:
                    continue
                if sum(thicknesses) + d_try > max_total_nm:
                    continue
                
                new_mats = materials[:]
                new_thick = thicknesses[:]
                new_mats.insert(insert_pos, mat_char)
                new_thick.insert(insert_pos, d_try)
                
                if _is_valid(new_mats, new_thick, problem):
                    trial_score = _mean_reflectance_percent(new_mats, new_thick, problem)
                    if trial_score < best_score - 1e-12:
                        best_mats = new_mats
                        best_thick = new_thick
                        best_score = trial_score
    
    return best_mats, best_thick, best_score


def _try_layer_removal(materials, thicknesses, score, problem, deadline, rng):
    """Try removing layers."""
    best_mats = materials[:]
    best_thick = thicknesses[:]
    best_score = score
    
    if len(materials) <= 1:
        return best_mats, best_thick, best_score
    
    for remove_pos in range(len(materials)):
        if time.time() > deadline - 10:
            break
        
        new_mats = materials[:]
        new_thick = thicknesses[:]
        new_mats.pop(remove_pos)
        new_thick.pop(remove_pos)
        
        if _is_valid(new_mats, new_thick, problem):
            trial_score = _mean_reflectance_percent(new_mats, new_thick, problem)
            if trial_score < best_score - 1e-12:
                best_mats = new_mats
                best_thick = new_thick
                best_score = trial_score
    
    return best_mats, best_thick, best_score


def _generate_candidates(problem):
    """Generate diverse candidate designs."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = 1.38
    nH = 2.35
    band0, band1 = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    centre_wl = 0.5 * (band0 + band1)
    dL_qw = centre_wl / (4.0 * nL)
    dH_qw = centre_wl / (4.0 * nH)
    
    candidates = []
    
    # Single layer
    candidates.append((["L"], [dL_qw]))
    candidates.append((["H"], [dH_qw]))
    
    # Two layers
    candidates.append((["L", "H"], [dL_qw, dH_qw]))
    candidates.append((["H", "L"], [dH_qw, dL_qw]))
    
    # Three layers
    candidates.append((["L", "H", "L"], [dL_qw, dH_qw, dL_qw]))
    candidates.append((["H", "L", "H"], [dH_qw, dL_qw, dH_qw]))
    
    # Four layers
    candidates.append((["L", "H", "L", "H"], [dL_qw, dH_qw, dL_qw, dH_qw]))
    candidates.append((["H", "L", "H", "L"], [dH_qw, dL_qw, dH_qw, dL_qw]))
    candidates.append((["L", "H", "H", "L"], [dL_qw, dH_qw, dH_qw, dL_qw]))
    candidates.append((["H", "L", "L", "H"], [dH_qw, dL_qw, dL_qw, dH_qw]))
    
    # Five layers
    candidates.append((["L", "H", "L", "H", "L"], [dL_qw, dH_qw, dL_qw, dH_qw, dL_qw]))
    candidates.append((["H", "L", "H", "L", "H"], [dH_qw, dL_qw, dH_qw, dL_qw, dH_qw]))
    
    # Six layers
    candidates.append((["L", "H", "L", "H", "L", "H"], [dL_qw, dH_qw, dL_qw, dH_qw, dL_qw, dH_qw]))
    candidates.append((["H", "L", "H", "L", "H", "L"], [dH_qw, dL_qw, dH_qw, dL_qw, dH_qw, dL_qw]))
    
    # Seven layers
    candidates.append((["L", "H", "L", "H", "L", "H", "L"], [dL_qw, dH_qw, dL_qw, dH_qw, dL_qw, dH_qw, dL_qw]))
    candidates.append((["H", "L", "H", "L", "H", "L", "H"], [dH_qw, dL_qw, dH_qw, dL_qw, dH_qw, dL_qw, dH_qw]))
    
    # Eight layers
    candidates.append((["L", "H"] * 4, [dL_qw, dH_qw] * 4))
    candidates.append((["H", "L"] * 4, [dH_qw, dL_qw] * 4))
    
    # Nine layers
    candidates.append((["L", "H", "L", "H", "L", "H", "L", "H", "L"], [dL_qw, dH_qw, dL_qw, dH_qw, dL_qw, dH_qw, dL_qw, dH_qw, dL_qw]))
    
    # Ten layers
    candidates.append((["L", "H"] * 5, [dL_qw, dH_qw] * 5))
    
    # Non-quarter-wave variants for broadband
    for factor in [0.2, 0.22, 0.25, 0.28, 0.3]:
        dL_f = factor * centre_wl / nL
        dH_f = factor * centre_wl / nH
        if dL_f <= max_layer_nm and dH_f <= max_layer_nm:
            candidates.append((["L", "H", "L"], [dL_f, dH_f, dL_f]))
            candidates.append((["H", "L", "H"], [dH_f, dL_f, dH_f]))
            candidates.append((["L", "H", "L", "H"], [dL_f, dH_f, dL_f, dH_f]))
            candidates.append((["H", "L", "H", "L"], [dH_f, dL_f, dH_f, dL_f]))
    
    # Graded index approximations
    for n_layers in [3, 5, 7]:
        if n_layers % 2 == 1:
            # Symmetric L-H-L... pattern
            mats = []
            thicks = []
            for i in range(n_layers):
                if i % 2 == 0:
                    mats.append('L')
                    thicks.append(dL_qw)
                else:
                    mats.append('H')
                    thicks.append(dH_qw)
            candidates.append((mats, thicks))
            
            # H-L-H... pattern
            mats = []
            thicks = []
            for i in range(n_layers):
                if i % 2 == 0:
                    mats.append('H')
                    thicks.append(dH_qw)
                else:
                    mats.append('L')
                    thicks.append(dL_qw)
            candidates.append((mats, thicks))
    
    # Try designs with some layers at 0.5 * quarter-wave
    for n_pairs in [2, 3]:
        mats = []
        thicks = []
        for _ in range(n_pairs):
            mats.extend(['L', 'H'])
            thicks.extend([dL_qw * 0.8, dH_qw * 0.8])
        if sum(thicks) <= max_total_nm and len(mats) <= max_layers:
            candidates.append((mats, thicks))
    
    # Filter valid candidates
    valid_candidates = []
    for mats, thicks in candidates:
        if _is_valid(mats, thicks, problem):
            valid_candidates.append((mats, thicks))
    
    return valid_candidates


def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = 1.38
    nH = 2.35
    band0, band1 = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    import random
    rng = random.Random(42)
    
    # Generate candidates
    candidates = _generate_candidates(problem)
    
    if not candidates:
        # Fallback
        d1 = 0.5 * (band0 + band1) / (4.0 * nL)
        return ["L"], [d1]
    
    # Evaluate all candidates and find the best
    best_score = float('inf')
    best_mats = candidates[0][0][:]
    best_thick = candidates[0][1][:]
    
    for mats, thicks in candidates:
        if time.time() > deadline - 30:
            break
        score, valid = _evaluate_design(mats, thicks, problem)
        if valid and score < best_score:
            best_score = score
            best_mats = mats[:]
            best_thick = thicks[:]
    
    # Local optimization on the best candidate using coordinate descent
    if time.time() < deadline - 30:
        opt_mats, opt_thick, opt_score = _local_optimize(best_mats, best_thick, problem, deadline, rng)
        if opt_score < best_score:
            best_score = opt_score
            best_mats = opt_mats
            best_thick = opt_thick
    
    # Nelder-Mead optimization for more thorough local search
    if time.time() < deadline - 20:
        nm_mats, nm_thick, nm_score = _nelder_mead_optimize(best_mats, best_thick, problem, deadline, rng, max_iter=150)
        if nm_score < best_score:
            best_score = nm_score
            best_mats = nm_mats
            best_thick = nm_thick
    
    # Try adding layers to the optimized design
    if time.time() < deadline - 15:
        add_mats, add_thick, add_score = _try_layer_addition(best_mats, best_thick, best_score, problem, deadline, rng)
        if add_score < best_score:
            best_score = add_score
            best_mats = add_mats
            best_thick = add_thick
    
    # Try removing layers
    if time.time() < deadline - 10:
        rem_mats, rem_thick, rem_score = _try_layer_removal(best_mats, best_thick, best_score, problem, deadline, rng)
        if rem_score < best_score:
            best_score = rem_score
            best_mats = rem_mats
            best_thick = rem_thick
    
    # Final local optimization
    if time.time() < deadline - 10:
        final_mats, final_thick, final_score = _local_optimize(best_mats, best_thick, problem, deadline, rng)
        if final_score < best_score:
            best_score = final_score
            best_mats = final_mats
            best_thick = final_thick
    
    # Final Nelder-Mead polish
    if time.time() < deadline - 5:
        polish_mats, polish_thick, polish_score = _nelder_mead_optimize(best_mats, best_thick, problem, deadline, rng, max_iter=100)
        if polish_score < best_score:
            best_score = polish_score
            best_mats = polish_mats
            best_thick = polish_thick
    
    # Final validation
    if not _is_valid(best_mats, best_thick, problem):
        d1 = 0.5 * (band0 + band1) / (4.0 * nL)
        best_mats = ["L"]
        best_thick = [d1]
    
    return best_mats, best_thick

# EVOLVE-BLOCK-END
