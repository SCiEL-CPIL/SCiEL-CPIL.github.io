"""Design broadband antireflection coating using genetic algorithm + local optimization."""
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
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    current_mats = materials[:]
    current_thick = thicknesses[:]
    current_score = _mean_reflectance_percent(current_mats, current_thick, problem)
    
    if not _is_valid(current_mats, current_thick, problem):
        return materials, thicknesses, float('inf')
    
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
    """
    n_layers = len(thicknesses)
    if n_layers == 0:
        return materials, thicknesses, float('inf')
    
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    initial_thick = np.array(thicknesses, dtype=float)
    
    simplex = [initial_thick.copy()]
    for i in range(n_layers):
        perturbed = initial_thick.copy()
        perturbed[i] = initial_thick[i] + 10.0
        if perturbed[i] > max_layer_nm:
            perturbed[i] = max_layer_nm
        if perturbed[i] < 0:
            perturbed[i] = 0.0
        if sum(perturbed) <= max_total_nm:
            simplex.append(perturbed)
        else:
            perturbed[i] = initial_thick[i] + 5.0
            if perturbed[i] > max_layer_nm:
                perturbed[i] = max_layer_nm
            if sum(perturbed) <= max_total_nm:
                simplex.append(perturbed)
            else:
                for j in range(n_layers):
                    if j != i:
                        perturbed2 = initial_thick.copy()
                        perturbed2[j] = initial_thick[j] + 5.0
                        if 0 <= perturbed2[j] <= max_layer_nm and sum(perturbed2) <= max_total_nm:
                            simplex.append(perturbed2)
                            break
                else:
                    simplex.append(initial_thick.copy())
    
    def objective(thick_arr):
        if not _is_valid(materials, list(thick_arr), problem):
            return float('inf')
        return _mean_reflectance_percent(materials, list(thick_arr), problem)
    
    simplex_scores = [objective(pt) for pt in simplex]
    
    alpha = 0.5
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for iteration in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        sorted_pairs = sorted(zip(simplex_scores, simplex), key=lambda x: x[0])
        simplex = [pt for _, pt in sorted_pairs]
        simplex_scores = [s for s, _ in sorted_pairs]
        
        if abs(simplex_scores[0] - simplex_scores[-1]) < 1e-10:
            break
        
        worst = simplex[-1]
        worst_score = simplex_scores[-1]
        best = simplex[0]
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        reflected = centroid + alpha * (centroid - worst)
        reflected = np.clip(reflected, 0, max_layer_nm)
        if sum(reflected) > max_total_nm:
            scale = max_total_nm / sum(reflected)
            reflected = reflected * scale
            reflected = np.clip(reflected, 0, max_layer_nm)
        
        reflected_score = objective(reflected)
        
        if reflected_score < simplex_scores[0]:
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
            simplex[-1] = reflected
            simplex_scores[-1] = reflected_score
        else:
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
                for i in range(1, len(simplex)):
                    simplex[i] = best + sigma * (simplex[i] - best)
                    simplex[i] = np.clip(simplex[i], 0, max_layer_nm)
                    if sum(simplex[i]) > max_total_nm:
                        scale = max_total_nm / sum(simplex[i])
                        simplex[i] = simplex[i] * scale
                        simplex[i] = np.clip(simplex[i], 0, max_layer_nm)
                    simplex_scores[i] = objective(simplex[i])
    
    best_idx = np.argmin(simplex_scores)
    best_thick = list(simplex[best_idx])
    best_score = simplex_scores[best_idx]
    
    return materials, best_thick, best_score


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


def _genetic_algorithm(materials, thicknesses, problem, deadline, rng):
    """
    Genetic algorithm to explore the design space.
    Encodes a design as (n_layers, [mat_0, mat_1, ...], [t_0, t_1, ...])
    but we fix the number of layers and only optimize thicknesses and material choices.
    """
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band0, band1 = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    centre_wl = 0.5 * (band0 + band1)
    dL_qw = centre_wl / (4.0 * nL)
    dH_qw = centre_wl / (4.0 * nH)
    
    # Population size
    pop_size = 30
    generations = 20
    mutation_rate = 0.3
    crossover_rate = 0.7
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        # Random number of layers
        n_layers = rng.randint(2, min(10, max_layers))
        mats = []
        thicks = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append('L')
                thicks.append(rng.uniform(0.1 * dL_qw, 1.5 * dL_qw))
            else:
                mats.append('H')
                thicks.append(rng.uniform(0.1 * dH_qw, 1.5 * dH_qw))
        
        # Ensure validity
        if sum(thicks) > max_total_nm:
            scale = max_total_nm / sum(thicks)
            thicks = [t * scale for t in thicks]
        thicks = [min(t, max_layer_nm) for t in thicks]
        
        if _is_valid(mats, thicks, problem):
            population.append((mats[:], thicks[:]))
    
    # Also add the current best to the population
    if _is_valid(materials, thicknesses, problem):
        population.append((materials[:], thicknesses[:]))
    
    best_mats = materials[:]
    best_thick = thicknesses[:]
    best_score = _mean_reflectance_percent(best_mats, best_thick, problem)
    
    for gen in range(generations):
        if time.time() > deadline - 30:
            break
        
        # Evaluate population
        scored_pop = []
        for mats, thicks in population:
            if _is_valid(mats, thicks, problem):
                score = _mean_reflectance_percent(mats, thicks, problem)
            else:
                score = float('inf')
            scored_pop.append((score, mats, thicks))
        
        # Sort by score
        scored_pop.sort(key=lambda x: x[0])
        
        # Update best
        if scored_pop[0][0] < best_score:
            best_score = scored_pop[0][0]
            best_mats = scored_pop[0][1][:]
            best_thick = scored_pop[0][2][:]
        
        # Selection: keep top half
        elite = scored_pop[:pop_size // 2]
        
        # Generate next generation
        new_population = []
        
        # Keep elites
        for score, mats, thicks in elite:
            new_population.append((mats[:], thicks[:]))
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 30:
                break
            
            # Select parents (tournament selection)
            parent1 = rng.choice(elite)
            parent2 = rng.choice(elite)
            
            # Crossover
            if rng.random() < crossover_rate and len(parent1[1]) == len(parent2[1]):
                # Uniform crossover on thicknesses
                child_thicks = []
                child_mats = []
                for i in range(len(parent1[1])):
                    if rng.random() < 0.5:
                        child_thicks.append(parent1[2][i])
                        child_mats.append(parent1[1][i])
                    else:
                        child_thicks.append(parent2[2][i])
                        child_mats.append(parent2[1][i])
            else:
                child_mats = parent1[1][:]
                child_thicks = parent1[2][:]
            
            # Mutation
            for i in range(len(child_thicks)):
                if rng.random() < mutation_rate:
                    # Gaussian mutation
                    child_thicks[i] += rng.gauss(0, 20.0)
                    child_thicks[i] = max(0, min(child_thicks[i], max_layer_nm))
            
            # Ensure total thickness constraint
            if sum(child_thicks) > max_total_nm:
                scale = max_total_nm / sum(child_thicks)
                child_thicks = [t * scale for t in child_thicks]
            
            if _is_valid(child_mats, child_thicks, problem):
                new_population.append((child_mats, child_thicks))
        
        population = new_population
    
    return best_mats, best_thick, best_score


def _try_layer_addition(materials, thicknesses, score, problem, deadline, rng):
    """Try adding layers at various positions."""
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
    
    for insert_pos in range(len(materials) + 1):
        if time.time() > deadline - 10:
            break
        
        for mat_char in ['L', 'H']:
            if mat_char == 'L':
                n_try = nL
            else:
                n_try = nH
            
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


def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0
    
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
        d1 = 0.5 * (band0 + band1) / (4.0 * nL)
        return ["L"], [d1]
    
    # Evaluate all candidates and find the best
    best_score = float('inf')
    best_mats = candidates[0][0][:]
    best_thick = candidates[0][1][:]
    
    for mats, thicks in candidates:
        if time.time() > deadline - 40:
            break
        score, valid = _evaluate_design(mats, thicks, problem)
        if valid and score < best_score:
            best_score = score
            best_mats = mats[:]
            best_thick = thicks[:]
    
    # Local optimization on the best candidate
    if time.time() < deadline - 35:
        opt_mats, opt_thick, opt_score = _local_optimize(best_mats, best_thick, problem, deadline, rng)
        if opt_score < best_score:
            best_score = opt_score
            best_mats = opt_mats
            best_thick = opt_thick
    
    # Nelder-Mead optimization
    if time.time() < deadline - 25:
        nm_mats, nm_thick, nm_score = _nelder_mead_optimize(best_mats, best_thick, problem, deadline, rng, max_iter=150)
        if nm_score < best_score:
            best_score = nm_score
            best_mats = nm_mats
            best_thick = nm_thick
    
    # Genetic algorithm for global exploration
    if time.time() < deadline - 20:
        ga_mats, ga_thick, ga_score = _genetic_algorithm(best_mats, best_thick, problem, deadline, rng)
        if ga_score < best_score:
            best_score = ga_score
            best_mats = ga_mats
            best_thick = ga_thick
    
    # Try adding layers
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
