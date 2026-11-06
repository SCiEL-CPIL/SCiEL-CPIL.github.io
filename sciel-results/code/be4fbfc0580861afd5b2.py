"""
Broadband antireflection coating design via discrete genetic search.

Strategy: Use a genetic algorithm over discrete thickness values and layer
compositions. The search space is constrained to quarter-wave multiples at
several reference wavelengths, which are known to be near-optimal for
broadband antireflection coatings.
"""
import numpy as np
import time
import random

def _compute_reflectance(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength using the characteristic matrix method.
    """
    n_layers = len(indices)
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r0 * r0)
    
    k = 2.0 * np.pi / wavelengths_nm
    
    M00 = np.ones(len(wavelengths_nm), dtype=complex)
    M01 = np.zeros(len(wavelengths_nm), dtype=complex)
    M10 = np.zeros(len(wavelengths_nm), dtype=complex)
    M11 = np.ones(len(wavelengths_nm), dtype=complex)
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        B00 = cos_d
        B01 = 1j * sin_d / n_i
        B10 = 1j * n_i * sin_d
        B11 = cos_d
        
        new_M00 = M00 * B00 + M01 * B10
        new_M01 = M00 * B01 + M01 * B11
        new_M10 = M10 * B00 + M11 * B10
        new_M11 = M10 * B01 + M11 * B11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    return R

def _objective(labels, thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx, 
               max_layer_nm, max_total_nm):
    """
    Objective: mean reflectance in percent. Returns large penalty if constraints violated.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    if len(thicknesses) == 0:
        return 100.0
    if np.any(thicknesses < -1e-6) or np.any(thicknesses > max_layer_nm + 1e-6):
        return 100.0
    if np.sum(thicknesses) > max_total_nm + 1e-6:
        return 100.0
    
    thicknesses_clipped = np.clip(thicknesses, 0.0, max_layer_nm)
    if np.sum(thicknesses_clipped) > max_total_nm:
        return 100.0
    
    R = _compute_reflectance(wavelengths_nm, indices, thicknesses_clipped, 
                             ambient_idx, substrate_idx)
    return np.mean(R) * 100.0

def _generate_quarter_wave_thickesses(materials, band_nm, max_layer_nm, max_total_nm):
    """
    Generate a set of candidate thicknesses based on quarter-wave multiples
    at several reference wavelengths.
    """
    ref_wavelengths = [450, 500, 550, 600, 650, 700]
    candidates = set()
    
    for wl in ref_wavelengths:
        for mat_name, n in materials.items():
            qwt = wl / (4.0 * n)
            # Add multiples: 1x, 2x, 3x quarter-wave
            for mult in [1, 2, 3]:
                t = qwt * mult
                if 0 <= t <= max_layer_nm:
                    candidates.add(round(t, 1))
    
    # Also add some finer-grained values
    for wl in [500, 600]:
        for mat_name, n in materials.items():
            qwt = wl / (4.0 * n)
            for frac in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
                t = qwt * frac
                if 0 <= t <= max_layer_nm:
                    candidates.add(round(t, 1))
    
    # Ensure we have a reasonable set
    candidates = sorted(list(candidates))
    
    # If too many, subsample
    if len(candidates) > 30:
        step = max(1, len(candidates) // 30)
        candidates = candidates[::step][:30]
    
    # Always include some small and large values
    candidates = sorted(set(candidates + [10, 20, 30, 50, 100, 150, 200, 250, 300, 350]))
    candidates = [c for c in candidates if 0 <= c <= max_layer_nm]
    
    return candidates

def _random_individual(materials, max_layers, max_total_nm, max_layer_nm, thickness_candidates):
    """
    Generate a random individual: random number of layers (1 to max_layers),
    random L/H labels, random thicknesses from candidate set.
    """
    n_layers = random.randint(1, max_layers)
    labels = [random.choice(['L', 'H']) for _ in range(n_layers)]
    indices = [materials[label] for label in labels]
    
    # Random thicknesses from candidates, ensuring total <= max_total_nm
    thicknesses = []
    remaining = max_total_nm
    for i in range(n_layers):
        if remaining <= 0:
            thicknesses.append(0.0)
            continue
        # Choose a random thickness that fits
        valid_candidates = [t for t in thickness_candidates if t <= remaining]
        if not valid_candidates:
            thicknesses.append(0.0)
            continue
        t = random.choice(valid_candidates)
        thicknesses.append(t)
        remaining -= t
    
    return labels, thicknesses

def _crossover(ind1, ind2, materials, max_layers, max_total_nm, max_layer_nm, thickness_candidates):
    """
    Simple crossover: take a prefix from one parent and suffix from the other,
    then repair to satisfy constraints.
    """
    labels1, th1 = ind1
    labels2, th2 = ind2
    
    if len(labels1) == 0:
        return ind2
    if len(labels2) == 0:
        return ind1
    
    # Random crossover point
    point = random.randint(1, min(len(labels1), len(labels2)) - 1)
    
    new_labels = labels1[:point] + labels2[point:]
    new_th = th1[:point] + th2[point:]
    
    # Truncate to max_layers
    if len(new_labels) > max_layers:
        new_labels = new_labels[:max_layers]
        new_th = new_th[:max_layers]
    
    # Repair: ensure total thickness <= max_total_nm
    total = sum(new_th)
    if total > max_total_nm:
        scale = max_total_nm / total
        new_th = [t * scale for t in new_th]
        # Re-quantize to nearest candidate
        for i in range(len(new_th)):
            nearest = min(thickness_candidates, key=lambda c: abs(c - new_th[i]))
            new_th[i] = nearest
    
    # Ensure individual layer constraints
    for i in range(len(new_th)):
        if new_th[i] > max_layer_nm:
            new_th[i] = max_layer_nm
    
    # Final check: if still over, zero out some layers
    total = sum(new_th)
    while total > max_total_nm and len(new_th) > 1:
        # Remove the thickest layer
        max_idx = np.argmax(new_th)
        new_th[max_idx] = 0.0
        total -= new_th[max_idx]
    
    return new_labels, new_th

def _mutate(ind, materials, max_layers, max_total_nm, max_layer_nm, thickness_candidates, mutation_rate=0.3):
    """
    Mutate an individual with some probability.
    """
    labels, thicknesses = ind
    labels = list(labels)
    thicknesses = list(thicknesses)
    
    # Maybe change a layer's index
    if labels and random.random() < mutation_rate:
        idx = random.randint(0, len(labels) - 1)
        labels[idx] = random.choice(['L', 'H'])
    
    # Maybe change a thickness
    if thicknesses and random.random() < mutation_rate:
        idx = random.randint(0, len(thicknesses) - 1)
        thicknesses[idx] = random.choice(thickness_candidates)
    
    # Maybe add a layer
    if len(labels) < max_layers and random.random() < mutation_rate * 0.5:
        labels.append(random.choice(['L', 'H']))
        thicknesses.append(random.choice(thickness_candidates))
    
    # Maybe remove a layer
    if len(labels) > 1 and random.random() < mutation_rate * 0.5:
        idx = random.randint(0, len(labels) - 1)
        labels.pop(idx)
        thicknesses.pop(idx)
    
    # Repair constraints
    total = sum(thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
        for i in range(len(thicknesses)):
            nearest = min(thickness_candidates, key=lambda c: abs(c - thicknesses[i]))
            thicknesses[i] = nearest
    
    for i in range(len(thicknesses)):
        if thicknesses[i] > max_layer_nm:
            thicknesses[i] = max_layer_nm
    
    return labels, thicknesses

def _genetic_search(problem, deadline):
    """
    Run a genetic algorithm to find a good AR coating design.
    """
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Generate candidate thicknesses
    thickness_candidates = _generate_quarter_wave_thickesses(materials, band_nm, max_layer_nm, max_total_nm)
    
    # Initialize population
    pop_size = 50
    population = []
    for _ in range(pop_size):
        ind = _random_individual(materials, max_layers, max_total_nm, max_layer_nm, thickness_candidates)
        population.append(ind)
    
    # Evaluate initial population
    scores = []
    for ind in population:
        labels, thicknesses = ind
        indices = [materials[label] for label in labels]
        score = _objective(labels, thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx,
                          max_layer_nm, max_total_nm)
        scores.append(score)
    
    # Find initial best
    best_idx = np.argmin(scores)
    best_ind = population[best_idx]
    best_score = scores[best_idx]
    
    # Genetic algorithm loop
    start_time = time.time()
    generation = 0
    no_improve_generations = 0
    max_no_improve = 15
    
    while time.time() < deadline - 10:
        generation += 1
        no_improve_generations += 1
        
        # Sort population by score
        sorted_indices = np.argsort(scores)
        population = [population[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # Elitism: keep top 10%
        elite_count = max(1, pop_size // 10)
        new_population = population[:elite_count]
        
        # Generate new individuals
        while len(new_population) < pop_size:
            if random.random() < 0.7:
                # Crossover
                p1 = random.choice(population[:pop_size // 2])
                p2 = random.choice(population[:pop_size // 2])
                child = _crossover(p1, p2, materials, max_layers, max_total_nm, max_layer_nm, thickness_candidates)
            else:
                # Mutation of a random individual
                parent = random.choice(population)
                child = _mutate(parent, materials, max_layers, max_total_nm, max_layer_nm, thickness_candidates)
            
            new_population.append(child)
        
        # Evaluate new population
        new_scores = []
        for ind in new_population:
            labels, thicknesses = ind
            indices = [materials[label] for label in labels]
            score = _objective(labels, thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx,
                              max_layer_nm, max_total_nm)
            new_scores.append(score)
        
        # Combine and select
        all_inds = new_population
        all_scores = new_scores
        
        # Update best
        best_idx_new = np.argmin(all_scores)
        if all_scores[best_idx_new] < best_score:
            best_ind = all_inds[best_idx_new]
            best_score = all_scores[best_idx_new]
            no_improve_generations = 0
        
        # Prepare next generation: keep best individuals
        sorted_new = np.argsort(all_scores)
        population = [all_inds[i] for i in sorted_new]
        scores = [all_scores[i] for i in sorted_new]
        
        # Early stop if no improvement for many generations
        if no_improve_generations >= max_no_improve:
            break
    
    return best_ind, best_score

def _local_refine(labels, thicknesses, problem, deadline):
    """
    Local refinement using coordinate descent on the best individual.
    """
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    indices = [materials[label] for label in labels]
    n_layers = len(labels)
    
    x = np.array(thicknesses, dtype=float)
    f_current = _objective(labels, x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                          max_layer_nm, max_total_nm)
    
    step_sizes = [20, 10, 5, 2, 1, 0.5, 0.2, 0.1]
    
    for step in step_sizes:
        if time.time() > deadline - 5:
            break
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(n_layers):
                # Try increasing
                x_trial = x.copy()
                x_trial[i] += step
                if x_trial[i] <= max_layer_nm and np.sum(x_trial) <= max_total_nm:
                    f_trial = _objective(labels, x_trial, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                        max_layer_nm, max_total_nm)
                    if f_trial < f_current:
                        x = x_trial
                        f_current = f_trial
                        improved = True
                
                # Try decreasing
                x_trial = x.copy()
                x_trial[i] -= step
                if x_trial[i] >= 0:
                    f_trial = _objective(labels, x_trial, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                        max_layer_nm, max_total_nm)
                    if f_trial < f_current:
                        x = x_trial
                        f_current = f_trial
                        improved = True
    
    return labels, x.tolist(), f_current

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Run genetic search
    best_ind, best_score = _genetic_search(problem, deadline)
    labels, thicknesses = best_ind
    
    # Local refinement if time permits
    if time.time() < deadline - 20:
        labels, thicknesses, refined_score = _local_refine(labels, thicknesses, problem, deadline)
        if refined_score < best_score:
            best_score = refined_score
    
    # Final constraint check
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layers = problem.get("max_layers", 20)
    
    thicknesses = np.array(thicknesses, dtype=float)
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    
    # Ensure total thickness constraint
    if np.sum(thicknesses) > max_total_nm:
        scale = max_total_nm / np.sum(thicknesses)
        thicknesses = thicknesses * scale
        # Re-clip
        thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    
    # Ensure layer count constraint
    if len(labels) > max_layers:
        labels = labels[:max_layers]
        thicknesses = thicknesses[:max_layers]
    
    # Remove zero-thickness layers
    nonzero_mask = thicknesses > 1e-6
    labels = [labels[i] for i in range(len(labels)) if nonzero_mask[i]]
    thicknesses = thicknesses[nonzero_mask]
    
    if len(labels) == 0:
        # Fallback: single quarter-wave layer
        materials = problem["materials"]
        band_nm = problem["band_nm"]
        center_wl = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    return labels, thicknesses.tolist()
