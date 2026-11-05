"""Rewritten program: genetic-algorithm search over alternating H/L stacks."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _char_matrix(n, d, w):
    """Return 2x2 characteristic matrix for a layer of index n, thickness d (nm), wavelength w (nm)."""
    beta = 2.0 * np.pi * n * d / w
    c = np.cos(beta)
    s = np.sin(beta)
    return np.array([[c, 1j * s / n], [1j * n * s, c]], dtype=complex)

def _reflectance_spectrum(materials, thicknesses, band_nm, ambient_index, substrate_index, n_waves=401):
    """Compute mean reflectance (%) over the band using characteristic-matrix method."""
    w0, w1 = band_nm
    wavelengths = np.linspace(w0, w1, n_waves)
    n_layers = len(materials)
    if n_layers == 0:
        # bare substrate
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return 100.0 * np.abs(r) ** 2

    total = 0.0
    for w in wavelengths:
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        for i in range(n_layers):
            n = materials[materials[i]] if isinstance(materials, dict) else materials[i]
            d = thicknesses[i]
            Mi = _char_matrix(n, d, w)
            M = M @ Mi
        # substrate matrix
        Ms = np.array([[1.0, 0.0], [0.0, substrate_index]], dtype=complex)
        M = M @ Ms
        Y = M[1, 1] / M[0, 1]  # admittance at air side
        r = (ambient_index - Y) / (ambient_index + Y)
        total += np.abs(r) ** 2
    return 100.0 * total / n_waves

def _evaluate(materials_dict, labels, thicknesses, problem):
    """Return mean reflectance in percent."""
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    band_nm = problem["band_nm"]
    # Build list of indices
    idx_list = [materials_dict[l] for l in labels]
    return _reflectance_spectrum(idx_list, thicknesses, band_nm, ambient_index, substrate_index)

def _valid(labels, thicknesses, problem):
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    if len(labels) > max_layers:
        return False
    if any(t < 0 or t > max_layer_nm for t in thicknesses):
        return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True

def _random_individual(problem):
    """Generate a random alternating stack."""
    max_layers = problem["max_layers"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    n = random.randint(1, max_layers)
    # Start with L or H randomly
    start = random.choice(['L', 'H'])
    labels = []
    thicknesses = []
    remaining = max_total_nm
    for i in range(n):
        if i == 0:
            lab = start
        else:
            lab = 'H' if labels[-1] == 'L' else 'L'
        labels.append(lab)
        # random thickness
        t = random.uniform(0.0, max_layer_nm)
        if t > remaining:
            t = remaining if remaining > 0 else 0.0
        thicknesses.append(t)
        remaining -= t
    return labels, thicknesses

def _crossover(parent1, parent2, problem):
    """Uniform crossover on alternating stacks."""
    n1 = len(parent1[0])
    n2 = len(parent2[0])
    n = random.randint(1, max(n1, n2, 1))
    labels = []
    thicknesses = []
    start = random.choice(['L', 'H'])
    for i in range(n):
        if i == 0:
            lab = start
        else:
            lab = 'H' if labels[-1] == 'L' else 'L'
        labels.append(lab)
        # pick thickness from one parent if available, else random
        if i < n1 and random.random() < 0.5:
            t = parent1[1][i]
        elif i < n2:
            t = parent2[1][i]
        else:
            t = random.uniform(0.0, problem["max_layer_nm"])
        thicknesses.append(t)
    return labels, thicknesses

def _mutate(labels, thicknesses, problem):
    """Mutate: change a thickness, add or remove a layer."""
    max_layers = problem["max_layers"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    action = random.random()
    if action < 0.5:
        # Mutate one thickness
        if thicknesses:
            idx = random.randint(0, len(thicknesses) - 1)
            thicknesses[idx] = random.uniform(0.0, max_layer_nm)
    elif action < 0.75 and len(labels) < max_layers:
        # Add a layer
        lab = 'H' if labels[-1] == 'L' else 'L' if labels else random.choice(['L', 'H'])
        labels.append(lab)
        t = random.uniform(0.0, max_layer_nm)
        thicknesses.append(t)
    elif action < 0.9 and len(labels) > 1:
        # Remove a layer
        idx = random.randint(0, len(labels) - 1)
        labels.pop(idx)
        thicknesses.pop(idx)
    else:
        # Random restart of one layer
        if thicknesses:
            idx = random.randint(0, len(thicknesses) - 1)
            thicknesses[idx] = random.uniform(0.0, max_layer_nm)
    
    # Clamp total thickness
    total = sum(thicknesses)
    if total > max_total_nm and thicknesses:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
    
    return labels, thicknesses

def design(problem):
    """Genetic algorithm to find low-reflectance coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    materials = problem["materials"]
    max_layers = problem["max_layers"]
    
    # Initialize population
    pop_size = 60
    population = []
    scores = []
    
    for _ in range(pop_size):
        labels, thicknesses = _random_individual(problem)
        if not _valid(labels, thicknesses, problem):
            # Fix validity
            if sum(thicknesses) > problem["max_total_nm"]:
                scale = problem["max_total_nm"] / sum(thicknesses)
                thicknesses = [t * scale for t in thicknesses]
        score = _evaluate(materials, labels, thicknesses, problem)
        population.append((labels, thicknesses))
        scores.append(score)
    
    best_idx = np.argmin(scores)
    best_labels = population[best_idx][0][:]
    best_thicknesses = population[best_idx][1][:]
    best_score = scores[best_idx]
    
    # Elite preservation
    elite_size = 5
    generations = 0
    
    while time.time() < deadline:
        generations += 1
        new_population = []
        new_scores = []
        
        # Keep elites
        sorted_indices = np.argsort(scores)
        for i in range(min(elite_size, len(population))):
            idx = sorted_indices[i]
            new_population.append((population[idx][0][:], population[idx][1][:]))
            new_scores.append(scores[idx])
        
        # Generate rest via crossover and mutation
        while len(new_population) < pop_size and time.time() < deadline:
            # Pick parents by tournament
            p1_idx = random.randint(0, len(population) - 1)
            p2_idx = random.randint(0, len(population) - 1)
            
            child_labels, child_thicknesses = _crossover(population[p1_idx], population[p2_idx], problem)
            
            if random.random() < 0.4:
                child_labels, child_thicknesses = _mutate(child_labels, child_thicknesses, problem)
            
            if not _valid(child_labels, child_thicknesses, problem):
                if sum(child_thicknesses) > problem["max_total_nm"] and child_thicknesses:
                    scale = problem["max_total_nm"] / sum(child_thicknesses)
                    child_thicknesses = [t * scale for t in child_thicknesses]
                if len(child_labels) > max_layers:
                    child_labels = child_labels[:max_layers]
                    child_thicknesses = child_thicknesses[:max_layers]
            
            score = _evaluate(materials, child_labels, child_thicknesses, problem)
            new_population.append((child_labels, child_thicknesses))
            new_scores.append(score)
        
        population = new_population
        scores = new_scores
        
        # Update best
        best_idx = np.argmin(scores)
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_labels = population[best_idx][0][:]
            best_thicknesses = population[best_idx][1][:]
        
        # Adaptive mutation rate increase if stagnating
        if generations % 10 == 0 and random.random() < 0.3:
            # Inject some random individuals
            for _ in range(3):
                labels, thicknesses = _random_individual(problem)
                if sum(thicknesses) > problem["max_total_nm"] and thicknesses:
                    scale = problem["max_total_nm"] / sum(thicknesses)
                    thicknesses = [t * scale for t in thicknesses]
                score = _evaluate(materials, labels, thicknesses, problem)
                population.append((labels, thicknesses))
                scores.append(score)
    
    # Final local refinement on best
    if time.time() < deadline - 5.0:
        for _ in range(50):
            if time.time() >= deadline - 3.0:
                break
            labels = best_labels[:]
            thicknesses = best_thicknesses[:]
            if thicknesses:
                idx = random.randint(0, len(thicknesses) - 1)
                old_t = thicknesses[idx]
                thicknesses[idx] = old_t + random.gauss(0, 10.0)
                thicknesses[idx] = max(0.0, min(thicknesses[idx], problem["max_layer_nm"]))
            if _valid(labels, thicknesses, problem):
                score = _evaluate(materials, labels, thicknesses, problem)
                if score < best_score:
                    best_score = score
                    best_labels = labels[:]
                    best_thicknesses = thicknesses[:]
    
    return best_labels, best_thicknesses
# EVOLVE-BLOCK-END
