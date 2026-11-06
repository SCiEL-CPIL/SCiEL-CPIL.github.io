"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm):
    """Compute mean reflectance in percent over the band using characteristic matrix method."""
    if len(labels) == 0:
        r0 = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return r0 * r0 * 100.0
    
    n_L = materials['L']
    n_H = materials['H']
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    n_layers = len(labels)
    n_indices = np.empty(n_layers + 2, dtype=float)
    n_indices[0] = ambient_index
    n_indices[-1] = substrate_index
    for i, lab in enumerate(labels):
        if lab == 'L':
            n_indices[i + 1] = n_L
        else:
            n_indices[i + 1] = n_H
    
    d = np.array(thicknesses, dtype=float)
    
    reflectances = np.zeros(401, dtype=float)
    for idx, lam in enumerate(wavelengths):
        m11 = 1.0 + 0.0j
        m12 = 0.0 + 0.0j
        m21 = 0.0 + 0.0j
        m22 = 1.0 + 0.0j
        
        for j in range(n_layers):
            n_j = n_indices[j + 1]
            d_j = d[j]
            delta_j = 2.0 * np.pi * n_j * d_j / lam
            cos_d = np.cos(delta_j)
            sin_d = np.sin(delta_j)
            
            lm11 = cos_d
            lm12 = 1j * sin_d / n_j
            lm21 = 1j * n_j * sin_d
            lm22 = cos_d
            
            new_m11 = m11 * lm11 + m12 * lm21
            new_m12 = m11 * lm12 + m12 * lm22
            new_m21 = m21 * lm11 + m22 * lm21
            new_m22 = m21 * lm12 + m22 * lm22
            
            m11 = new_m11
            m12 = new_m12
            m21 = new_m21
            m22 = new_m22
        
        Y = (m21 + m22 * substrate_index) / (m11 + m12 * substrate_index)
        r = (ambient_index - Y) / (ambient_index + Y)
        reflectances[idx] = np.abs(r) ** 2
    
    return np.mean(reflectances) * 100.0


def _decode_individual(individual, thickness_options):
    """Decode a GA individual into labels and thicknesses."""
    n_layers = individual[0]
    labels = []
    thicknesses = []
    for i in range(1, 21):
        if i - 1 < n_layers:
            bit = individual[i]
            if bit == 0:
                labels.append('L')
            else:
                labels.append('H')
            t_idx = individual[i + 20]
            thicknesses.append(thickness_options[t_idx])
    
    return labels, thicknesses


def _is_valid(labels, thicknesses, max_layers, max_total_nm, max_layer_nm):
    """Check if a design satisfies all constraints."""
    if len(labels) == 0:
        return True
    if len(labels) > max_layers:
        return False
    if any(t < 0 or t > max_layer_nm for t in thicknesses):
        return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True


def _crossover(parent1, parent2):
    """Single-point crossover for GA individuals."""
    point = random.randint(1, 40)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def _mutate(individual, mutation_rate):
    """Bit-flip mutation for GA individuals."""
    individual = individual.copy()
    for i in range(1, 41):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


def design(problem):
    """Design an antireflection coating stack using genetic algorithm with discrete thicknesses."""
    start_time = time.time()
    deadline = start_time + 130.0  # 20 second safety margin
    
    materials = problem["materials"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    n_L = materials['L']
    n_H = materials['H']
    
    # Define discrete thickness options based on quarter-wave at various center wavelengths
    # Center wavelengths from 450 to 750 nm in 50 nm steps
    center_wavelengths = list(range(450, 751, 50))
    thickness_options = set()
    
    for cw in center_wavelengths:
        # Quarter-wave thicknesses
        thickness_options.add(round(cw / (4.0 * n_L), 2))
        thickness_options.add(round(cw / (4.0 * n_H), 2))
        # Half-wave thicknesses
        thickness_options.add(round(cw / (2.0 * n_L), 2))
        thickness_options.add(round(cw / (2.0 * n_H), 2))
        # Some additional values
        thickness_options.add(round(cw / (6.0 * n_L), 2))
        thickness_options.add(round(cw / (6.0 * n_H), 2))
    
    # Filter to valid range and sort
    thickness_options = sorted([t for t in thickness_options if 0 < t <= max_layer_nm])
    
    # Ensure we have a reasonable number of options
    if len(thickness_options) < 5:
        thickness_options = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]
    
    # Initial population
    pop_size = 100
    population = []
    
    # Add some known good designs to the population
    known_designs = [
        (["L"], [600 / (4.0 * n_L)]),
        (["L", "H"], [600 / (4.0 * n_L), 600 / (4.0 * n_H)]),
        (["H", "L", "H"], [600 / (4.0 * n_H), 600 / (4.0 * n_L), 600 / (4.0 * n_H)]),
        (["H", "L", "H", "L", "H"], [600 / (4.0 * n_H), 600 / (4.0 * n_L), 600 / (4.0 * n_H), 600 / (4.0 * n_L), 600 / (4.0 * n_H)]),
        (["H", "L", "H", "L", "H", "L", "H"], [600 / (4.0 * n_H), 600 / (4.0 * n_L), 600 / (4.0 * n_H), 600 / (4.0 * n_L), 600 / (4.0 * n_H), 600 / (4.0 * n_L), 600 / (4.0 * n_H)]),
    ]
    
    for labels, thicknesses in known_designs:
        if _is_valid(labels, thicknesses, max_layers, max_total_nm, max_layer_nm):
            individual = [len(labels)]
            for i in range(20):
                if i < len(labels):
                    individual.append(0 if labels[i] == 'L' else 1)
                else:
                    individual.append(0)
            for i in range(20):
                if i < len(thicknesses):
                    # Find closest thickness option
                    t = thicknesses[i]
                    min_dist = float('inf')
                    min_idx = 0
                    for idx, opt in enumerate(thickness_options):
                        dist = abs(opt - t)
                        if dist < min_dist:
                            min_dist = dist
                            min_idx = idx
                    individual.append(min_idx)
                else:
                    individual.append(0)
            population.append(individual)
    
    # Fill rest of population with random individuals
    while len(population) < pop_size:
        individual = [random.randint(1, max_layers)]
        for i in range(40):
            individual.append(random.randint(0, 1))
        population.append(individual)
    
    # Evaluate initial population
    best_individual = None
    best_score = float('inf')
    best_labels = None
    best_thicknesses = None
    
    for individual in population:
        labels, thicknesses = _decode_individual(individual, thickness_options)
        if not _is_valid(labels, thicknesses, max_layers, max_total_nm, max_layer_nm):
            continue
        score = _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm)
        if score < best_score:
            best_score = score
            best_individual = individual[:]
            best_labels = labels[:]
            best_thicknesses = thicknesses[:]
    
    # If no valid design found, use a simple fallback
    if best_labels is None:
        centre = 0.5 * (band_nm[0] + band_nm[1])
        best_labels = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
        best_score = _compute_reflectance(best_labels, best_thicknesses, materials, ambient_index, substrate_index, band_nm)
    
    # Genetic algorithm loop
    generations = 50
    elite_size = 5
    mutation_rate = 0.05
    crossover_rate = 0.8
    
    for gen in range(generations):
        if time.time() > deadline:
            break
        
        # Evaluate all individuals
        scored_population = []
        for individual in population:
            labels, thicknesses = _decode_individual(individual, thickness_options)
            if not _is_valid(labels, thicknesses, max_layers, max_total_nm, max_layer_nm):
                scored_population.append((float('inf'), individual))
            else:
                score = _compute_reflectance(labels, thicknesses, materials, ambient_index, substrate_index, band_nm)
                scored_population.append((score, individual))
        
        # Sort by score (ascending, lower is better)
        scored_population.sort(key=lambda x: x[0])
        
        # Update best
        if scored_population[0][0] < best_score:
            best_score = scored_population[0][0]
            best_individual = scored_population[0][1][:]
            best_labels, best_thicknesses = _decode_individual(best_individual, thickness_options)
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        for i in range(min(elite_size, len(scored_population))):
            new_population.append(scored_population[i][1][:])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Selection: tournament selection
            def tournament_select():
                tournament_size = 5
                candidates = random.sample(scored_population, tournament_size)
                candidates.sort(key=lambda x: x[0])
                return candidates[0][1]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            if random.random() < crossover_rate:
                child1, child2 = _crossover(parent1, parent2)
            else:
                child1 = parent1[:]
                child2 = parent2[:]
            
            child1 = _mutate(child1, mutation_rate)
            child2 = _mutate(child2, mutation_rate)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        population = new_population
    
    # Final evaluation of best individual
    if best_individual is not None:
        final_labels, final_thicknesses = _decode_individual(best_individual, thickness_options)
        if _is_valid(final_labels, final_thicknesses, max_layers, max_total_nm, max_layer_nm):
            final_score = _compute_reflectance(final_labels, final_thicknesses, materials, ambient_index, substrate_index, band_nm)
            if final_score < best_score:
                best_score = final_score
                best_labels = final_labels
                best_thicknesses = final_thicknesses
    
    # Final constraint check
    if best_labels is None or not _is_valid(best_labels, best_thicknesses, max_layers, max_total_nm, max_layer_nm):
        centre = 0.5 * (band_nm[0] + band_nm[1])
        best_labels = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
