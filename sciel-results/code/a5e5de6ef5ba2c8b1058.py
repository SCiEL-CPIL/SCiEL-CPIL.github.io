"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _reflectance_spectrum(materials, thicknesses, n_layers, ambient_index, substrate_index, wavelengths):
    """
    Compute reflectance spectrum using characteristic-matrix method (optimized).
    Returns array of reflectance values (0 to 1) for each wavelength.
    """
    n_w = len(wavelengths)
    
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return np.full(n_w, r ** 2)
    
    # Precompute optical thicknesses: n_j * d_j for each layer
    optical_thick = np.array(materials, dtype=float) * np.array(thicknesses, dtype=float)
    
    # delta[j, i] = 2*pi*optical_thick[j]/wavelengths[i]  shape (n_layers, n_w)
    delta = 2.0 * np.pi * optical_thick[:, None] / wavelengths[None, :]
    
    cos_d = np.cos(delta)  # (n_layers, n_w)
    sin_d = np.sin(delta)  # (n_layers, n_w)
    
    # Initialize the running product as identity for each wavelength
    M00 = np.ones(n_w, dtype=complex)
    M01 = np.zeros(n_w, dtype=complex)
    M10 = np.zeros(n_w, dtype=complex)
    M11 = np.ones(n_w, dtype=complex)
    
    for j in range(n_layers):
        c = cos_d[j, :]
        s = sin_d[j, :]
        n_j = materials[j]
        
        # New M = M_old @ M_j
        # M_j = [[c, 1j*s/n_j], [1j*n_j*s, c]]
        new_M00 = M00 * c + M01 * (1j * s / n_j)
        new_M01 = M00 * (1j * s / n_j) + M01 * c
        new_M10 = M10 * c + M11 * (1j * n_j * s)
        new_M11 = M10 * (1j * n_j * s) + M11 * c
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    # Now compute B and C:
    # [B, C]^T = M @ [1, n_s]^T
    B = M00 * 1.0 + M01 * substrate_index
    C = M10 * 1.0 + M11 * substrate_index
    
    # Reflectance:
    # r = (n_0 * B - C) / (n_0 * B + C)
    # R = |r|^2
    r = (ambient_index * B - C) / (ambient_index * B + C)
    R = np.abs(r) ** 2
    
    return R

def _evaluate_discrete_genome(genome, n_L, n_H, ambient_index, substrate_index, 
                                wavelengths, max_total_nm, max_layer_nm):
    """
    Evaluate a discrete genome: (layer_sequence, thickness_indices)
    layer_sequence: list of 0s (L) and 1s (H)
    thickness_indices: list of indices into thickness_steps
    Returns mean reflectance in percent, or penalty if constraints violated.
    """
    n_layers = len(layer_sequence)
    if n_layers == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return r ** 2 * 100.0
    
    materials = [n_L if l == 0 else n_H for l in layer_sequence]
    thicknesses = [thickness_steps[idx] for idx, thickness_steps in zip(thickness_indices, thickness_steps_list)]
    
    # Check constraints
    if sum(thicknesses) > max_total_nm:
        return 100.0
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return 100.0
    
    R = _reflectance_spectrum(materials, thicknesses, n_layers,
                               ambient_index, substrate_index, wavelengths)
    return np.mean(R) * 100.0

def _ga_discrete_search(n_layers, n_L, n_H, ambient_index, substrate_index,
                         wavelengths, max_total_nm, max_layer_nm, 
                         pop_size=50, n_generations=200, deadline=None):
    """
    Genetic algorithm over discrete layer sequences and thickness indices.
    """
    # Define discrete thickness steps (quarter-wave optical thicknesses)
    # Optical thickness ~ lambda/4, so physical thickness = lambda/(4*n)
    # For broadband 400-800nm, use steps around 50-150nm
    thickness_steps = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400]
    n_steps = len(thickness_steps)
    
    def decode_and_evaluate(genome):
        """Decode genome and evaluate fitness."""
        layer_seq = genome[:n_layers]
        thick_idx = genome[n_layers:]
        
        materials = [n_L if l == 0 else n_H for l in layer_seq]
        thicknesses = [thickness_steps[idx] for idx in thick_idx]
        
        # Check constraints
        if sum(thicknesses) > max_total_nm:
            return 100.0
        for t in thicknesses:
            if t < 0 or t > max_layer_nm:
                return 100.0
        
        R = _reflectance_spectrum(materials, thicknesses, n_layers,
                                   ambient_index, substrate_index, wavelengths)
        return np.mean(R) * 100.0
    
    def crossover(parent1, parent2):
        """Uniform crossover."""
        child = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child
    
    def mutate(genome, mutation_rate=0.1):
        """Point mutation."""
        child = genome[:]
        for i in range(len(child)):
            if random.random() < mutation_rate:
                if i < n_layers:
                    # Mutate layer type
                    child[i] = 1 - child[i]
                else:
                    # Mutate thickness index
                    child[i] = random.randint(0, n_steps - 1)
        return child
    
    def tournament_selection(population, fitnesses, k=3):
        """Tournament selection."""
        indices = random.sample(range(len(population)), k)
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx][:]
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        genome = []
        # Random layer sequence
        for _ in range(n_layers):
            genome.append(random.randint(0, 1))
        # Random thickness indices
        for _ in range(n_layers):
            genome.append(random.randint(0, n_steps - 1))
        population.append(genome)
    
    # Evaluate initial population
    fitnesses = [decode_and_evaluate(g) for g in population]
    
    best_genome = population[np.argmin(fitnesses)][:]
    best_fitness = min(fitnesses)
    
    for gen in range(n_generations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Create next generation
        new_population = []
        
        # Elitism: keep top 10%
        elite_count = max(1, pop_size // 10)
        elite_indices = np.argsort(fitnesses)[:elite_count]
        for idx in elite_indices:
            new_population.append(population[idx][:])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate=0.15)
            new_population.append(child)
        
        population = new_population
        fitnesses = [decode_and_evaluate(g) for g in population]
        
        # Update best
        if min(fitnesses) < best_fitness:
            best_fitness = min(fitnesses)
            best_genome = population[np.argmin(fitnesses)][:]
    
    # Decode best genome
    layer_seq = best_genome[:n_layers]
    thick_idx = best_genome[n_layers:]
    materials = [n_L if l == 0 else n_H for l in layer_seq]
    thicknesses = [thickness_steps[idx] for idx in thick_idx]
    
    # Final constraint check
    if sum(thicknesses) > max_total_nm:
        # Scale down proportionally
        scale = max_total_nm / sum(thicknesses)
        thicknesses = [t * scale for t in thicknesses]
    
    thicknesses = [min(max(t, 0.0), max_layer_nm) for t in thicknesses]
    
    return materials, thicknesses, best_fitness

def design(problem):
    """
    Design a broadband antireflection coating using discrete genetic algorithm
    over layer sequences and discrete thickness indices.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    materials_dict = problem["materials"]
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    band = problem["band_nm"]
    
    # Precompute wavelength grid for fast evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Track best solution
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Try different layer counts with GA
    # Focus on 10-20 layers where better solutions typically exist
    layer_counts = list(range(10, min(max_layers + 1, 21)))
    layer_counts += list(range(9, 5, -1))  # Fallback to lower counts
    
    for n_layers in layer_counts:
        if time.time() > deadline:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 5.0:
            break
        
        # Allocate generations based on remaining time
        # Each generation evaluates ~50 individuals, each evaluation ~1-2ms
        # So ~50-100ms per generation, allowing ~50-200 generations in 5-20s
        n_generations = min(150, int(remaining_time * 10))
        pop_size = 40
        
        if n_generations < 20:
            break
        
        try:
            materials, thicknesses, score = _ga_discrete_search(
                n_layers, n_L, n_H, ambient_index, substrate_index,
                wavelengths, max_total_nm, max_layer_nm,
                pop_size=pop_size, n_generations=n_generations, deadline=deadline
            )
        except Exception:
            continue
        
        if score < best_score:
            best_score = score
            best_materials = materials[:]
            best_thicknesses = thicknesses[:]
    
    # If no good solution found, use a simple quarter-wave stack as fallback
    if best_materials is None:
        centre = 0.5 * (band[0] + band[1])
        n_layers_fallback = min(12, max_layers)
        best_materials = []
        best_thicknesses = []
        for i in range(n_layers_fallback):
            if i % 2 == 0:
                best_materials.append(n_L)
                best_thicknesses.append(centre / (4.0 * n_L))
            else:
                best_materials.append(n_H)
                best_thicknesses.append(centre / (4.0 * n_H))
        
        # Scale to fit constraint
        total = sum(best_thicknesses)
        if total > max_total_nm:
            scale = max_total_nm / total
            best_thicknesses = [t * scale for t in best_thicknesses]
        best_thicknesses = [min(max(t, 0.0), max_layer_nm) for t in best_thicknesses]
    
    # Convert materials from indices back to labels
    material_labels = []
    for n in best_materials:
        if abs(n - n_L) < 1e-6:
            material_labels.append("L")
        else:
            material_labels.append("H")
    
    # Final constraint check
    total_thickness = sum(best_thicknesses)
    if total_thickness > max_total_nm:
        scale = max_total_nm / total_thickness
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    best_thicknesses = [min(max(t, 0.0), max_layer_nm) for t in best_thicknesses]
    
    return material_labels, best_thicknesses
# EVOLVE-BLOCK-END
