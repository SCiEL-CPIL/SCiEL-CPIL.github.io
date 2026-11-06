"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _reflectance_spectrum(materials, thicknesses, n_layers, ambient_index, substrate_index, wavelengths):
    """
    Compute reflectance spectrum using characteristic-matrix method.
    Returns array of reflectance values (0 to 1) for each wavelength.
    """
    n_w = len(wavelengths)
    if n_layers == 0:
        R = np.full(n_w, ((ambient_index - substrate_index) / (ambient_index + substrate_index)) ** 2)
        return R
    
    # delta[i, j] = 2*pi*materials[j]*thicknesses[j]/wavelengths[i]
    delta = 2.0 * np.pi * np.outer(1.0 / wavelengths, materials * thicknesses)
    # delta shape: (n_w, n_layers)
    
    cos_d = np.cos(delta)  # (n_w, n_layers)
    sin_d = np.sin(delta)  # (n_w, n_layers)
    
    # Initialize running product as identity for each wavelength
    M00 = np.ones(n_w, dtype=complex)
    M01 = np.zeros(n_w, dtype=complex)
    M10 = np.zeros(n_w, dtype=complex)
    M11 = np.ones(n_w, dtype=complex)
    
    for j in range(n_layers):
        c = cos_d[:, j]
        s = sin_d[:, j]
        n_j = materials[j]
        
        new_M00 = M00 * c + M01 * (1j * s / n_j)
        new_M01 = M00 * (1j * s / n_j) + M01 * c
        new_M10 = M10 * c + M11 * (1j * n_j * s)
        new_M11 = M10 * (1j * n_j * s) + M11 * c
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    B = M00 * 1.0 + M01 * substrate_index
    C = M10 * 1.0 + M11 * substrate_index
    
    r = (ambient_index * B - C) / (ambient_index * B + C)
    R = np.abs(r) ** 2
    
    return R

def _mean_reflectance_percent(materials, thicknesses, problem, wavelengths):
    """Compute mean reflectance in percent over the band."""
    n_layers = len(materials)
    if n_layers == 0:
        R = ((problem["ambient_index"] - problem["substrate_index"]) / 
             (problem["ambient_index"] + problem["substrate_index"])) ** 2
        return R * 100.0
    
    R = _reflectance_spectrum(materials, thicknesses, n_layers, 
                               problem["ambient_index"], problem["substrate_index"], wavelengths)
    return np.mean(R) * 100.0

def _genetic_algorithm(problem, wavelengths, deadline):
    """
    Genetic algorithm over discrete layer sequences and thicknesses.
    Each individual is a sequence of (label, thickness) pairs.
    """
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    
    # Discrete thickness grid: quarter-wave at various center wavelengths
    # Optical thickness = n*d, so d = optical_thickness / n
    # Quarter-wave optical thickness = lambda/4
    # For lambda in [400, 800], optical thickness in [100, 200] nm
    # Let's use a grid of optical thicknesses from 80 to 220 nm in steps of 10 nm
    opt_thick_grid = np.arange(80, 221, 10)  # 15 values
    n_opt_values = len(opt_thick_grid)
    
    # Population size
    pop_size = 200
    max_generations = 50
    
    # Represent an individual as:
    # - n_layers: 1 to max_layers
    # - labels: array of 0 (L) or 1 (H), length n_layers
    # - opt_indices: array of indices into opt_thick_grid, length n_layers
    # We'll use a fixed maximum length representation and a "valid" count
    
    max_len = max_layers
    
    def decode_individual(ind):
        """Decode a flat array into (materials_list, thicknesses_list)."""
        # First element: number of layers (1 to max_layers)
        n_layers = int(np.clip(ind[0], 1, max_len))
        labels = ind[1:1+n_layers].astype(int)
        opt_indices = ind[1+n_layers:1+2*n_layers].astype(int)
        opt_indices = np.clip(opt_indices, 0, n_opt_values - 1)
        
        materials = []
        thicknesses = []
        for i in range(n_layers):
            if labels[i] == 0:
                materials.append(n_L)
            else:
                materials.append(n_H)
            opt_thick = opt_thick_grid[opt_indices[i]]
            if labels[i] == 0:
                thicknesses.append(opt_thick / n_L)
            else:
                thicknesses.append(opt_thick / n_H)
        
        return materials, thicknesses
    
    def evaluate(ind):
        """Evaluate an individual. Returns mean reflectance in percent."""
        if time.time() > deadline:
            return 100.0
        materials, thicknesses = decode_individual(ind)
        total = sum(thicknesses)
        if total > max_total_nm:
            return 100.0
        for t in thicknesses:
            if t > max_layer_nm:
                return 100.0
        return _mean_reflectance_percent(materials, thicknesses, problem, wavelengths)
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        ind = np.zeros(1 + 2 * max_len)
        n_layers = np.random.randint(1, max_len + 1)
        ind[0] = n_layers
        ind[1:1+n_layers] = np.random.randint(0, 2, n_layers)
        ind[1+n_layers:1+2*n_layers] = np.random.randint(0, n_opt_values, n_layers)
        population.append(ind)
    
    # Evaluate initial population
    fitnesses = np.array([evaluate(ind) for ind in population])
    
    best_ind = population[np.argmin(fitnesses)].copy()
    best_fitness = np.min(fitnesses)
    
    # Genetic algorithm loop
    for gen in range(max_generations):
        if time.time() > deadline:
            break
        
        # Sort by fitness
        order = np.argsort(fitnesses)
        population = [population[i] for i in order]
        fitnesses = fitnesses[order]
        
        if fitnesses[0] < best_fitness:
            best_fitness = fitnesses[0]
            best_ind = population[0].copy()
        
        # Elitism: keep top 10%
        elite_size = max(1, pop_size // 10)
        new_population = [population[i].copy() for i in range(elite_size)]
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            # Tournament selection
            def select():
                idxs = np.random.randint(0, pop_size, 3)
                return population[idxs[np.argmin(fitnesses[idxs])]].copy()
            
            parent1 = select()
            parent2 = select()
            
            # Crossover: uniform
            child = np.zeros_like(parent1)
            for i in range(len(child)):
                if np.random.rand() < 0.5:
                    child[i] = parent1[i]
                else:
                    child[i] = parent2[i]
            
            # Mutation
            if np.random.rand() < 0.3:
                # Mutate number of layers
                child[0] = np.clip(child[0] + np.random.randint(-2, 3), 1, max_len)
            
            n_layers = int(child[0])
            # Mutate labels
            for i in range(1, 1 + n_layers):
                if np.random.rand() < 0.1:
                    child[i] = 1 - child[i]
            # Mutate optical thickness indices
            for i in range(1 + n_layers, 1 + 2 * n_layers):
                if np.random.rand() < 0.1:
                    child[i] = np.clip(child[i] + np.random.randint(-2, 3), 0, n_opt_values - 1)
            
            new_population.append(child)
        
        population = new_population[:pop_size]
        fitnesses = np.array([evaluate(ind) for ind in population])
        
        if fitnesses[0] < best_fitness:
            best_fitness = fitnesses[0]
            best_ind = population[0].copy()
    
    return decode_individual(best_ind), best_fitness

def _refine_with_nelder_mead(problem, wavelengths, materials_init, thicknesses_init, deadline):
    """
    Refine a discrete solution using Nelder-Mead on continuous thicknesses.
    """
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    
    n_layers = len(materials_init)
    if n_layers == 0:
        return materials_init, thicknesses_init, _mean_reflectance_percent([], [], problem, wavelengths)
    
    x0 = np.array(thicknesses_init, dtype=float)
    bounds = np.array([[0.0, max_layer_nm]] * n_layers)
    
    def objective(x):
        th = np.clip(x, 0.0, max_layer_nm)
        if np.sum(th) > max_total_nm:
            return 100.0
        return _mean_reflectance_percent(materials_init, th, problem, wavelengths)
    
    # Simple Nelder-Mead
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    simplex = [x0.copy()]
    for i in range(n_layers):
        pert = x0.copy()
        pert[i] += 0.1 * (bounds[i][1] - bounds[i][0])
        simplex.append(np.clip(pert, bounds[:, 0], bounds[:, 1]))
    
    f_values = [objective(x) for x in simplex]
    best_x = simplex[0].copy()
    best_f = f_values[0]
    
    for iteration in range(300):
        if time.time() > deadline:
            break
        
        order = np.argsort(f_values)
        simplex = [simplex[i].copy() for i in order]
        f_values = [f_values[i] for i in order]
        
        if f_values[0] < best_f:
            best_f = f_values[0]
            best_x = simplex[0].copy()
        
        if np.max(f_values) - np.min(f_values) < 1e-8:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        x_r = centroid + alpha * (centroid - simplex[-1])
        x_r = np.clip(x_r, bounds[:, 0], bounds[:, 1])
        f_r = objective(x_r)
        
        if f_r < f_values[0]:
            x_e = centroid + gamma * (x_r - centroid)
            x_e = np.clip(x_e, bounds[:, 0], bounds[:, 1])
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
        elif f_r < f_values[-2]:
            simplex[-1] = x_r
            f_values[-1] = f_r
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
            x_c = np.clip(x_c, bounds[:, 0], bounds[:, 1])
            f_c = objective(x_c)
            if f_c < f_values[-1]:
                simplex[-1] = x_c
                f_values[-1] = f_c
            else:
                for i in range(1, n_layers + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds[:, 0], bounds[:, 1])
                    f_values[i] = objective(simplex[i])
    
    return materials_init, list(best_x), best_f

def design(problem):
    """
    Design a broadband antireflection coating using genetic algorithm
    over discrete quarter-wave optical thicknesses, followed by
    Nelder-Mead refinement.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Run genetic algorithm
    materials, thicknesses = _genetic_algorithm(problem, wavelengths, deadline)
    
    # Convert to labels
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    material_labels = []
    for n in materials:
        if abs(n - n_L) < 1e-6:
            material_labels.append("L")
        else:
            material_labels.append("H")
    
    # Refine with Nelder-Mead if time permits
    if time.time() < deadline - 20.0:
        materials, thicknesses, _ = _refine_with_nelder_mead(
            problem, wavelengths, materials, thicknesses, deadline
        )
        material_labels = []
        for n in materials:
            if abs(n - n_L) < 1e-6:
                material_labels.append("L")
            else:
                material_labels.append("H")
    
    # Final constraint check
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    total_thickness = sum(thicknesses)
    if total_thickness > max_total_nm:
        scale = max_total_nm / total_thickness
        thicknesses = [t * scale for t in thicknesses]
    thicknesses = [min(max(t, 0.0), max_layer_nm) for t in thicknesses]
    
    return material_labels, thicknesses
# EVOLVE-BLOCK-END
