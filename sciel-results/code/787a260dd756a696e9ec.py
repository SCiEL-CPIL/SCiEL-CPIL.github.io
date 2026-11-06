"""Program for broadband antireflection coating design using discrete GA search."""
import numpy as np
import itertools
import time

# EVOLVE-BLOCK-START

def _char_matrix_reflectance_fast(wavelengths_nm, ambient_idx, substrate_idx,
                              material_indices, thicknesses_nm):
    """
    Compute normal-incidence reflectance (fraction, not percent) for each wavelength
    using the characteristic-matrix method. Optimized for speed.
    """
    n_layers = len(material_indices)
    if n_layers == 0:
        r = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(len(wavelengths_nm), r ** 2)
        
    reflectances = np.zeros(len(wavelengths_nm))
    
    # Pre-compute layer matrices for each wavelength? No, they depend on wavelength.
    # But we can vectorize the inner loop over layers if we store matrices.
    # However, 401 wavelengths * 20 layers is small enough that a simple loop is fine,
    # but we can avoid allocating complex arrays repeatedly.
    
    for i, lam in enumerate(wavelengths_nm):
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            d_j = thicknesses_nm[j]
            if d_j == 0:
                continue
            n_j = material_indices[j]
            delta = 2.0 * np.pi * n_j * d_j / lam
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            L = np.array([[cos_d, 1j * sin_d / n_j],
                          [1j * n_j * sin_d, cos_d]], dtype=complex)
            M = M @ L
        
        n_sub = substrate_idx
        M00, M01 = M[0,0], M[0,1]
        M10, M11 = M[1,0], M[1,1]
        
        denom = M00 + M01 * n_sub
        if abs(denom) < 1e-15:
            reflectances[i] = 1.0
            continue
        Y = (M10 + M11 * n_sub) / denom
        r = (ambient_idx - Y) / (ambient_idx + Y)
        reflectances[i] = np.abs(r) ** 2
    
    return reflectances

def _evaluate_candidate_fast(materials, thicknesses, problem, wavelengths_subset):
    """
    Evaluate a candidate design using a subset of wavelengths for speed.
    Returns mean reflectance in percent.
    """
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    materials_dict = problem.get("materials", {"L": 1.38, "H": 2.35})
    ambient_index = problem.get("ambient_index", 1.0)
    substrate_index = problem.get("substrate_index", 1.52)
    
    # Check constraints
    if len(materials) > max_layers:
        return float('inf')
    if len(materials) != len(thicknesses):
        return float('inf')
    
    total_thickness = sum(thicknesses)
    if total_thickness > max_total_nm:
        return float('inf')
    
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return float('inf')
    
    if len(materials) == 0:
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return 100.0 * r ** 2
    
    material_indices = []
    for m in materials:
        if m not in materials_dict:
            return float('inf')
        material_indices.append(materials_dict[m])
    
    # Compute reflectance over subset of wavelengths
    reflectances = _char_matrix_reflectance_fast(wavelengths_subset, ambient_index, substrate_index,
                                             material_indices, thicknesses)
    mean_R = np.mean(reflectances)
    return 100.0 * mean_R

def _evaluate_candidate_full(materials, thicknesses, problem):
    """
    Evaluate a candidate design using the full set of wavelengths for final scoring.
    """
    band_nm = problem.get("band_nm", [400, 800])
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    return _evaluate_candidate_fast(materials, thicknesses, problem, wavelengths)

def design(problem):
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    band_nm = problem.get("band_nm", [400, 800])
    materials_dict = problem.get("materials", {"L": 1.38, "H": 2.35})
    ambient_index = problem.get("ambient_index", 1.0)
    substrate_index = problem.get("substrate_index", 1.52)
    
    # Define discrete thickness grids
    # Center wavelength 600nm
    # L quarter wave: 600 / (4 * 1.38) = 108.7 nm
    # H quarter wave: 600 / (4 * 2.35) = 63.8 nm
    
    # We will use a grid that covers a reasonable range around the quarter-wave value.
    # Step size 2nm for finer resolution.
    L_grid = np.arange(0, 200, 2)
    H_grid = np.arange(0, 150, 2)
    
    # Ensure grids are within max_layer_nm
    L_grid = L_grid[L_grid <= max_layer_nm]
    H_grid = H_grid[H_grid <= max_layer_nm]
    
    # Add 0 if not present (it is, as we start from 0)
    
    # Subset of wavelengths for GA search (faster)
    wavelengths_subset = np.linspace(band_nm[0], band_nm[1], 41)
    
    # GA Parameters
    pop_size = 50
    max_generations = 40
    elite_size = 5
    
    # Chromosome: list of 20 (mat, t_idx) pairs
    # mat: 0 for L, 1 for H
    # t_idx: index into L_grid or H_grid
    
    def decode_chromosome(chrom):
        materials = []
        thicknesses = []
        for mat, t_idx in chrom:
            if mat == 0:
                t = L_grid[t_idx]
            else:
                t = H_grid[t_idx]
            if t > 0:
                materials.append('L' if mat == 0 else 'H')
                thicknesses.append(float(t))
        return materials, thicknesses

    def encode_design(materials, thicknesses):
        chrom = []
        for i in range(max_layers):
            if i < len(materials):
                mat = materials[i]
                t = thicknesses[i]
                if mat == 'L':
                    # Find closest index in L_grid
                    t_idx = np.argmin(np.abs(L_grid - t))
                    chrom.append((0, int(t_idx)))
                else:
                    t_idx = np.argmin(np.abs(H_grid - t))
                    chrom.append((1, int(t_idx)))
            else:
                # No layer
                chrom.append((0, 0)) # 0 thickness
        return chrom

    # Initialize population
    population = []
    
    # Add some known good structures
    # 1-layer quarter wave L
    chrom1 = encode_design(['L'], [L_grid[np.argmin(np.abs(L_grid - 108.7))]])
    population.append(chrom1)
    
    # 3-layer quarter wave L-H-L
    dL = L_grid[np.argmin(np.abs(L_grid - 108.7))]
    dH = H_grid[np.argmin(np.abs(H_grid - 63.8))]
    chrom2 = encode_design(['L', 'H', 'L'], [dL, dH, dL])
    population.append(chrom2)
    
    # 5-layer quarter wave L-H-L-H-L
    chrom3 = encode_design(['L', 'H', 'L', 'H', 'L'], [dL, dH, dL, dH, dL])
    population.append(chrom3)
    
    # Fill rest with random
    while len(population) < pop_size:
        chrom = []
        for _ in range(max_layers):
            # 70% chance to have a layer
            if np.random.random() < 0.7:
                mat = 0 if np.random.random() < 0.5 else 1
                if mat == 0:
                    t_idx = np.random.randint(1, len(L_grid)) # Avoid 0
                else:
                    t_idx = np.random.randint(1, len(H_grid))
                chrom.append((mat, t_idx))
            else:
                chrom.append((0, 0))
        population.append(chrom)
        
    # Evaluate initial population
    def evaluate_chrom(chrom):
        materials, thicknesses = decode_chromosome(chrom)
        return _evaluate_candidate_fast(materials, thicknesses, problem, wavelengths_subset)
        
    fitnesses = [evaluate_chrom(chrom) for chrom in population]
    
    best_fitness = min(fitnesses)
    best_chrom = population[fitnesses.index(best_fitness)]
    
    for gen in range(max_generations):
        # Sort
        indexed = list(enumerate(fitnesses))
        indexed.sort(key=lambda x: x[1])
        sorted_pop = [population[i] for i, _ in indexed]
        sorted_fit = [f for _, f in indexed]
        
        if sorted_fit[0] < best_fitness:
            best_fitness = sorted_fit[0]
            best_chrom = sorted_pop[0]
            
        new_population = [sorted_pop[i] for i in range(elite_size)]
        
        while len(new_population) < pop_size:
            # Tournament selection
            def tournament_select(pop, fit, k=3):
                indices = np.random.choice(len(pop), size=k, replace=False)
                best_i = indices[np.argmin([fit[i] for i in indices])]
                return pop[best_i]
                
            p1 = tournament_select(sorted_pop, sorted_fit)
            p2 = tournament_select(sorted_pop, sorted_fit)
            
            # Crossover
            if np.random.random() < 0.8:
                # Uniform crossover
                child1 = []
                child2 = []
                for i in range(max_layers):
                    if np.random.random() < 0.5:
                        child1.append(p1[i])
                        child2.append(p2[i])
                    else:
                        child1.append(p2[i])
                        child2.append(p1[i])
            else:
                child1 = list(p1)
                child2 = list(p2)
                
            # Mutation
            for i in range(max_layers):
                if np.random.random() < 0.2:
                    # Mutate layer i
                    if np.random.random() < 0.5:
                        # Change material
                        mat, t_idx = child1[i]
                        new_mat = 1 - mat
                        # Keep thickness index, but adjust if necessary? 
                        # Just keep the index, the grid is different but index is valid
                        child1[i] = (new_mat, t_idx)
                    else:
                        # Change thickness
                        mat, t_idx = child1[i]
                        if mat == 0:
                            new_t_idx = np.random.randint(0, len(L_grid))
                        else:
                            new_t_idx = np.random.randint(0, len(H_grid))
                        child1[i] = (mat, new_t_idx)
                        
                if np.random.random() < 0.2:
                    # Mutate layer i in child2
                    if np.random.random() < 0.5:
                        mat, t_idx = child2[i]
                        new_mat = 1 - mat
                        child2[i] = (new_mat, t_idx)
                    else:
                        mat, t_idx = child2[i]
                        if mat == 0:
                            new_t_idx = np.random.randint(0, len(L_grid))
                        else:
                            new_t_idx = np.random.randint(0, len(H_grid))
                        child2[i] = (mat, new_t_idx)
                        
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
                
        population = new_population
        fitnesses = [evaluate_chrom(chrom) for chrom in population]
        
        # Early stop if good enough
        if best_fitness < 1.0 and gen > 20:
            break
            
    # Local Refinement of top candidates
    top_candidates = []
    indexed = list(enumerate(fitnesses))
    indexed.sort(key=lambda x: x[1])
    for i in range(min(5, len(indexed))):
        top_candidates.append(population[indexed[i][0]])
    top_candidates.append(best_chrom)
    
    best_final_fitness = float('inf')
    best_final_chrom = best_chrom
    
    for chrom in top_candidates:
        # Hill climbing on discrete grid
        current_chrom = list(chrom)
        current_fit = evaluate_chrom(current_chrom)
        
        for step in range(20):
            improved = False
            # Try changing one layer
            for i in range(max_layers):
                # Try changing thickness
                mat, t_idx = current_chrom[i]
                if mat == 0:
                    for new_t_idx in [t_idx-1, t_idx+1]:
                        if 0 <= new_t_idx < len(L_grid):
                            new_chrom = list(current_chrom)
                            new_chrom[i] = (mat, new_t_idx)
                            new_fit = evaluate_chrom(new_chrom)
                            if new_fit < current_fit:
                                current_chrom = new_chrom
                                current_fit = new_fit
                                improved = True
                else:
                    for new_t_idx in [t_idx-1, t_idx+1]:
                        if 0 <= new_t_idx < len(H_grid):
                            new_chrom = list(current_chrom)
                            new_chrom[i] = (mat, new_t_idx)
                            new_fit = evaluate_chrom(new_chrom)
                            if new_fit < current_fit:
                                current_chrom = new_chrom
                                current_fit = new_fit
                                improved = True
                # Try changing material
                if np.random.random() < 0.5: # Only try material change sometimes to save time
                    new_mat = 1 - mat
                    new_chrom = list(current_chrom)
                    new_chrom[i] = (new_mat, t_idx)
                    new_fit = evaluate_chrom(new_chrom)
                    if new_fit < current_fit:
                        current_chrom = new_chrom
                        current_fit = new_fit
                        improved = True
                        
            if not improved:
                break
                
        # Evaluate with full wavelengths
        materials, thicknesses = decode_chromosome(current_chrom)
        final_fit = _evaluate_candidate_full(materials, thicknesses, problem)
        
        if final_fit < best_final_fitness:
            best_final_fitness = final_fit
            best_final_chrom = current_chrom
            
    # Decode best final chromosome
    materials, thicknesses = decode_chromosome(best_final_chrom)
    
    return materials, thicknesses

# EVOLVE-BLOCK-END
