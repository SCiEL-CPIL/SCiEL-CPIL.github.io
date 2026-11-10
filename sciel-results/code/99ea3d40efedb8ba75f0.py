import numpy as np
import time
import random

def design(problem):
    """
    Design a broadband antireflection coating using local search with
    differential-evolution-style global search and a final polish.
    """
    n_0 = problem["ambient_index"]
    n_s = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation (401 points)
    wavelengths = np.linspace(band[0], band[1], 401)
    n_wav = len(wavelengths)
    
    # Precompute optical thickness factors: 2*pi*n/dlambda for each layer type
    # For each wavelength, compute k = 2*pi*n/lambda
    # We'll vectorize the characteristic matrix computation
    
    def compute_reflectance_vec(materials, thicknesses):
        """
        Compute mean reflectance (in percent) for given materials and thicknesses.
        materials: list of 'L' or 'H'
        thicknesses: list of floats in nm
        """
        n_layers = len(materials)
        if n_layers == 0:
            # Bare substrate
            r = (n_0 - n_s) / (n_0 + n_s)
            return 100.0 * r * r
        
        # Build the characteristic matrix for each wavelength
        # M_i = [[cos(delta_i), i*sin(delta_i)/n_i], [i*n_i*sin(delta_i), cos(delta_i)]]
        # where delta_i = 2*pi*n_i*d_i/lambda
        
        # For efficiency, we process all wavelengths at once using vectorized ops
        # But the matrix multiplication is sequential over layers, so we'll loop layers
        # but vectorize over wavelengths.
        
        # Start with identity matrix for all wavelengths
        # M_total = I (2x2) for each wavelength
        # Represent as two arrays: M[0,0], M[0,1], M[1,0], M[1,1] each of shape (n_wav,)
        M00 = np.ones(n_wav, dtype=complex)
        M01 = np.zeros(n_wav, dtype=complex)
        M10 = np.zeros(n_wav, dtype=complex)
        M11 = np.ones(n_wav, dtype=complex)
        
        for i in range(n_layers):
            mat = materials[i]
            d = thicknesses[i]
            if mat == 'L':
                n_i = n_L
            else:
                n_i = n_H
            
            # delta = 2*pi*n_i*d/lambda
            delta = 2.0 * np.pi * n_i * d / wavelengths  # shape (n_wav,)
            
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            # Layer matrix
            # L00 = cos_d
            # L01 = 1j * sin_d / n_i
            # L10 = 1j * n_i * sin_d
            # L11 = cos_d
            
            L00 = cos_d
            L01 = 1j * sin_d / n_i
            L10 = 1j * n_i * sin_d
            L11 = cos_d
            
            # M_new = M_old @ L
            # M_new[0,0] = M00*L00 + M01*L10
            # M_new[0,1] = M00*L01 + M01*L11
            # M_new[1,0] = M10*L00 + M11*L10
            # M_new[1,1] = M10*L01 + M11*L11
            
            new_M00 = M00 * L00 + M01 * L10
            new_M01 = M00 * L01 + M01 * L11
            new_M10 = M10 * L00 + M11 * L10
            new_M11 = M10 * L01 + M11 * L11
            
            M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
        
        # Admittance Y = n_0 * M11 / M01 (for normal incidence, p-polarization same as s)
        # Actually, the admittance is Y = M11 / M01 (in units of n_0)
        # Reflectance R = |(Y - 1)/(Y + 1)|^2 where Y is normalized by n_0
        # More precisely: r = (Y - n_0) / (Y + n_0) where Y = M11/M01 * n_0... 
        # Let me be careful. The characteristic matrix gives:
        # [E]   [M00 M01] [E_s]
        # [H] = [M10 M11] [H_s]
        # At substrate: H_s = (n_s/n_0) * E_s (for normal incidence, H = n*E in normalized units)
        # So: M10*E_s + M11*H_s = 0 => M10 + M11*(n_s/n_0) = 0... 
        # Actually, the standard formula: 
        # Y = n_0 * M11 / M01 (if we normalize H by n_0)
        # r = (Y - n_0) / (Y + n_0)
        
        # Let me use the standard result:
        # The input admittance Y_in = M11 / M01 (in units where H is normalized)
        # Then r = (Y_in - n_0) / (Y_in + n_0)
        
        Y_in = M11 / M01
        r = (Y_in - n_0) / (Y_in + n_0)
        R = np.abs(r) ** 2
        return float(np.mean(R) * 100.0)
    
    def is_valid(materials, thicknesses):
        """Check if the design satisfies all constraints."""
        n_layers = len(materials)
        if n_layers > max_layers:
            return False
        total = sum(thicknesses)
        if total > max_total_nm:
            return False
        for t in thicknesses:
            if t < 0 or t > max_layer_nm:
                return False
        return True
    
    # Initial design: quarter-wave stack centered at 600 nm
    centre = 0.5 * (band[0] + band[1])
    # Start with a few layers
    initial_n_layers = 4
    initial_materials = []
    initial_thicknesses = []
    for i in range(initial_n_layers):
        if i % 2 == 0:
            initial_materials.append('L')
            initial_thicknesses.append(centre / (4.0 * n_L))
        else:
            initial_materials.append('H')
            initial_thicknesses.append(centre / (4.0 * n_H))
    
    # Ensure validity
    if not is_valid(initial_materials, initial_thicknesses):
        # Scale down if needed
        total = sum(initial_thicknesses)
        if total > max_total_nm:
            scale = max_total_nm / total
            initial_thicknesses = [t * scale for t in initial_thicknesses]
    
    best_score = compute_reflectance_vec(initial_materials, initial_thicknesses)
    best_materials = list(initial_materials)
    best_thicknesses = list(initial_thicknesses)
    
    # Time budget: aim to finish within 120s, leave 10s margin
    start_time = time.time()
    time_budget = 110.0  # seconds
    
    # Local search: perturb thicknesses and material choices
    random.seed(42)
    
    # Differential evolution parameters
    pop_size = 30
    F = 0.8  # scaling factor
    CR = 0.9  # crossover rate
    
    # Population: list of (materials, thicknesses, score)
    # We'll use a fixed number of layers for simplicity in the population,
    # but allow varying numbers in the final polish
    
    # Start with the initial design and random perturbations
    population = []
    for i in range(pop_size):
        if i == 0:
            mats = list(best_materials)
            thicks = list(best_thicknesses)
        else:
            # Random design
            n_layers = random.randint(2, min(8, max_layers))
            mats = []
            thicks = []
            for j in range(n_layers):
                if j == 0:
                    # First layer should be low index for antireflection
                    mats.append('L')
                elif j == n_layers - 1:
                    # Last layer should be high index to match substrate
                    mats.append('H')
                else:
                    mats.append(random.choice(['L', 'H']))
                thicks.append(random.uniform(10, min(200, max_layer_nm)))
            # Ensure validity
            if not is_valid(mats, thicks):
                total = sum(thicks)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    thicks = [t * scale for t in thicks]
        
        score = compute_reflectance_vec(mats, thicks)
        population.append((mats, thicks, score))
    
    # Sort population by score
    population.sort(key=lambda x: x[2])
    
    # Update best
    if population[0][2] < best_score:
        best_score = population[0][2]
        best_materials = list(population[0][0])
        best_thicknesses = list(population[0][1])
    
    # Evolution loop
    generation = 0
    while time.time() - start_time < time_budget:
        generation += 1
        
        # Check time
        if time.time() - start_time > time_budget - 5:
            break
        
        # Differential evolution step
        new_population = []
        
        for i in range(pop_size):
            if i == 0:
                # Keep the best
                new_population.append(population[0])
                continue
            
            # Select three distinct random individuals
            a, b, c = random.sample(range(pop_size), 3)
            
            # Create trial vector
            trial_mats = list(population[a][0])
            trial_thicks = list(population[a][0] if False else population[a][1])
            
            # Perturb thicknesses
            n_layers = len(trial_thicks)
            for j in range(n_layers):
                if random.random() < CR:
                    # Differential mutation
                    delta = F * (population[b][1][j] - population[c][1][j]) if j < len(population[b][1]) and j < len(population[c][1]) else 0
                    trial_thicks[j] += delta
                    # Clamp
                    trial_thicks[j] = max(0, min(max_layer_nm, trial_thicks[j]))
                else:
                    # Keep original
                    pass
            
            # Sometimes change a material
            if random.random() < 0.1:
                idx = random.randint(0, len(trial_mats) - 1)
                if idx == 0:
                    # Keep first as L
                    pass
                elif idx == len(trial_mats) - 1:
                    # Keep last as H
                    pass
                else:
                    trial_mats[idx] = 'L' if trial_mats[idx] == 'H' else 'H'
            
            # Check validity and adjust
            if not is_valid(trial_mats, trial_thicks):
                total = sum(trial_thicks)
                if total > max_total_nm:
                    scale = max_total_nm / total
                    trial_thicks = [t * scale for t in trial_thicks]
            
            trial_score = compute_reflectance_vec(trial_mats, trial_thicks)
            
            # Selection
            if trial_score < population[i][2]:
                new_population.append((trial_mats, trial_thicks, trial_score))
            else:
                new_population.append(population[i])
        
        # Update population
        population = new_population
        population.sort(key=lambda x: x[2])
        
        # Update best
        if population[0][2] < best_score:
            best_score = population[0][2]
            best_materials = list(population[0][0])
            best_thicknesses = list(population[0][1])
    
    # Final polish: local search on the best design
    if time.time() - start_time < time_budget - 2:
        # Try to improve by adjusting number of layers and fine-tuning
        for n_layers_try in [len(best_materials), len(best_materials) + 1, len(best_materials) - 1]:
            if n_layers_try < 1 or n_layers_try > max_layers:
                continue
            
            # Create a design with n_layers_try layers
            mats = []
            thicks = []
            for j in range(n_layers_try):
                if j == 0:
                    mats.append('L')
                    thicks.append(centre / (4.0 * n_L))
                elif j == n_layers_try - 1:
                    mats.append('H')
                    thicks.append(centre / (4.0 * n_H))
                else:
                    # Alternate
                    if j % 2 == 0:
                        mats.append('L')
                        thicks.append(centre / (4.0 * n_L))
                    else:
                        mats.append('H')
                        thicks.append(centre / (4.0 * n_H))
            
            # Fine-tune with local search
            step = 10.0
            for _ in range(50):
                if time.time() - start_time > time_budget - 1:
                    break
                improved = False
                for j in range(len(thicks)):
                    for delta in [-step, step]:
                        new_thicks = list(thicks)
                        new_thicks[j] = max(0, min(max_layer_nm, new_thicks[j] + delta))
                        if is_valid(mats, new_thicks):
                            score = compute_reflectance_vec(mats, new_thicks)
                            if score < best_score:
                                best_score = score
                                best_materials = list(mats)
                                best_thicknesses = list(new_thicks)
                                thicks = new_thicks
                                improved = True
                if not improved:
                    step *= 0.5
                    if step < 0.5:
                        break
    
    # Final validity check
    if not is_valid(best_materials, best_thicknesses):
        # Fallback to a simple valid design
        best_materials = ['L']
        best_thicknesses = [centre / (4.0 * n_L)]
    
    return best_materials, best_thicknesses
