"""Broadband antireflection coating design via discrete genetic algorithm."""
import time
import numpy as np

def _reflectance_stack(materials, thicknesses, n_ambient, n_substrate, wavelengths):
    """
    Compute reflectance spectrum using characteristic matrix method.
    
    Args:
        materials: list of 'L' or 'H' labels
        thicknesses: list of physical thicknesses in nm
        n_ambient: ambient refractive index
        n_substrate: substrate refractive index
        wavelengths: array of wavelengths in nm
        
    Returns:
        array of reflectance values (0 to 1)
    """
    n_L = materials['L']
    n_H = materials['H']
    
    # Build complex refractive indices (lossless, so real)
    n_layers = []
    for m, t in zip(materials, thicknesses):
        n_layers.append(n_L if m == 'L' else n_H)
    
    num_layers = len(n_layers)
    R = np.zeros_like(wavelengths, dtype=float)
    
    for i, wl in enumerate(wavelengths):
        # Build characteristic matrix
        M = np.array([[1, 0], [0, 1]], dtype=complex)
        
        for j in range(num_layers):
            n_j = n_layers[j]
            t_j = thicknesses[j]
            k_j = 2 * np.pi * n_j / wl
            # For normal incidence, B matrix is [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
            delta = k_j * t_j
            c = np.cos(delta)
            s = np.sin(delta)
            B = np.array([[c, 1j*s/n_j], [1j*n_j*s, c]], dtype=complex)
            M = M @ B
        
        # Admittance
        Y = M[1, 1] / M[0, 1] if abs(M[0, 1]) > 1e-12 else 0
        
        # Reflectance
        r = (n_ambient - Y) / (n_ambient + Y)
        R[i] = np.abs(r) ** 2
    
    return R

def _fast_mean_R(materials, thicknesses, n_ambient, n_substrate, wavelengths):
    """Fast evaluation of mean reflectance."""
    R = _reflectance_stack(materials, thicknesses, n_ambient, n_substrate, wavelengths)
    return np.mean(R) * 100  # in percent

def design(problem):
    """
    Design antireflection coating using discrete genetic algorithm.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelength grid for evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    
    # Discrete thickness options: multiples of 25 nm, capped at max_layer_nm
    # Also include some quarter-wave values for the center wavelength
    center_wl = 0.5 * (band[0] + band[1])
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    # Base thickness options
    base_thick = list(range(25, min(max_layer_nm, 200) + 1, 25))
    # Add quarter-wave optical thicknesses
    qw_L = center_wl / (4 * n_L)
    qw_H = center_wl / (4 * n_H)
    # Round to nearest 25 and add neighbors
    for qw in [qw_L, qw_H]:
        rounded = round(qw / 25) * 25
        for delta in [-25, 0, 25]:
            t = rounded + delta
            if 25 <= t <= max_layer_nm and t not in base_thick:
                base_thick.append(t)
    
    base_thick = sorted(set(base_thick))
    num_thick_opts = len(base_thick)
    
    # Genetic algorithm setup
    pop_size = 100
    elite_size = 10
    mutation_rate = 0.15
    crossover_rate = 0.8
    
    # Initialize population: random sequences of L/H and thickness indices
    population = []
    for _ in range(pop_size):
        num_layers = np.random.randint(3, min(max_layers, 12) + 1)
        materials = [np.random.choice(['L', 'H']) for _ in range(num_layers)]
        thick_indices = [np.random.randint(0, num_thick_opts) for _ in range(num_layers)]
        population.append((materials, thick_indices))
    
    # Evaluate initial population
    scores = []
    for mats, tinds in population:
        thicknesses = [base_thick[i] for i in tinds]
        if sum(thicknesses) > max_total_nm:
            scores.append(np.inf)
        else:
            scores.append(_fast_mean_R(mats, thicknesses, n_ambient, n_substrate, wavelengths))
    
    # Track best
    best_idx = np.argmin(scores)
    best_mats = population[best_idx][0][:]
    best_tinds = population[best_idx][1][:]
    best_score = scores[best_idx]
    
    # Also try a simple quarter-wave stack as a baseline
    qw_stack_mats = ['L']
    qw_stack_tinds = [base_thick.index(round(qw_L / 25) * 25) if round(qw_L / 25) * 25 in base_thick else 0]
    qw_thicknesses = [base_thick[i] for i in qw_stack_tinds]
    if sum(qw_thicknesses) <= max_total_nm:
        qw_score = _fast_mean_R(qw_stack_mats, qw_thicknesses, n_ambient, n_substrate, wavelengths)
        if qw_score < best_score:
            best_score = qw_score
            best_mats = qw_stack_mats[:]
            best_tinds = qw_stack_tinds[:]
    
    # Evolution loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Sort by score
        order = np.argsort(scores)
        sorted_pop = [population[i] for i in order]
        sorted_scores = [scores[i] for i in order]
        
        # Elitism
        new_population = [sorted_pop[i] for i in range(min(elite_size, pop_size))]
        new_scores = [sorted_scores[i] for i in range(min(elite_size, pop_size))]
        
        # Fill rest with offspring
        while len(new_population) < pop_size and time.time() < deadline:
            # Tournament selection
            def select():
                idx1 = np.random.randint(0, pop_size)
                idx2 = np.random.randint(0, pop_size)
                return population[idx1] if scores[idx1] < scores[idx2] else population[idx2]
            
            parent1 = select()
            parent2 = select()
            
            # Crossover
            if np.random.random() < crossover_rate:
                # Simple crossover: take layers from parent1 or parent2
                max_len = max(len(parent1[0]), len(parent2[0]))
                child_mats = []
                child_tinds = []
                for i in range(max_len):
                    if np.random.random() < 0.5:
                        if i < len(parent1[0]):
                            child_mats.append(parent1[0][i])
                            child_tinds.append(parent1[1][i])
                    else:
                        if i < len(parent2[0]):
                            child_mats.append(parent2[0][i])
                            child_tinds.append(parent2[1][i])
                
                # Ensure at least 1 layer
                if len(child_mats) == 0:
                    child_mats = [np.random.choice(['L', 'H'])]
                    child_tinds = [np.random.randint(0, num_thick_opts)]
            else:
                child_mats = parent1[0][:]
                child_tinds = parent1[1][:]
            
            # Mutation
            if np.random.random() < mutation_rate:
                # Randomly mutate a layer type
                if child_mats:
                    idx = np.random.randint(0, len(child_mats))
                    child_mats[idx] = 'H' if child_mats[idx] == 'L' else 'L'
            
            if np.random.random() < mutation_rate:
                # Randomly mutate a thickness
                if child_tinds:
                    idx = np.random.randint(0, len(child_tinds))
                    child_tinds[idx] = np.random.randint(0, num_thick_opts)
            
            if np.random.random() < mutation_rate * 0.5:
                # Add or remove a layer
                if len(child_mats) < max_layers and np.random.random() < 0.5:
                    child_mats.append(np.random.choice(['L', 'H']))
                    child_tinds.append(np.random.randint(0, num_thick_opts))
                elif len(child_mats) > 1 and np.random.random() < 0.5:
                    idx = np.random.randint(0, len(child_mats))
                    child_mats.pop(idx)
                    child_tinds.pop(idx)
            
            new_population.append((child_mats, child_tinds))
            
            # Evaluate
            thicknesses = [base_thick[i] for i in child_tinds]
            if sum(thicknesses) > max_total_nm:
                new_scores.append(np.inf)
            else:
                new_scores.append(_fast_mean_R(child_mats, thicknesses, n_ambient, n_substrate, wavelengths))
        
        # Update population
        population = new_population[:pop_size]
        scores = new_scores[:pop_size]
        
        # Update best
        min_idx = np.argmin(scores)
        if scores[min_idx] < best_score:
            best_score = scores[min_idx]
            best_mats = population[min_idx][0][:]
            best_tinds = population[min_idx][1][:]
        
        # Local refinement: try small perturbations around best
        if time.time() < deadline - 10:
            for _ in range(5):
                pert_mats = best_mats[:]
                pert_tinds = best_tinds[:]
                
                # Perturb one thickness
                if pert_tinds:
                    idx = np.random.randint(0, len(pert_tinds))
                    new_idx = np.clip(pert_tinds[idx] + np.random.choice([-1, 1]), 0, num_thick_opts - 1)
                    pert_tinds[idx] = new_idx
                
                thicknesses = [base_thick[i] for i in pert_tinds]
                if sum(thicknesses) <= max_total_nm:
                    score = _fast_mean_R(pert_mats, thicknesses, n_ambient, n_substrate, wavelengths)
                    if score < best_score:
                        best_score = score
                        best_mats = pert_mats[:]
                        best_tinds = pert_tinds[:]
    
    # Final result
    thicknesses = [base_thick[i] for i in best_tinds]
    return best_mats, thicknesses
