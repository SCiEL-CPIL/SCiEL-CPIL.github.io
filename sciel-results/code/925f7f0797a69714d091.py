"""Broadband AR coating design using Differential Evolution with vectorized reflectance."""
import numpy as np
import time
import sys

# EVOLVE-BLOCK-START

def _reflectance_vectorized(n0, ns, n_layers, thicknesses, lambdas):
    """
    Compute reflectance for all wavelengths simultaneously.
    
    Parameters:
    - n0: ambient index (float)
    - ns: substrate index (float)
    - n_layers: array of layer indices, shape (N,)
    - thicknesses: array of physical thicknesses in nm, shape (N,)
    - lambdas: array of wavelengths in nm, shape (M,)
    
    Returns:
    - reflectance: array of reflectance values (0 to 1), shape (M,)
    """
    N = len(thicknesses)
    M = len(lambdas)
    
    if N == 0:
        # Bare substrate
        r = (n0 - ns) / (n0 + ns)
        return np.full(M, r * r)
    
    # Precompute delta for each layer and wavelength: delta = 2*pi*n*d/lambda
    # Shape: (N, M)
    delta = 2.0 * np.pi * n_layers[:, None] * thicknesses[:, None] / lambdas[None, :]
    
    # Characteristic matrix for each layer:
    # M_j = [[cos(delta_j), i*sin(delta_j)/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # We multiply all matrices together: M_total = M_1 * M_2 * ... * M_N
    
    # Start with identity
    m11 = np.ones((M,), dtype=complex)
    m12 = np.zeros((M,), dtype=complex)
    m21 = np.zeros((M,), dtype=complex)
    m22 = np.ones((M,), dtype=complex)
    
    for j in range(N):
        cos_d = np.cos(delta[j])
        sin_d = np.sin(delta[j])
        n_j = n_layers[j]
        
        # Layer matrix elements
        a = cos_d
        b = 1j * sin_d / n_j
        c = 1j * n_j * sin_d
        d = cos_d
        
        # Multiply current total matrix by layer matrix
        # new_m11 = m11*a + m12*c
        # new_m12 = m11*b + m12*d
        # new_m21 = m21*a + m22*c
        # new_m22 = m21*b + m22*d
        new_m11 = m11 * a + m12 * c
        new_m12 = m11 * b + m12 * d
        new_m21 = m21 * a + m22 * c
        new_m22 = m21 * b + m22 * d
        
        m11, m12, m21, m22 = new_m11, new_m12, new_m21, new_m22
    
    # Admittance: Y = (m11 + m12 * ns) / (m21 + m22 * ns)
    Y = (m11 + m12 * ns) / (m21 + m22 * ns)
    
    # Reflectance: R = |(Y - n0)/(Y + n0)|^2
    r = (Y - n0) / (Y + n0)
    reflectance = np.abs(r) ** 2
    
    return reflectance


def _mean_reflectance_percent(n0, ns, labels, thicknesses, lambdas):
    """
    Compute mean reflectance in percent.
    """
    N = len(thicknesses)
    if N == 0:
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r
    
    n_layers = np.array([problem_materials[label] for label in labels], dtype=float)
    thicknesses_arr = np.array(thicknesses, dtype=float)
    
    # Check constraints
    if np.any(thicknesses_arr < 0) or np.any(thicknesses_arr > 400.0):
        return 100.0 * (1.0 + 1.0)  # Large penalty
    
    if np.sum(thicknesses_arr) > 1000.0:
        return 100.0 * (1.0 + 1.0)
    
    refl = _reflectance_vectorized(n0, ns, n_layers, thicknesses_arr, lambdas)
    return 100.0 * np.mean(refl)


def design(problem):
    """
    Design a broadband antireflection coating using Differential Evolution.
    """
    global problem_materials
    
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    n0 = problem["ambient_index"]  # 1.0
    ns = problem["substrate_index"]  # 1.52
    materials = problem["materials"]  # {'L': 1.38, 'H': 2.35}
    band_nm = problem["band_nm"]  # [400, 800]
    max_layers = problem["max_layers"]  # 20
    max_total_nm = problem["max_total_nm"]  # 1000
    max_layer_nm = problem["max_layer_nm"]  # 400
    
    problem_materials = materials
    
    lambdas = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Define search space:
    # We'll search over a fixed number of layers (say 12) with variable thicknesses.
    # Labels are also part of the search. We'll encode labels as binary: 0=L, 1=H.
    
    # Strategy: Use a fixed number of layers (e.g., 12) and optimize both labels and thicknesses.
    # To handle variable number of layers, we can use a penalty for zero-thickness layers.
    
    num_layers = 12  # Fixed number of layers to search over
    
    def decode(params):
        """
        Decode parameters into labels and thicknesses.
        params: array of length 2*num_layers
        First num_layers: continuous values in [0,1] that map to labels
        Last num_layers: continuous values in [0, max_layer_nm]
        """
        label_params = params[:num_layers]
        thickness_params = params[num_layers:]
        
        labels = []
        thicknesses = []
        
        for i in range(num_layers):
            if label_params[i] < 0.5:
                labels.append('L')
            else:
                labels.append('H')
            
            # Clip thickness to [0, max_layer_nm]
            t = np.clip(thickness_params[i], 0.0, max_layer_nm)
            thicknesses.append(t)
        
        # Check total thickness constraint
        total = sum(thicknesses)
        if total > max_total_nm:
            # Scale down proportionally
            scale = max_total_nm / total
            thicknesses = [t * scale for t in thicknesses]
        
        # Remove zero-thickness layers
        final_labels = []
        final_thicknesses = []
        for l, t in zip(labels, thicknesses):
            if t > 1e-6:
                final_labels.append(l)
                final_thicknesses.append(t)
        
        # Enforce max layers
        if len(final_labels) > max_layers:
            # Keep the first max_layers layers
            final_labels = final_labels[:max_layers]
            final_thicknesses = final_thicknesses[:max_layers]
        
        return final_labels, final_thicknesses
    
    def objective(params):
        """Objective function for optimization."""
        labels, thicknesses = decode(params)
        return _mean_reflectance_percent(n0, ns, labels, thicknesses, lambdas)
    
    # Differential Evolution
    np.random.seed(42)
    
    # Bounds for each parameter
    # Label params: [0, 1]
    # Thickness params: [0, max_layer_nm]
    bounds = []
    for i in range(num_layers):
        bounds.append((0.0, 1.0))  # Label
    for i in range(num_layers):
        bounds.append((0.0, max_layer_nm))  # Thickness
    
    # Initial population
    pop_size = 30
    population = np.random.uniform(
        [b[0] for b in bounds],
        [b[1] for b in bounds],
        size=(pop_size, 2 * num_layers)
    )
    
    # Evaluate initial population
    fitness = np.array([objective(p) for p in population])
    best_idx = np.argmin(fitness)
    best_params = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # DE parameters
    CR = 0.7  # Crossover rate
    F = 0.8   # Differential weight
    max_generations = 200
    
    for gen in range(max_generations):
        if time.time() > deadline - 5.0:
            break
        
        # Create new population
        new_population = []
        new_fitness = []
        
        for i in range(pop_size):
            # Select three distinct random indices
            a, b, c = np.random.choice(pop_size, 3, replace=False)
            
            # Create trial vector
            trial = population[a].copy()
            for j in range(2 * num_layers):
                if np.random.rand() < CR:
                    trial[j] = population[a][j] + F * (population[b][j] - population[c][j])
                else:
                    trial[j] = population[a][j]
            
            # Clip to bounds
            for j in range(2 * num_layers):
                trial[j] = np.clip(trial[j], bounds[j][0], bounds[j][1])
            
            # Evaluate
            f_trial = objective(trial)
            
            if f_trial <= fitness[i]:
                new_population.append(trial)
                new_fitness.append(f_trial)
            else:
                new_population.append(population[i])
                new_fitness.append(fitness[i])
        
        population = np.array(new_population)
        fitness = np.array(new_fitness)
        
        # Update best
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_fitness:
            best_fitness = fitness[gen_best_idx]
            best_params = population[gen_best_idx].copy()
    
    # Final result
    labels, thicknesses = decode(best_params)
    
    # Ensure valid output
    if len(labels) == 0:
        # Fallback to single layer
        centre = 0.5 * (band_nm[0] + band_nm[1])
        labels = ['L']
        thicknesses = [centre / (4.0 * materials['L'])]
    
    return labels, thicknesses

# EVOLVE-BLOCK-END
