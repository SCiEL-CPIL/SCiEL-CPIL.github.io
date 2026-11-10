"""Broadband AR coating design using Differential Evolution with vectorized reflectance."""
import numpy as np
import time
import sys

# EVOLVE-BLOCK-START

def _reflectance_vectorized(materials, thicknesses, lambdas, n0, ns):
    """
    Compute reflectance spectrum using characteristic matrix method, vectorized over wavelengths.
    materials: list of refractive indices (one per layer)
    thicknesses: list of physical thicknesses in nm
    lambdas: array of wavelengths in nm
    n0: ambient index, ns: substrate index
    Returns: array of reflectance values (0 to 1)
    """
    N = len(thicknesses)
    M = len(lambdas)
    
    # Pre-compute phase factors and admittances for all wavelengths
    # d_j = 2*pi*n_j*t_j / lambda
    # For each layer j and wavelength m: d[j,m] = 2*pi*n[j]*t[j]/lambda[m]
    
    # Build the total characteristic matrix by multiplying layer matrices
    # Start with identity for the substrate interface
    # Actually, we need to build from the substrate side or air side.
    # Convention: M_total = M_1 * M_2 * ... * M_N * M_substrate
    # where M_j is the j-th layer matrix (from air side)
    # M_substrate = [[1, 0], [1/ns, 0]] ... no wait.
    
    # Standard characteristic matrix method:
    # For a stack of N layers between ambient (n0) and substrate (ns):
    # The characteristic matrix of layer j is:
    #   [[cos(d_j), i*sin(d_j)/n_j], [i*n_j*sin(d_j), cos(d_j)]]
    # where d_j = 2*pi*n_j*t_j/lambda
    # 
    # Total matrix M = M_1 * M_2 * ... * M_N
    # Then:
    #   Y = (M[0,0] + M[0,1]*ns) / (M[1,0] + M[1,1]*ns)  ... wait, let me be careful.
    #
    # Actually the standard formula:
    # B = M[0,0] + M[0,1]*ns
    # C = M[1,0] + M[1,1]*ns
    # Y = C/B  (input admittance)
    # r = (n0 - Y) / (n0 + Y)
    # R = |r|^2
    
    # Initialize total matrix as identity
    M11 = np.ones((M,), dtype=complex)
    M12 = np.zeros((M,), dtype=complex)
    M21 = np.zeros((M,), dtype=complex)
    M22 = np.ones((M,), dtype=complex)
    
    # Multiply by each layer matrix from left (air side) to right (substrate side)
    for j in range(N):
        nj = materials[j]
        tj = thicknesses[j]
        d = 2.0 * np.pi * nj * tj / lambdas  # shape (M,)
        
        cd = np.cos(d)
        sd = np.sin(d)
        
        # Layer matrix: [[cos(d), i*sin(d)/n], [i*n*sin(d), cos(d)]]
        # New total = old_total * layer_matrix
        # M_new[0,0] = M11*cd + M12*(i*sd/nj)
        # M_new[0,1] = M11*(i*sd/nj) + M12*cd
        # M_new[1,0] = M21*cd + M22*(i*nj*sd)
        # M_new[1,1] = M21*(i*nj*sd) + M22*cd
        
        i_sd_nj = 1j * sd / nj
        i_nj_sd = 1j * nj * sd
        
        new_M11 = M11 * cd + M12 * i_sd_nj
        new_M12 = M11 * i_sd_nj + M12 * cd
        new_M21 = M21 * cd + M22 * i_nj_sd
        new_M22 = M21 * i_nj_sd + M22 * cd
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    # Compute input admittance
    B = M11 + M12 * ns
    C = M21 + M22 * ns
    Y = C / B
    r = (n0 - Y) / (n0 + Y)
    R = np.abs(r)**2
    return R

def _objective(params, problem, lambdas, n0, ns, max_layer_nm, max_total_nm):
    """
    Objective function for optimization.
    params: array of [n_layers, t_1, t_2, ..., t_n_layers]
    First element is the number of layers (will be rounded), rest are thicknesses.
    """
    n_layers = int(round(params[0]))
    n_layers = max(0, min(n_layers, 20))
    if n_layers == 0:
        return 10.0  # penalize empty stack
    
    thicknesses = params[1:1+n_layers]
    # Clip thicknesses to valid range
    thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
    
    total_thick = np.sum(thicknesses)
    if total_thick > max_total_nm:
        return 10.0 + (total_thick - max_total_nm) / max_total_nm * 10.0
    
    # Assign materials: alternate L, H, L, H, ... starting with L
    materials = []
    for i in range(n_layers):
        if i % 2 == 0:
            materials.append(problem["materials"]["L"])
        else:
            materials.append(problem["materials"]["H"])
    
    R = _reflectance_vectorized(materials, thicknesses, lambdas, n0, ns)
    return np.mean(R) * 100.0  # in percent

def design(problem):
    """Design broadband AR coating using Differential Evolution."""
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mat_L = problem["materials"]["L"]
    mat_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths: 401 equally spaced points in [400, 800]
    lambdas = np.linspace(band[0], band[1], 401)
    
    # Custom Differential Evolution implementation (NumPy only)
    # Parameter vector: [n_layers, t_1, ..., t_20]
    # n_layers in [0, 20], each t_i in [0, max_layer_nm]
    
    def _de_optimize(objective, bounds, popsize=40, maxiter=200, seed=42, deadline=None):
        np.random.seed(seed)
        D = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        # Initialize population
        pop = lower + (upper - lower) * np.random.rand(popsize, D)
        # Ensure n_layers (first param) is integer-ish for objective, but keep continuous for DE
        # The objective handles rounding.
        
        fitness = np.array([objective(ind, *objective_args) for ind in pop])
        best_idx = np.argmin(fitness)
        best_individual = pop[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        
        for it in range(maxiter):
            if deadline and time.time() > deadline - 5.0:
                break
            
            for i in range(popsize):
                # Select 3 distinct random individuals
                r1, r2, r3 = np.random.choice([j for j in range(popsize) if j != i], 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(D)
                for j in range(D):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Bounds check
                trial = np.clip(trial, lower, upper)
                
                # Selection
                trial_fitness = objective(trial, *objective_args)
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()
            
            # Optional: restart if stagnant? Keep simple for now.
        
        return best_individual, best_fitness

    objective_args = (problem, lambdas, n0, ns, max_layer_nm, max_total_nm)
    bounds = [(0, max_layers)] + [(0.0, max_layer_nm)] * max_layers
    
    best_result = None
    best_score = 10.0
    
    try:
        # Run DE
        # Use a smaller population and more iterations if time permits, but keep it safe.
        # 40 pop, 200 iter is ~8000 evals. Each eval is ~401 wavelengths * 20 layers.
        # Vectorized, this should be fast.
        best_result, best_score = _de_optimize(
            _objective,
            bounds,
            popsize=30,
            maxiter=100,
            seed=42,
            deadline=deadline
        )
    except Exception as e:
        pass
    
    # Check if we have a result
    if best_result is None:
        # Fallback: simple 2-layer quarter-wave stack
        centre = 0.5 * (band[0] + band[1])
        t_L = centre / (4.0 * mat_L)
        t_H = centre / (4.0 * mat_H)
        # Check total thickness
        if t_L + t_H <= max_total_nm and t_L <= max_layer_nm and t_H <= max_layer_nm:
            return ["L", "H"], [t_L, t_H]
        else:
            return ["L"], [min(t_L, max_layer_nm)]
    
    # Extract solution
    n_layers = int(round(best_result[0]))
    n_layers = max(0, min(n_layers, max_layers))
    thicknesses = best_result[1:1+n_layers]
    thicknesses = np.clip(thicknesses, 0.0, max_layer_nm)
    
    # Remove zero-thickness layers
    valid_indices = [i for i, t in enumerate(thicknesses) if t > 1.0]
    if not valid_indices:
        return ["L"], [0.0]
    
    thicknesses = [thicknesses[i] for i in valid_indices]
    n_layers = len(thicknesses)
    
    # Assign materials: alternate L, H starting with L
    materials = []
    for i in range(n_layers):
        if i % 2 == 0:
            materials.append("L")
        else:
            materials.append("H")
    
    # Verify constraints
    total = sum(thicknesses)
    if total > max_total_nm:
        # Scale down
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
    
    return materials, thicknesses

# EVOLVE-BLOCK-END
