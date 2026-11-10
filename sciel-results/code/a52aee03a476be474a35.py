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

def _decode_candidate(x, n0, ns, nL, nH, max_layers, max_total_nm, max_layer_nm):
    """
    Decode a candidate vector into materials and thicknesses.
    x[0] encodes number of layers (0 to max_layers-1)
    x[1:] encodes thicknesses and material choices
    """
    # Number of layers: x[0] in [0, max_layers)
    n_layers = int(x[0])
    n_layers = max(1, min(n_layers, max_layers))
    
    # Material choices: x[1:1+n_layers] in [0, 1) -> L or H
    mat_choices = x[1:1+n_layers]
    materials = []
    for mc in mat_choices:
        if mc < 0.5:
            materials.append(nL)
        else:
            materials.append(nH)
    
    # Thicknesses: x[1+n_layers:1+n_layers+n_layers] in [0, max_layer_nm]
    thicknesses = np.clip(x[1+n_layers:1+n_layers+n_layers], 0.0, max_layer_nm)
    
    # Ensure total thickness constraint
    total = np.sum(thicknesses)
    if total > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / total)
    
    return materials, thicknesses

def _objective_candidate(x, lambdas, n0, ns, nL, nH, max_layers, max_total_nm, max_layer_nm):
    """Objective function for a single candidate."""
    materials, thicknesses = _decode_candidate(x, n0, ns, nL, nH, max_layers, max_total_nm, max_layer_nm)
    R = _reflectance_vectorized(materials, thicknesses, lambdas, n0, ns)
    return np.mean(R) * 100.0

def _nelder_mead(objective, x0, bounds=None, max_iter=100, tol=1e-5, time_limit=None, start_time=None):
    """
    Simple Nelder-Mead implementation using only NumPy.
    """
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 * xi[i] if abs(xi[i]) > 1e-6 else 0.1
        xi[i] += step
        simplex.append(xi)
    
    # Evaluate initial points
    values = [objective(x) for x in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Sort simplex
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[0] - values[-1]) < tol:
            break
        
        # Centroid of best n points
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < values[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            # Accept reflection
            simplex[-1] = xr
            values[-1] = fr
        else:
            # Contraction
            if fr < values[-1]:
                # Outside contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = objective(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        values[i] = objective(simplex[i])
            else:
                # Inside contraction
                xc = centroid - rho * (centroid - simplex[-1])
                fc = objective(xc)
                if fc <= values[-1]:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        values[i] = objective(simplex[i])
        
        # Apply bounds if provided
        if bounds is not None:
            for i in range(n + 1):
                for j in range(n):
                    if bounds[j] is not None:
                        low, high = bounds[j]
                        if simplex[i][j] < low:
                            simplex[i][j] = low
                        elif simplex[i][j] > high:
                            simplex[i][j] = high
    
    # Return best point
    best_idx = np.argmin(values)
    return simplex[best_idx], values[best_idx]

def design(problem):
    """Design broadband AR coating using Differential Evolution with vectorized reflectance."""
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths: 401 equally spaced points in [400, 800]
    lambdas = np.linspace(band[0], band[1], 401)
    
    # Candidate encoding:
    # x[0]: number of layers (0 to max_layers-1, will be int'd)
    # x[1:1+max_layers]: material choices (0=L, 1=H)
    # x[1+max_layers:1+2*max_layers]: thicknesses
    dim = 1 + 2 * max_layers
    
    # Bounds
    bounds = [(0.0, float(max_layers - 1))] + [(0.0, 1.0)] * max_layers + [(0.0, max_layer_nm)] * max_layers
    
    # Objective wrapper
    def _obj(x):
        return _objective_candidate(x, lambdas, n0, ns, nL, nH, max_layers, max_total_nm, max_layer_nm)
    
    # Differential Evolution parameters
    pop_size = 30
    F = 0.8
    CR = 0.9
    max_gen = 80
    
    # Initialize population
    np.random.seed(42)
    population = np.random.rand(pop_size, dim)
    # Apply bounds
    for i in range(dim):
        low, high = bounds[i]
        population[:, i] = low + population[:, i] * (high - low)
    
    # Evaluate initial population
    fitness = np.array([_obj(population[i]) for i in range(pop_size)])
    
    best_idx = np.argmin(fitness)
    best_x = population[best_idx].copy()
    best_f = fitness[best_idx]
    
    for gen in range(max_gen):
        if time.time() - t_start > deadline - 10.0:
            break
        
        # Sort population by fitness
        order = np.argsort(fitness)
        population = population[order]
        fitness = fitness[order]
        
        for i in range(pop_size):
            # Select three distinct random individuals
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            mutant = population[r1] + F * (population[r2] - population[r3])
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]
                # Apply bounds
                low, high = bounds[j]
                trial[j] = max(low, min(high, trial[j]))
            
            # Selection
            f_trial = _obj(trial)
            if f_trial < fitness[i]:
                population[i] = trial
                fitness[i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
        
        # Elitism: ensure best individual is preserved
        best_idx = np.argmin(fitness)
        if fitness[best_idx] > best_f:
            population[best_idx] = best_x
            fitness[best_idx] = best_f
    
    # Decode best candidate
    materials, thicknesses = _decode_candidate(best_x, n0, ns, nL, nH, max_layers, max_total_nm, max_layer_nm)
    
    # Local refinement with Nelder-Mead on the best candidate
    if time.time() < deadline - 20.0:
        try:
            x0 = best_x.copy()
            t_opt, fun_t = _nelder_mead(
                _obj,
                x0,
                bounds=bounds,
                max_iter=50,
                tol=1e-6,
                time_limit=deadline - time.time() - 5.0,
                start_time=t_start
            )
            if fun_t < best_f:
                best_x = t_opt
                best_f = fun_t
                materials, thicknesses = _decode_candidate(best_x, n0, ns, nL, nH, max_layers, max_total_nm, max_layer_nm)
        except Exception:
            pass
    
    # Final cleanup: remove zero-thickness layers
    valid_indices = [i for i, t in enumerate(thicknesses) if t > 0.5]
    if not valid_indices:
        # Fallback: simple 2-layer quarter-wave stack
        centre = 0.5 * (band[0] + band[1])
        t_L = centre / (4.0 * nL)
        t_H = centre / (4.0 * nH)
        if t_L + t_H <= max_total_nm and t_L <= max_layer_nm and t_H <= max_layer_nm:
            return ["L", "H"], [t_L, t_H]
        else:
            return ["L"], [min(t_L, max_layer_nm)]
    
    materials = [materials[i] for i in valid_indices]
    thicknesses = [thicknesses[i] for i in valid_indices]
    
    # Convert numeric indices to 'L'/'H' labels
    mat_labels = ['L' if m == nL else 'H' for m in materials]
    
    # Verify constraints
    total = sum(thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
    
    return mat_labels, thicknesses

# EVOLVE-BLOCK-END
