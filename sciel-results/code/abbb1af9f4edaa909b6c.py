"""Broadband antireflection coating design using characteristic-matrix method."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    """
    Compute reflectance (0-1) for a stack using the characteristic matrix method.
    """
    n = len(n_layers)
    if n == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.abs(r) ** 2
    
    W = len(wavelengths)
    M11 = np.ones(W, dtype=complex)
    M12 = np.zeros(W, dtype=complex)
    M21 = np.zeros(W, dtype=complex)
    M22 = np.ones(W, dtype=complex)
    
    for i in range(n):
        ni = n_layers[i]
        di = d_layers[i]
        if di < 1e-10:
            continue
        delta = 2.0 * np.pi * ni * di / wavelengths
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        L11 = cos_d
        L12 = 1j * sin_d / ni
        L21 = 1j * ni * sin_d
        L22 = cos_d
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
    
    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    
    return R


def _mean_reflectance_percent(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    """Compute mean reflectance in percent."""
    R = _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers)
    return np.mean(R) * 100.0


def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if len(materials) == 0:
        return True
    
    total = sum(thicknesses)
    if total > max_total_nm + 1e-6:
        return False
    
    for t in thicknesses:
        if t < -1e-6 or t > max_layer_nm + 1e-6:
            return False
    
    for m in materials:
        if m not in problem["materials"]:
            return False
    
    return True


def _evaluate_design(materials, thicknesses, problem, wavelengths):
    """Evaluate a design and return mean reflectance in percent."""
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_layers = np.array([problem["materials"][m] for m in materials], dtype=float)
    d_layers = np.array(thicknesses, dtype=float)
    
    return _mean_reflectance_percent(wavelengths, n_ambient, n_substrate, n_layers, d_layers)


def _project_feasible(x, max_total_nm, max_layer_nm):
    """Project to feasible region."""
    x = np.clip(x, 0.0, max_layer_nm)
    total = np.sum(x)
    if total > max_total_nm:
        x = x * (max_total_nm / total)
    return x


def _nelder_mead_optimize(materials, initial_thicks, problem, time_limit):
    """Nelder-Mead optimization on layer thicknesses."""
    n_params = len(initial_thicks)
    if n_params == 0:
        return initial_thicks, _evaluate_design(materials, initial_thicks, problem, 
            np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401))
        
    n_vals = np.array([problem["materials"][m] for m in materials], dtype=float)
    lambdas = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    def eval_vec(x):
        x = np.clip(x, 0.0, max_layer)
        if np.sum(x) > max_total + 1e-6:
            return float('inf')
        return _mean_reflectance_percent(lambdas, n_ambient, n_substrate, n_vals, x)

    x0 = _project_feasible(np.array(initial_thicks, dtype=float), max_total, max_layer)
    
    # Build simplex
    simplex = [x0.copy()]
    step = 5.0
    for i in range(n_params):
        xi = x0.copy()
        xi[i] += step
        xi = _project_feasible(xi, max_total, max_layer)
        simplex.append(xi)
    
    simplex = np.array(simplex)
    f_values = np.array([eval_vec(simplex[i]) for i in range(n_params + 1)])
    
    best_score = np.min(f_values)
    best_thicks = simplex[np.argmin(f_values)].tolist()
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    max_iter = 500
    
    for _ in range(max_iter):
        if time.time() > time_limit:
            break
            
        idx = np.argsort(f_values)
        simplex = simplex[idx]
        f_values = f_values[idx]
        
        if np.abs(f_values[0] - f_values[-1]) < 1e-9:
            break
        if np.max(np.std(simplex, axis=0)) < 1e-7:
            break

        x_worst = simplex[-1]
        x_best = simplex[0]
        x_c = np.mean(simplex[:-1], axis=0)
        
        x_r = x_c + alpha * (x_c - x_worst)
        x_r = _project_feasible(x_r, max_total, max_layer)
        f_r = eval_vec(x_r)
        
        if f_r < f_values[0]:
            x_e = x_c + gamma * (x_r - x_c)
            x_e = _project_feasible(x_e, max_total, max_layer)
            f_e = eval_vec(x_e)
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
            if f_r < f_values[-1]:
                x_c_new = x_c + rho * (x_r - x_c)
            else:
                x_c_new = x_c - rho * (x_c - x_r)
            x_c_new = _project_feasible(x_c_new, max_total, max_layer)
            f_c = eval_vec(x_c_new)
            
            if f_c < min(f_r, f_values[-1]):
                simplex[-1] = x_c_new
                f_values[-1] = f_c
            else:
                for i in range(1, n_params + 1):
                    simplex[i] = x_best + sigma * (simplex[i] - x_best)
                    simplex[i] = _project_feasible(simplex[i], max_total, max_layer)
                    f_values[i] = eval_vec(simplex[i])
                    
        current_best_idx = np.argmin(f_values)
        if f_values[current_best_idx] < best_score:
            best_score = f_values[current_best_idx]
            best_thicks = simplex[current_best_idx].tolist()
            
    return best_thicks, best_score


def _differential_evolution(materials, initial_thicks, problem, time_limit, pop_size=20, max_iter=100):
    """Differential evolution for global optimization of thicknesses."""
    n_params = len(initial_thicks)
    if n_params == 0:
        return initial_thicks, _evaluate_design(materials, initial_thicks, problem,
            np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401))
    
    n_vals = np.array([problem["materials"][m] for m in materials], dtype=float)
    lambdas = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    def eval_vec(x):
        x = np.clip(x, 0.0, max_layer)
        if np.sum(x) > max_total + 1e-6:
            return float('inf')
        return _mean_reflectance_percent(lambdas, n_ambient, n_substrate, n_vals, x)

    lb = np.zeros(n_params)
    ub = np.full(n_params, max_layer)
    
    # Initialize population around the initial point
    x0 = _project_feasible(np.array(initial_thicks, dtype=float), max_total, max_layer)
    population = np.empty((pop_size, n_params))
    population[0] = x0
    for i in range(1, pop_size):
        # Mix of random and perturbed initial
        if random.random() < 0.3:
            population[i] = np.random.uniform(lb, ub)
        else:
            population[i] = x0 + random.gauss(0, 20)
        population[i] = _project_feasible(population[i], max_total, max_layer)
    
    fitness = np.array([eval_vec(population[i]) for i in range(pop_size)])
    
    best_idx = np.argmin(fitness)
    best_individual = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    F = 0.7
    CR = 0.8
    
    for gen in range(max_iter):
        if time.time() > time_limit:
            break
        
        for i in range(pop_size):
            # Select three distinct random individuals
            r1, r2, r3 = random.sample(range(pop_size), 3)
            
            # Mutation
            mutant = population[r1] + F * (population[r2] - population[r3])
            mutant = _project_feasible(mutant, max_total, max_layer)
            
            # Crossover
            trial = population[i].copy()
            j_rand = random.randint(0, n_params - 1)
            for j in range(n_params):
                if random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            trial = _project_feasible(trial, max_total, max_layer)
            
            # Selection
            f_trial = eval_vec(trial)
            if f_trial <= fitness[i]:
                population[i] = trial
                fitness[i] = f_trial
                if f_trial < best_fitness:
                    best_fitness = f_trial
                    best_individual = trial.copy()
        
        # Elitism: keep best
        pop_best_idx = np.argmin(fitness)
        if fitness[pop_best_idx] < best_fitness:
            best_fitness = fitness[pop_best_idx]
            best_individual = population[pop_best_idx].copy()
    
    return best_individual.tolist(), best_fitness


def _polish_thicknesses(mats, thicks, problem, wavelengths, deadline, max_iter=3000):
    """High-precision coordinate descent polish."""
    current_thicks = list(thicks)
    current_score = _evaluate_design(mats, current_thicks, problem, wavelengths)
    
    n = len(mats)
    if n == 0:
        return mats, current_thicks, current_score
    
    step_sizes = [2.0] * n
    min_step = 0.005
    max_step = 10.0
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            best_local_score = current_score
            best_local_thick = current_thicks[i]
            
            for factor in [1.0, 0.5, 0.25, 0.1, 2.0, 4.0]:
                step = step_sizes[i] * factor
                
                for sign in [1.0, -1.0]:
                    new_thick = current_thicks[i] + sign * step
                    new_thick = np.clip(new_thick, 0.0, problem["max_layer_nm"])
                    
                    new_thicks = list(current_thicks)
                    new_thicks[i] = new_thick
                    
                    if _is_valid(mats, new_thicks, problem):
                        new_score = _evaluate_design(mats, new_thicks, problem, wavelengths)
                        if new_score < best_local_score:
                            best_local_score = new_score
                            best_local_thick = new_thick
            
            if best_local_score < current_score - 1e-10:
                current_thicks[i] = best_local_thick
                current_score = best_local_score
                improved = True
        
        if not improved:
            step_sizes = [s * 0.6 for s in step_sizes]
            if min(step_sizes) < min_step:
                break
        else:
            step_sizes = [min(s * 1.2, max_step) for s in step_sizes]
    
    return mats, current_thicks, current_score


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s wall-clock limit
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    # Generate diverse initial designs
    initial_designs = []
    
    # Quarter-wave thicknesses
    qw_L = centre / (4.0 * n_L)
    qw_H = centre / (4.0 * n_H)
    
    # Single layer
    initial_designs.append((["L"], [qw_L]))
    initial_designs.append((["H"], [qw_H]))
    
    # Two-layer
    initial_designs.append((["L", "H"], [qw_L, qw_H]))
    initial_designs.append((["H", "L"], [qw_H, qw_L]))
    
    # Three-layer
    initial_designs.append((["H", "L", "H"], [qw_H, qw_L, qw_H]))
    initial_designs.append((["L", "H", "L"], [qw_L, qw_H, qw_L]))
    
    # Four-layer
    initial_designs.append((["H", "L", "H", "L"], [qw_H, qw_L, qw_H, qw_L]))
    initial_designs.append((["L", "H", "L", "H"], [qw_L, qw_H, qw_L, qw_H]))
    
    # Five-layer
    initial_designs.append((["H", "L", "H", "L", "H"], [qw_H, qw_L, qw_H, qw_L, qw_H]))
    initial_designs.append((["L", "H", "L", "H", "L"], [qw_L, qw_H, qw_L, qw_H, qw_L]))
    
    # Six-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L"], [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]))
    initial_designs.append((["L", "H", "L", "H", "L", "H"], [qw_L, qw_H, qw_L, qw_H, qw_L, qw_H]))
    
    # Seven-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H"], 
                           [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H]))
    
    # Eight-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L"],
                           [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]))
    
    # Nine-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H"],
                           [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H]))
    
    # Ten-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
                           [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]))
    
    # Non-standard patterns
    initial_designs.append((["H", "L", "L", "H"], [qw_H, qw_L, qw_L, qw_H]))
    initial_designs.append((["L", "H", "H", "L"], [qw_L, qw_H, qw_H, qw_L]))
    initial_designs.append((["H", "L", "L", "L", "H"], [qw_H, qw_L, qw_L, qw_L, qw_H]))
    initial_designs.append((["L", "H", "H", "H", "L"], [qw_L, qw_H, qw_H, qw_H, qw_L]))
    
    # Non-quarter-wave starting points for diversity
    initial_designs.append((["H", "L", "H", "L"], [qw_H * 0.8, qw_L * 1.2, qw_H * 0.9, qw_L * 1.1]))
    initial_designs.append((["L", "H", "L", "H"], [qw_L * 1.1, qw_H * 0.9, qw_L * 1.2, qw_H * 0.8]))
    
    # Evaluate all initial designs and keep the best
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    for mats, thicks in initial_designs:
        if not _is_valid(mats, thicks, problem):
            continue
        score = _evaluate_design(mats, thicks, problem, wavelengths)
        if score < best_score:
            best_score = score
            best_materials = list(mats)
            best_thicknesses = list(thicks)
    
    # Fallback if no valid initial design
    if best_materials is None:
        best_materials = ["L"]
        best_thicknesses = [qw_L]
        best_score = _evaluate_design(best_materials, best_thicknesses, problem, wavelengths)
    
    # Phase 1: Differential Evolution for global exploration (first 40s)
    de_deadline = start_time + 40.0
    de_thicks, de_score = _differential_evolution(
        best_materials, best_thicknesses, problem, de_deadline, pop_size=15, max_iter=80
    )
    if de_score < best_score:
        best_score = de_score
        best_thicknesses = de_thicks
    
    # Phase 2: Nelder-Mead local refinement (next 30s)
    nm_deadline = start_time + 70.0
    best_thicknesses, nm_score = _nelder_mead_optimize(
        best_materials, best_thicknesses, problem, nm_deadline
    )
    if nm_score < best_score:
        best_score = nm_score
    
    # Phase 3: Try a few alternative layer counts with DE+NM
    if time.time() < deadline - 40.0:
        # Try different layer counts
        for n_try in [4, 6, 8]:
            if time.time() > deadline - 40.0:
                break
            # Generate a pattern for this layer count
            mats_try = []
            thicks_try = []
            for i in range(n_try):
                if i % 2 == 0:
                    mats_try.append("H")
                    thicks_try.append(qw_H)
                else:
                    mats_try.append("L")
                    thicks_try.append(qw_L)
            
            if not _is_valid(mats_try, thicks_try, problem):
                continue
            
            # Quick DE
            sub_deadline = time.time() + 15.0
            de_thicks_try, de_score_try = _differential_evolution(
                mats_try, thicks_try, problem, sub_deadline, pop_size=10, max_iter=40
            )
            if de_score_try < best_score:
                best_score = de_score_try
                best_materials = list(mats_try)
                best_thicknesses = de_thicks_try
            
            # Quick NM
            if time.time() < deadline - 25.0:
                nm_thicks_try, nm_score_try = _nelder_mead_optimize(
                    best_materials, best_thicknesses, problem, time.time() + 10.0
                )
                if nm_score_try < best_score:
                    best_score = nm_score_try
                    best_thicknesses = nm_thicks_try
    
    # Phase 4: Final coordinate descent polish
    if time.time() < deadline - 5.0:
        best_materials, best_thicknesses, best_score = _polish_thicknesses(
            best_materials, best_thicknesses, problem, wavelengths, deadline - 2.0, max_iter=3000
        )
    
    # Final validation
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [qw_L]
        best_score = _evaluate_design(best_materials, best_thicknesses, problem, wavelengths)
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
