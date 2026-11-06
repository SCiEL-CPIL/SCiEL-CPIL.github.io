"""Broadband antireflection coating design with multi-start differential evolution + SA + Nelder-Mead polishing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _reflectance_vectorized(wavelengths, thicknesses, indices, ambient_n, substrate_n):
    """Compute reflectance spectrum using vectorized characteristic matrix method."""
    n_layers = len(thicknesses)
    if n_layers == 0:
        r0 = (ambient_n - substrate_n) / (ambient_n + substrate_n)
        return np.full(len(wavelengths), np.abs(r0) ** 2)
    
    M00 = np.ones(len(wavelengths), dtype=complex)
    M01 = np.zeros(len(wavelengths), dtype=complex)
    M10 = np.zeros(len(wavelengths), dtype=complex)
    M11 = np.ones(len(wavelengths), dtype=complex)
    
    for j in range(n_layers):
        n_j = indices[j]
        d_j = thicknesses[j]
        
        if d_j < 1e-10:
            continue
        
        delta_j = 2.0 * np.pi * n_j * d_j / wavelengths
        cos_d = np.cos(delta_j)
        sin_d = np.sin(delta_j)
        
        m00 = cos_d
        m01 = 1j * sin_d / n_j
        m10 = 1j * n_j * sin_d
        m11 = cos_d
        
        new_M00 = M00 * m00 + M01 * m10
        new_M01 = M00 * m01 + M01 * m11
        new_M10 = M10 * m00 + M11 * m10
        new_M11 = M10 * m01 + M11 * m11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    B = M00 + M01 * substrate_n
    C = M10 * ambient_n + M11 * substrate_n
    
    denom = C + B
    denom_safe = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    
    r = (C - B) / denom_safe
    reflectance = np.abs(r) ** 2
    
    return reflectance

def _evaluate(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    wavelengths = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    indices = np.array([problem["materials"][m] for m in materials], dtype=float)
    thicknesses_arr = np.array(thicknesses, dtype=float)
    refl = _reflectance_vectorized(wavelengths, thicknesses_arr, indices,
                                   problem["ambient_index"], problem["substrate_index"])
    return 100.0 * np.mean(refl)

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if len(materials) < 0:
        return False
    if any(t < -1e-9 or t > problem["max_layer_nm"] + 1e-9 for t in thicknesses):
        return False
    if sum(thicknesses) > problem["max_total_nm"] + 1e-9:
        return False
    if any(m not in problem["materials"] for m in materials):
        return False
    return True

def _project_to_feasible(x, max_total_nm, max_layer_nm):
    """Project a candidate onto the feasible region (non-negative, per-layer and total thickness constraints)."""
    x = np.clip(x, 0.0, max_layer_nm)
    total = x.sum()
    if total > max_total_nm:
        x = x * (max_total_nm / total)
    return x

def _nelder_mead(f, x0, bounds, maxiter=200, tol=1e-8, time_limit=None, start_time=None):
    """Nelder-Mead optimization with bounds and time limit."""
    if time_limit is None:
        time_limit = 10.0
    if start_time is None:
        start_time = time.time()
    
    n = len(x0)
    x0 = np.array(x0, dtype=float)
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        if abs(xi[i]) > 1e-10:
            xi[i] *= 1.05
        else:
            xi[i] += 0.05 * (bounds[i][1] - bounds[i][0])
        simplex.append(xi)
    
    # Evaluate initial simplex
    f_values = [f(x) for x in simplex]
    
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink
    
    for _ in range(maxiter):
        if time.time() - start_time > time_limit:
            break
        
        # Sort simplex
        order = np.argsort(f_values)
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]
        
        # Check convergence
        if abs(f_values[-1] - f_values[0]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = f(xr)
        
        if fr < f_values[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            xe = np.clip(xe, [b[0] for b in bounds], [b[1] for b in bounds])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                f_values[-1] = fe
            else:
                simplex[-1] = xr
                f_values[-1] = fr
        elif fr < f_values[-2]:
            simplex[-1] = xr
            f_values[-1] = fr
        else:
            # Contraction
            if fr < f_values[-1]:
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
            else:
                xc = centroid - rho * (centroid - simplex[-1])
                fc = f(xc)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = f(xc)
            
            if fc <= fr:
                simplex[-1] = xc
                f_values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_values[i] = f(simplex[i])
    
    # Return best
    best_idx = np.argmin(f_values)
    return simplex[best_idx], f_values[best_idx]

def _differential_evolution(objective, bounds, pop_size=40, max_iter=300, F=0.8, CR=0.9,
                            time_limit=30.0, start_time=None):
    """Differential Evolution optimization."""
    if start_time is None:
        start_time = time.time()
    
    n = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # Initialize population
    population = np.random.uniform(lb, ub, size=(pop_size, n))
    fitness = np.array([objective(ind) for ind in population])
    
    best_idx = np.argmin(fitness)
    best_individual = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    for gen in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        for i in range(pop_size):
            # Select three distinct random individuals
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            v = population[r1] + F * (population[r2] - population[r3])
            v = np.clip(v, lb, ub)
            
            # Crossover
            trial = population[i].copy()
            j_rand = np.random.randint(n)
            for j in range(n):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = v[j]
            
            # Selection
            ft = objective(trial)
            if ft < fitness[i]:
                population[i] = trial
                fitness[i] = ft
                if ft < best_fitness:
                    best_fitness = ft
                    best_individual = trial.copy()
    
    return best_individual, best_fitness

def _simulated_annealing(objective, x0, bounds, max_iter=50000, T0=1.0, T1=0.001,
                         time_limit=20.0, start_time=None):
    """Simulated Annealing with diverse move operators."""
    if start_time is None:
        start_time = time.time()
    
    n = len(x0)
    x = np.array(x0, dtype=float)
    f_current = objective(x)
    x_best = x.copy()
    f_best = f_current
    
    T = T0
    cooling_factor = (T1 / T0) ** (1.0 / max_iter)
    
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Choose move type
        move_type = np.random.random()
        x_new = x.copy()
        
        if move_type < 0.45:
            # Single thickness perturbation
            idx = np.random.randint(n)
            range_size = ub[idx] - lb[idx]
            x_new[idx] += range_size * 0.15 * np.random.randn()
        elif move_type < 0.60:
            # Double thickness perturbation
            idx1, idx2 = np.random.choice(n, 2, replace=False)
            for idx in [idx1, idx2]:
                range_size = ub[idx] - lb[idx]
                x_new[idx] += range_size * 0.15 * np.random.randn()
        elif move_type < 0.70:
            # Large single perturbation
            idx = np.random.randint(n)
            range_size = ub[idx] - lb[idx]
            x_new[idx] += range_size * 0.5 * np.random.randn()
        elif move_type < 0.80:
            # Set a random thickness to a random value
            idx = np.random.randint(n)
            x_new[idx] = np.random.uniform(lb[idx], ub[idx])
        elif move_type < 0.90:
            # Swap two thicknesses
            idx1, idx2 = np.random.choice(n, 2, replace=False)
            x_new[idx1], x_new[idx2] = x_new[idx2], x_new[idx1]
        else:
            # Global random restart (partial)
            n_restart = max(1, n // 4)
            idxs = np.random.choice(n, n_restart, replace=False)
            for idx in idxs:
                x_new[idx] = np.random.uniform(lb[idx], ub[idx])
        
        # Clip to bounds
        x_new = np.clip(x_new, lb, ub)
        
        f_new = objective(x_new)
        
        # Acceptance criterion
        if f_new < f_current:
            x = x_new
            f_current = f_new
            if f_new < f_best:
                x_best = x_new.copy()
                f_best = f_new
        else:
            delta_f = f_new - f_current
            if delta_f < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta_f / max(T, 1e-10))
            
            if np.random.rand() < accept_prob:
                x = x_new
                f_current = f_new
        
        T *= cooling_factor
    
    return x_best, f_best

def design(problem):
    """Design antireflection coating using DE + SA + Nelder-Mead polishing."""
    start_time = time.time()
    time_limit = 140.0  # Leave 10s safety margin for the 150s wall-clock limit
    
    L_index = problem["materials"]["L"]
    H_index = problem["materials"]["H"]
    center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    best_overall_score = float('inf')
    best_overall_materials = ["L"]
    best_overall_thicknesses = [center_wl / (4.0 * L_index)]
    
    # Try different layer counts and material patterns
    # Focus on 4-14 layers which are typically optimal for broadband AR
    layer_configs = []
    
    for n_layers in [4, 6, 8, 10, 12, 14]:
        if n_layers > max_layers:
            continue
        # Pattern 1: Start with L
        pat1 = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        # Pattern 2: Start with H
        pat2 = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        # Pattern 3: LHLH... but with some variation
        pat3 = ["L" if i % 4 < 2 else "H" for i in range(n_layers)]
        # Pattern 4: HLHL...
        pat4 = ["H" if i % 4 < 2 else "L" for i in range(n_layers)]
        
        for pat in [pat1, pat2, pat3, pat4]:
            layer_configs.append(pat)
    
    # Sort by expected quality: prefer even layer counts starting with L
    def config_priority(config):
        n = len(config)
        score = 0
        if n % 2 == 0:
            score += 1
        if config[0] == "L":
            score += 1
        # Prefer 8-12 layers
        if 8 <= n <= 12:
            score += 2
        return -score
    
    layer_configs.sort(key=config_priority)
    
    # For each configuration, run multi-start optimization
    for materials in layer_configs:
        if time.time() - start_time > time_limit - 15:
            break
        
        n_layers = len(materials)
        bounds = [(0.0, max_layer_nm)] * n_layers
        
        # Create objective function
        def objective(x, materials=materials, problem=problem, max_total_nm=max_total_nm, max_layer_nm=max_layer_nm):
            x = _project_to_feasible(x, max_total_nm, max_layer_nm)
            return _evaluate(materials, list(x), problem)
        
        # Generate multiple initial points
        n_starts = 5
        best_x_for_config = None
        best_f_for_config = float('inf')
        
        for start_i in range(n_starts):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Generate initial point
            if start_i == 0:
                # Quarter-wave initial guess
                x0 = np.array([center_wl / (4.0 * problem["materials"][m]) for m in materials])
                x0 = _project_to_feasible(x0, max_total_nm, max_layer_nm)
            elif start_i == 1:
                # Quarter-wave with some variation
                x0 = np.array([center_wl / (4.0 * problem["materials"][m]) * (1.0 + 0.3 * np.random.randn()) for m in materials])
                x0 = _project_to_feasible(x0, max_total_nm, max_layer_nm)
            elif start_i == 2:
                # Random within bounds
                x0 = np.random.uniform(0, max_layer_nm, n_layers)
                x0 = _project_to_feasible(x0, max_total_nm, max_layer_nm)
            else:
                # Perturbed quarter-wave
                x0 = np.array([center_wl / (4.0 * problem["materials"][m]) for m in materials])
                x0 += 0.2 * max_layer_nm * np.random.randn(n_layers)
                x0 = _project_to_feasible(x0, max_total_nm, max_layer_nm)
            
            # Run DE for a short time
            de_time = min(8.0, (time_limit - (time.time() - start_time)) * 0.3)
            if de_time < 2.0:
                break
            
            x_de, f_de = _differential_evolution(
                objective, bounds, pop_size=30, max_iter=200,
                F=0.8, CR=0.9, time_limit=de_time, start_time=start_time
            )
            
            # Run SA from DE result
            sa_time = min(10.0, (time_limit - (time.time() - start_time)) * 0.4)
            if sa_time < 2.0:
                break
            
            x_sa, f_sa = _simulated_annealing(
                objective, x_de, bounds, max_iter=30000,
                T0=0.5, T1=0.001, time_limit=sa_time, start_time=start_time
            )
            
            # Run Nelder-Mead polish
            nm_time = min(5.0, (time_limit - (time.time() - start_time)) * 0.3)
            if nm_time < 1.0:
                break
            
            x_nm, f_nm = _nelder_mead(
                objective, x_sa, bounds, maxiter=100, tol=1e-8,
                time_limit=nm_time, start_time=start_time
            )
            
            # Track best for this configuration
            if f_nm < best_f_for_config:
                best_f_for_config = f_nm
                best_x_for_config = x_nm
        
        # Update global best
        if best_x_for_config is not None:
            x_final = _project_to_feasible(best_x_for_config, max_total_nm, max_layer_nm)
            thicknesses_opt = [float(xi) for xi in x_final]
            
            if _is_valid(materials, thicknesses_opt, problem):
                score = _evaluate(materials, thicknesses_opt, problem)
                if score < best_overall_score:
                    best_overall_score = score
                    best_overall_materials = list(materials)
                    best_overall_thicknesses = list(thicknesses_opt)
    
    # Final refinement: run one more SA + NM from the best found
    if time.time() - start_time < time_limit - 5:
        n_layers = len(best_overall_materials)
        bounds = [(0.0, max_layer_nm)] * n_layers
        
        def objective_final(x):
            x = _project_to_feasible(x, max_total_nm, max_layer_nm)
            return _evaluate(best_overall_materials, list(x), problem)
        
        x0 = np.array(best_overall_thicknesses)
        
        # SA refinement
        sa_time = min(15.0, time_limit - (time.time() - start_time) - 3)
        if sa_time > 2.0:
            x_sa, f_sa = _simulated_annealing(
                objective_final, x0, bounds, max_iter=50000,
                T0=0.1, T1=0.0001, time_limit=sa_time, start_time=start_time
            )
            x0 = x_sa
        
        # NM polish
        nm_time = min(5.0, time_limit - (time.time() - start_time) - 1)
        if nm_time > 1.0:
            x_nm, f_nm = _nelder_mead(
                objective_final, x0, bounds, maxiter=200, tol=1e-10,
                time_limit=nm_time, start_time=start_time
            )
            x_final = _project_to_feasible(x_nm, max_total_nm, max_layer_nm)
            thicknesses_opt = [float(xi) for xi in x_final]
            
            if _is_valid(best_overall_materials, thicknesses_opt, problem):
                score = _evaluate(best_overall_materials, thicknesses_opt, problem)
                if score < best_overall_score:
                    best_overall_score = score
                    best_overall_thicknesses = list(thicknesses_opt)
    
    # Fallback
    if not _is_valid(best_overall_materials, best_overall_thicknesses, problem):
        best_overall_materials = ["L"]
        best_overall_thicknesses = [center_wl / (4.0 * L_index)]
    
    return best_overall_materials, best_overall_thicknesses
# EVOLVE-BLOCK-END
