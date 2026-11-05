"""Broadband antireflection coating design via vectorized Nelder-Mead optimization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _compute_reflectance_vectorized(materials, thicknesses, wavelengths, n_ambient, n_substrate, n_L, n_H):
    """Compute reflectance at all wavelengths using vectorized characteristic matrix method."""
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wavelengths), r * r)
    
    # Precompute beta for all layers and wavelengths
    # beta[i, j] = 2*pi * n_i * d_i / wl_j
    betas = np.zeros((n_layers, len(wavelengths)))
    for i in range(n_layers):
        n = n_L if materials[i] == 'L' else n_H
        d = thicknesses[i]
        betas[i, :] = 2.0 * math.pi * n * d / wavelengths
    
    cos_b = np.cos(betas)
    sin_b = np.sin(betas)
    
    # Initialize M matrices as arrays of shape (2, 2, len(wavelengths))
    # M11, M12, M21, M22 each of shape (len(wavelengths),)
    M11 = np.ones(len(wavelengths), dtype=complex)
    M12 = np.zeros(len(wavelengths), dtype=complex)
    M21 = np.zeros(len(wavelengths), dtype=complex)
    M22 = np.ones(len(wavelengths), dtype=complex)
    
    for i in range(n_layers):
        n = n_L if materials[i] == 'L' else n_H
        # L11 = cos_b[i], L12 = 1j*sin_b[i]/n, L21 = 1j*n*sin_b[i], L22 = cos_b[i]
        L11 = cos_b[i]
        L12 = 1j * sin_b[i] / n
        L21 = 1j * n * sin_b[i]
        L22 = cos_b[i]
        
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        
        M11, M12, M21, M22 = new_M11, new_M12, new_M21, new_M22
    
    B = M11 + M12 * n_substrate
    C = M21 + M22 * n_substrate
    Y = C / B
    r = (n_ambient - Y) / (n_ambient + Y)
    return np.abs(r) ** 2

def _mean_reflectance_percent(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    
    R = _compute_reflectance_vectorized(materials, thicknesses, wavelengths, n_ambient, n_substrate, n_L, n_H)
    return 100.0 * np.mean(R)

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    if sum(thicknesses) > max_total_nm + 1e-9:
        return False
    return True

def _clamp_thicknesses(thicknesses, max_layer_nm, max_total_nm):
    """Clamp thicknesses to valid range and ensure total doesn't exceed max."""
    clamped = np.array([max(0.0, min(t, max_layer_nm)) for t in thicknesses])
    total = np.sum(clamped)
    if total > max_total_nm:
        if total > 0:
            clamped = clamped * (max_total_nm / total)
        # Re-clamp after scaling
        clamped = np.array([max(0.0, min(t, max_layer_nm)) for t in clamped])
    return clamped

def _evaluate_design(materials, thicknesses, problem):
    """Evaluate the mean reflectance percent for a design. Returns inf if invalid."""
    if not _is_valid(materials, thicknesses, problem):
        return float('inf')
    return _mean_reflectance_percent(materials, thicknesses, problem)

def _nelder_mead_optimize(materials, thicknesses, problem, deadline, max_iter=500, tol=1e-6):
    """Nelder-Mead optimization of layer thicknesses for a fixed set of materials."""
    n_layers = len(thicknesses)
    if n_layers == 0:
        return materials, thicknesses, _evaluate_design(materials, thicknesses, problem)
    
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    # Initialize simplex
    x0 = np.array(thicknesses, dtype=float)
    x0 = _clamp_thicknesses(x0, max_layer_nm, max_total_nm)
    
    # Create simplex
    simplex = [x0.copy()]
    for i in range(n_layers):
        xi = x0.copy()
        step = max(10.0, 0.1 * x0[i])
        xi[i] += step
        xi = _clamp_thicknesses(xi, max_layer_nm, max_total_nm)
        simplex.append(xi)
    
    # Evaluate initial simplex
    scores = []
    for x in simplex:
        s = _evaluate_design(materials, x.tolist(), problem)
        scores.append(s)
    
    # Nelder-Mead iteration
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Sort simplex by score
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Check convergence
        if abs(scores[-1] - scores[0]) < tol * max(1.0, abs(scores[0])):
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = _clamp_thicknesses(xr, max_layer_nm, max_total_nm)
        fr = _evaluate_design(materials, xr.tolist(), problem)
        
        if fr < scores[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            xe = _clamp_thicknesses(xe, max_layer_nm, max_total_nm)
            fe = _evaluate_design(materials, xe.tolist(), problem)
            if fe < fr:
                simplex[-1] = xe
                scores[-1] = fe
            else:
                simplex[-1] = xr
                scores[-1] = fr
        elif fr < scores[-2]:
            simplex[-1] = xr
            scores[-1] = fr
        else:
            # Contraction
            if fr < scores[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (centroid - simplex[-1])
            xc = _clamp_thicknesses(xc, max_layer_nm, max_total_nm)
            fc = _evaluate_design(materials, xc.tolist(), problem)
            
            if fc < scores[-1]:
                simplex[-1] = xc
                scores[-1] = fc
            else:
                # Shrink
                for i in range(1, n_layers + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = _clamp_thicknesses(simplex[i], max_layer_nm, max_total_nm)
                    scores[i] = _evaluate_design(materials, simplex[i].tolist(), problem)
    
    # Find best
    best_idx = np.argmin(scores)
    best_x = simplex[best_idx].tolist()
    best_score = scores[best_idx]
    
    return materials, best_x, best_score

def _generate_candidates(problem, deadline):
    """Generate diverse initial candidate designs."""
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    candidates = []
    
    # Quarter-wave designs with varying number of layers
    for n_layers in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
        if n_layers > problem.get("max_layers", 20):
            continue
        
        # Alternating starting with L
        materials = []
        for i in range(n_layers):
            materials.append('L' if i % 2 == 0 else 'H')
        
        # Quarter-wave thicknesses
        thicknesses = []
        for mat in materials:
            n = n_L if mat == 'L' else n_H
            d = centre / (4.0 * n)
            thicknesses.append(d)
        
        # Check total thickness
        if sum(thicknesses) <= max_total_nm:
            candidates.append((materials, thicknesses))
        
        # Alternating starting with H
        materials = []
        for i in range(n_layers):
            materials.append('H' if i % 2 == 0 else 'L')
        
        thicknesses = []
        for mat in materials:
            n = n_L if mat == 'L' else n_H
            d = centre / (4.0 * n)
            thicknesses.append(d)
        
        if sum(thicknesses) <= max_total_nm:
            candidates.append((materials, thicknesses))
    
    # Random designs
    np.random.seed(42)
    for _ in range(20):
        n_layers = np.random.randint(1, 13)
        materials = []
        thicknesses = []
        total = 0.0
        for i in range(n_layers):
            mat = 'L' if np.random.random() < 0.5 else 'H'
            materials.append(mat)
            d = np.random.uniform(0, max_layer_nm)
            # Ensure total doesn't exceed max
            if total + d > max_total_nm:
                d = max(0, max_total_nm - total)
            thicknesses.append(d)
            total += d
        candidates.append((materials, thicknesses))
    
    # Homogeneous designs (all L or all H)
    for mat in ['L', 'H']:
        n = n_L if mat == 'L' else n_H
        d = max_total_nm / 4.0 / n * 0.8  # Use 80% of max total
        if d <= max_layer_nm:
            candidates.append(([mat], [d]))
    
    return candidates

def _try_layer_addition(materials, thicknesses, problem, deadline):
    """Try adding a layer at various positions and optimize."""
    max_layers = problem.get("max_layers", 20)
    if len(materials) >= max_layers:
        return None
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    max_total_nm = problem.get("max_total_nm", 1000)
    
    current_total = sum(thicknesses)
    candidates = []
    
    for pos in range(len(materials) + 1):
        for mat in ['L', 'H']:
            n = n_L if mat == 'L' else n_H
            d = centre / (4.0 * n)
            
            # Check if we have room
            if current_total + d > max_total_nm:
                continue
            
            new_materials = materials[:pos] + [mat] + materials[pos:]
            new_thicknesses = thicknesses[:pos] + [d] + thicknesses[pos:]
            
            if _is_valid(new_materials, new_thicknesses, problem):
                score = _evaluate_design(new_materials, new_thicknesses, problem)
                if score < float('inf'):
                    candidates.append((score, new_materials, new_thicknesses))
    
    if not candidates:
        return None
    
    candidates.sort(key=lambda x: x[0])
    best_score, best_materials, best_thicknesses = candidates[0]
    
    # Local optimize
    best_materials, best_thicknesses, best_score = _nelder_mead_optimize(
        best_materials, best_thicknesses, problem, deadline, max_iter=100)
    
    return best_materials, best_thicknesses, best_score

def _try_layer_removal(materials, thicknesses, problem, deadline):
    """Try removing a layer and optimizing."""
    if len(materials) <= 1:
        return None
    
    candidates = []
    for i in range(len(materials)):
        new_materials = materials[:i] + materials[i+1:]
        new_thicknesses = thicknesses[:i] + thicknesses[i+1:]
        
        if _is_valid(new_materials, new_thicknesses, problem):
            score = _evaluate_design(new_materials, new_thicknesses, problem)
            if score < float('inf'):
                candidates.append((score, new_materials, new_thicknesses))
    
    if not candidates:
        return None
    
    candidates.sort(key=lambda x: x[0])
    best_score, best_materials, best_thicknesses = candidates[0]
    
    best_materials, best_thicknesses, best_score = _nelder_mead_optimize(
        best_materials, best_thicknesses, problem, deadline, max_iter=100)
    
    return best_materials, best_thicknesses, best_score

def _genetic_algorithm(materials, thicknesses, problem, deadline, pop_size=20, n_generations=30):
    """Simple genetic algorithm to explore design space."""
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    # Initialize population
    population = []
    scores = []
    
    # Add current best
    population.append((list(materials), list(thicknesses)))
    scores.append(_evaluate_design(materials, thicknesses, problem))
    
    # Add random individuals
    np.random.seed(int(time.time() * 1000) % 100000)
    while len(population) < pop_size:
        n_layers = np.random.randint(1, min(max_layers + 1, 13))
        mats = []
        ths = []
        total = 0.0
        for i in range(n_layers):
            mat = 'L' if np.random.random() < 0.5 else 'H'
            mats.append(mat)
            d = np.random.uniform(0, max_layer_nm)
            if total + d > max_total_nm:
                d = max(0, max_total_nm - total)
            ths.append(d)
            total += d
        population.append((mats, ths))
        scores.append(_evaluate_design(mats, ths, problem))
    
    for gen in range(n_generations):
        if time.time() > deadline - 10.0:
            break
        
        # Sort by score
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Elitism: keep best 2
        new_population = [population[0], population[1]]
        new_scores = [scores[0], scores[1]]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection: tournament
            def tournament():
                idxs = np.random.choice(len(population), size=3, replace=False)
                best_idx = min(idxs, key=lambda i: scores[i])
                return population[best_idx]
            
            parent1 = tournament()
            parent2 = tournament()
            
            # Crossover: combine materials and thicknesses
            if len(parent1[0]) != len(parent2[0]):
                # Different lengths: take shorter and add random
                min_len = min(len(parent1[0]), len(parent2[0]))
                child_mats = []
                child_ths = []
                for i in range(min_len):
                    if np.random.random() < 0.5:
                        child_mats.append(parent1[0][i])
                        child_ths.append(parent1[1][i])
                    else:
                        child_mats.append(parent2[0][i])
                        child_ths.append(parent2[1][i])
                # Add remaining from longer parent
                if len(parent1[0]) > min_len:
                    for i in range(min_len, len(parent1[0])):
                        child_mats.append(parent1[0][i])
                        child_ths.append(parent1[1][i])
                elif len(parent2[0]) > min_len:
                    for i in range(min_len, len(parent2[0])):
                        child_mats.append(parent2[0][i])
                        child_ths.append(parent2[1][i])
            else:
                child_mats = []
                child_ths = []
                for i in range(len(parent1[0])):
                    if np.random.random() < 0.5:
                        child_mats.append(parent1[0][i])
                        child_ths.append(parent1[1][i])
                    else:
                        child_mats.append(parent2[0][i])
                        child_ths.append(parent2[1][i])
            
            # Mutation
            if np.random.random() < 0.3:
                # Add a layer
                if len(child_mats) < max_layers:
                    pos = np.random.randint(0, len(child_mats) + 1)
                    mat = 'L' if np.random.random() < 0.5 else 'H'
                    n = n_L if mat == 'L' else n_H
                    d = centre / (4.0 * n)
                    child_mats.insert(pos, mat)
                    child_ths.insert(pos, d)
            
            if np.random.random() < 0.3:
                # Remove a layer
                if len(child_mats) > 1:
                    pos = np.random.randint(0, len(child_mats))
                    child_mats.pop(pos)
                    child_ths.pop(pos)
            
            # Mutate thicknesses
            for i in range(len(child_ths)):
                if np.random.random() < 0.3:
                    child_ths[i] = max(0, min(child_ths[i] + np.random.uniform(-50, 50), max_layer_nm))
            
            # Clamp total
            total = sum(child_ths)
            if total > max_total_nm:
                if total > 0:
                    scale = max_total_nm / total
                    child_ths = [t * scale for t in child_ths]
                    child_ths = [max(0, min(t, max_layer_nm)) for t in child_ths]
            
            if _is_valid(child_mats, child_ths, problem):
                new_population.append((child_mats, child_ths))
                new_scores.append(_evaluate_design(child_mats, child_ths, problem))
        
        population = new_population
        scores = new_scores
    
    # Find best
    best_idx = np.argmin(scores)
    best_materials, best_thicknesses = population[best_idx]
    best_score = scores[best_idx]
    
    return best_materials, best_thicknesses, best_score

def design(problem):
    """Design a broadband antireflection coating."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Generate diverse initial candidates
    candidates = _generate_candidates(problem, deadline)
    
    # Evaluate all candidates
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    for mats, ths in candidates:
        if time.time() > deadline - 20.0:
            break
        score = _evaluate_design(mats, ths, problem)
        if score < best_score:
            best_materials = mats
            best_thicknesses = ths
            best_score = score
    
    if best_materials is None:
        # Fallback: single quarter-wave L layer
        n_L = problem["materials"]["L"]
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
        best_score = _evaluate_design(best_materials, best_thicknesses, problem)
    
    # Run genetic algorithm to explore broader space
    if time.time() < deadline - 30.0:
        ga_mats, ga_ths, ga_score = _genetic_algorithm(
            best_materials, best_thicknesses, problem, deadline - 15.0,
            pop_size=15, n_generations=20)
        if ga_score < best_score:
            best_materials, best_thicknesses, best_score = ga_mats, ga_ths, ga_score
    
    # Iteratively improve with Nelder-Mead and structural changes
    for iteration in range(30):
        if time.time() > deadline - 15.0:
            break
        
        # Nelder-Mead optimize current best
        best_materials, best_thicknesses, best_score = _nelder_mead_optimize(
            best_materials, best_thicknesses, problem, deadline - 10.0, max_iter=200)
        
        # Try adding a layer
        if time.time() > deadline - 15.0:
            break
        result = _try_layer_addition(best_materials, best_thicknesses, problem, deadline - 10.0)
        if result is not None:
            mat, th, score = result
            if score < best_score:
                best_materials, best_thicknesses, best_score = mat, th, score
        
        # Try removing a layer
        if time.time() > deadline - 15.0:
            break
        result = _try_layer_removal(best_materials, best_thicknesses, problem, deadline - 10.0)
        if result is not None:
            mat, th, score = result
            if score < best_score:
                best_materials, best_thicknesses, best_score = mat, th, score
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        best_materials, best_thicknesses, best_score = _nelder_mead_optimize(
            best_materials, best_thicknesses, problem, deadline - 2.0, max_iter=300, tol=1e-8)
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
