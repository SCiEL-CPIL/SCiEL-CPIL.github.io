"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _char_matrix_method(wavelengths, thicknesses, indices, ambient_n, substrate_n):
    """Compute reflectance using characteristic matrix method."""
    n_layers = len(thicknesses)
    if n_layers == 0:
        # Bare substrate
        r0 = (ambient_n - substrate_n) / (ambient_n + substrate_n)
        return np.full(len(wavelengths), np.abs(r0) ** 2)
    
    reflectance = np.zeros(len(wavelengths))
    
    for i, wl in enumerate(wavelengths):
        # Start with identity matrix
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        for j in range(n_layers):
            n_j = indices[j]
            d_j = thicknesses[j]
            delta_j = 2.0 * np.pi * n_j * d_j / wl
            
            cos_d = np.cos(delta_j)
            sin_d = np.sin(delta_j)
            
            # Handle sin_d near zero to avoid division by zero
            if abs(sin_d) < 1e-15:
                sin_d = 1e-15 if sin_d >= 0 else -1e-15
            
            M_j = np.array([
                [cos_d, 1j * sin_d / n_j],
                [1j * n_j * sin_d, cos_d]
            ], dtype=complex)
            
            M = M @ M_j
        
        n_s = substrate_n
        n_0 = ambient_n
        
        B = M[0, 0] + M[0, 1] * n_s
        C = M[1, 0] * n_0 + M[1, 1] * n_s
        
        if abs(C + B) < 1e-15:
            reflectance[i] = 1.0
        else:
            r = (C - B) / (C + B)
            reflectance[i] = np.abs(r) ** 2
    
    return reflectance

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance in percent over the band."""
    wavelengths = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    indices = [problem["materials"][m] for m in materials]
    refl = _char_matrix_method(wavelengths, np.array(thicknesses, dtype=float), np.array(indices, dtype=float),
                               problem["ambient_index"], problem["substrate_index"])
    return 100.0 * np.mean(refl)

def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if any(t < 0 or t > problem["max_layer_nm"] for t in thicknesses):
        return False
    if sum(thicknesses) > problem["max_total_nm"]:
        return False
    if any(m not in problem["materials"] for m in materials):
        return False
    return True

def _nelder_mead(materials, thicknesses, problem, time_limit, max_iter=200):
    """Optimize thicknesses using Nelder-Mead simplex method."""
    start_time = time.time()
    n_layers = len(thicknesses)
    if n_layers == 0:
        return list(materials), list(thicknesses), _mean_reflectance(materials, thicknesses, problem)
    
    # Initial simplex
    alpha = 1.0  # expansion
    gamma = 2.0  # expansion factor
    rho = 0.5    # reflection
    sigma = 0.5  # contraction
    
    # Start with current point
    best_thicknesses = list(thicknesses)
    best_score = _mean_reflectance(materials, best_thicknesses, problem)
    
    # Create simplex vertices
    simplex = [list(best_thicknesses)]
    simplex_scores = [best_score]
    
    # Add perturbed vertices
    for i in range(n_layers):
        perturbed = list(best_thicknesses)
        if perturbed[i] > 10:
            perturbed[i] *= 1.1
        else:
            perturbed[i] += 10
        # Clamp
        perturbed[i] = max(0, min(perturbed[i], problem["max_layer_nm"]))
        if sum(perturbed) <= problem["max_total_nm"]:
            simplex.append(perturbed)
            simplex_scores.append(_mean_reflectance(materials, perturbed, problem))
        else:
            simplex.append(list(best_thicknesses))
            simplex_scores.append(best_score)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Sort by score
        order = np.argsort(simplex_scores)
        simplex = [simplex[i] for i in order]
        simplex_scores = [simplex_scores[i] for i in order]
        
        # Check convergence
        if abs(simplex_scores[0] - simplex_scores[-1]) < 1e-8:
            break
        
        # Centroid of all but worst point
        centroid = [0.0] * n_layers
        for i in range(len(simplex) - 1):
            for j in range(n_layers):
                centroid[j] += simplex[i][j] / (len(simplex) - 1)
        
        # Reflection
        reflected = [centroid[j] + rho * (centroid[j] - simplex[-1][j]) for j in range(n_layers)]
        reflected = [max(0, min(t, problem["max_layer_nm"])) for t in reflected]
        
        if sum(reflected) <= problem["max_total_nm"]:
            reflected_score = _mean_reflectance(materials, reflected, problem)
        else:
            reflected_score = float('inf')
        
        if reflected_score < simplex_scores[0]:
            # Try expansion
            expanded = [centroid[j] + gamma * (reflected[j] - centroid[j]) for j in range(n_layers)]
            expanded = [max(0, min(t, problem["max_layer_nm"])) for t in expanded]
            
            if sum(expanded) <= problem["max_total_nm"]:
                expanded_score = _mean_reflectance(materials, expanded, problem)
            else:
                expanded_score = float('inf')
            
            if expanded_score < reflected_score:
                simplex[-1] = expanded
                simplex_scores[-1] = expanded_score
            else:
                simplex[-1] = reflected
                simplex_scores[-1] = reflected_score
        elif reflected_score < simplex_scores[-2]:
            simplex[-1] = reflected
            simplex_scores[-1] = reflected_score
        else:
            # Contraction
            contracted = [centroid[j] + sigma * (simplex[-1][j] - centroid[j]) for j in range(n_layers)]
            contracted = [max(0, min(t, problem["max_layer_nm"])) for t in contracted]
            
            if sum(contracted) <= problem["max_total_nm"]:
                contracted_score = _mean_reflectance(materials, contracted, problem)
            else:
                contracted_score = float('inf')
            
            if contracted_score < simplex_scores[-1]:
                simplex[-1] = contracted
                simplex_scores[-1] = contracted_score
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n_layers)]
                    simplex[i] = [max(0, min(t, problem["max_layer_nm"])) for t in simplex[i]]
                    if sum(simplex[i]) <= problem["max_total_nm"]:
                        simplex_scores[i] = _mean_reflectance(materials, simplex[i], problem)
                    else:
                        simplex_scores[i] = float('inf')
        
        # Track best
        min_idx = np.argmin(simplex_scores)
        if simplex_scores[min_idx] < best_score - 1e-10:
            best_score = simplex_scores[min_idx]
            best_thicknesses = list(simplex[min_idx])
    
    return list(materials), best_thicknesses, best_score

def _polish_thicknesses(materials, thicknesses, problem, time_limit):
    """Final high-precision polish using fine coordinate descent."""
    start_time = time.time()
    current_score = _mean_reflectance(materials, thicknesses, problem)
    best_materials = list(materials)
    best_thicknesses = list(thicknesses)
    best_score = current_score
    
    n_layers = len(thicknesses)
    if n_layers == 0:
        return best_materials, best_thicknesses, best_score
    
    # Very fine step sizes for polishing
    step_sizes = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05]
    
    for step in step_sizes:
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3:
                break
            improved = False
            for i in range(n_layers):
                if time.time() - start_time > time_limit - 3:
                    break
                for delta in [step, -step]:
                    new_thickness = thicknesses[i] + delta
                    if new_thickness < 0 or new_thickness > problem["max_layer_nm"]:
                        continue
                    new_thicknesses = list(thicknesses)
                    new_thicknesses[i] = new_thickness
                    if sum(new_thicknesses) > problem["max_total_nm"]:
                        continue
                    
                    new_score = _mean_reflectance(materials, new_thicknesses, problem)
                    if new_score < current_score - 1e-10:
                        current_score = new_score
                        thicknesses = new_thicknesses
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_materials = list(materials)
                            best_thicknesses = list(thicknesses)
    
    return best_materials, best_thicknesses, best_score

def _simulated_annealing(materials, thicknesses, problem, time_limit, max_iter=500):
    """Simulated annealing to explore variable-length, non-periodic layer sequences."""
    start_time = time.time()
    best_materials = list(materials)
    best_thicknesses = list(thicknesses)
    best_score = _mean_reflectance(best_materials, best_thicknesses, problem)
    
    current_materials = list(materials)
    current_thicknesses = list(thicknesses)
    current_score = best_score
    
    T = 5.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    
    center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Generate neighbor
        new_materials = list(current_materials)
        new_thicknesses = list(current_thicknesses)
        
        action = random.random()
        
        if action < 0.3 and len(new_materials) < problem["max_layers"]:
            # Add a layer
            pos = random.randint(0, len(new_materials))
            mat = random.choice(["L", "H"])
            n_idx = problem["materials"][mat]
            # Random thickness around quarter-wave
            base_thick = center_wl / (4.0 * n_idx)
            thick = base_thick * random.uniform(0.5, 1.5)
            thick = max(0, min(thick, problem["max_layer_nm"]))
            
            if sum(new_thicknesses) + thick <= problem["max_total_nm"]:
                new_materials.insert(pos, mat)
                new_thicknesses.insert(pos, thick)
            else:
                continue
        elif action < 0.5 and len(new_materials) > 1:
            # Remove a layer
            pos = random.randint(0, len(new_materials) - 1)
            new_materials.pop(pos)
            new_thicknesses.pop(pos)
        elif action < 0.8:
            # Perturb a thickness
            if len(new_thicknesses) == 0:
                continue
            pos = random.randint(0, len(new_thicknesses) - 1)
            perturbation = random.gauss(0, 20)
            new_thick = new_thicknesses[pos] + perturbation
            new_thick = max(0, min(new_thick, problem["max_layer_nm"]))
            
            if sum(new_thicknesses) - new_thicknesses[pos] + new_thick <= problem["max_total_nm"]:
                new_thicknesses[pos] = new_thick
            else:
                continue
        else:
            # Swap two layers
            if len(new_materials) >= 2:
                i, j = random.sample(range(len(new_materials)), 2)
                new_materials[i], new_materials[j] = new_materials[j], new_materials[i]
                new_thicknesses[i], new_thicknesses[j] = new_thicknesses[j], new_thicknesses[i]
        
        # Evaluate
        new_score = _mean_reflectance(new_materials, new_thicknesses, problem)
        
        # Accept or reject
        if new_score < current_score:
            current_materials = new_materials
            current_thicknesses = new_thicknesses
            current_score = new_score
            
            if current_score < best_score - 1e-10:
                best_score = current_score
                best_materials = list(current_materials)
                best_thicknesses = list(current_thicknesses)
        else:
            delta = new_score - current_score
            if T > T_min:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_materials = new_materials
                    current_thicknesses = new_thicknesses
                    current_score = new_score
        
        T *= cooling_rate
    
    return best_materials, best_thicknesses, best_score

def design(problem):
    """Design antireflection coating using greedy layer addition, Nelder-Mead, and simulated annealing."""
    start_time = time.time()
    time_limit = 140.0  # Leave 10s safety margin
    
    L_index = problem["materials"]["L"]
    H_index = problem["materials"]["H"]
    center_wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    
    # Start with a good initial guess: alternating L-H quarter-wave stack
    # For broadband AR, start with a few layers
    
    best_materials = ["L"]
    best_thicknesses = [center_wl / (4.0 * L_index)]
    best_score = _mean_reflectance(best_materials, best_thicknesses, problem)
    
    # Try adding layers greedily
    current_materials = list(best_materials)
    current_thicknesses = list(best_thicknesses)
    current_score = best_score
    
    for _ in range(problem["max_layers"] - 1):
        if time.time() - start_time > time_limit - 30:
            break
        
        best_new_score = current_score
        best_new_materials = list(current_materials)
        best_new_thicknesses = list(current_thicknesses)
        
        # Try adding L or H at the end
        for new_mat in ["L", "H"]:
            new_materials = current_materials + [new_mat]
            n_idx = problem["materials"][new_mat]
            
            # Try several thicknesses
            for frac in [0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]:
                new_thickness = center_wl * frac / n_idx
                if new_thickness > problem["max_layer_nm"]:
                    continue
                if sum(current_thicknesses) + new_thickness > problem["max_total_nm"]:
                    continue
                
                new_thicknesses = current_thicknesses + [new_thickness]
                new_score = _mean_reflectance(new_materials, new_thicknesses, problem)
                
                if new_score < best_new_score - 1e-10:
                    best_new_score = new_score
                    best_new_materials = new_materials
                    best_new_thicknesses = new_thicknesses
        
        if best_new_score < current_score - 1e-10:
            current_materials = best_new_materials
            current_thicknesses = best_new_thicknesses
            current_score = best_new_score
            if current_score < best_score:
                best_score = current_score
                best_materials = list(current_materials)
                best_thicknesses = list(current_thicknesses)
        else:
            # Try inserting in the middle as well
            best_insert_score = current_score
            best_insert_materials = list(current_materials)
            best_insert_thicknesses = list(current_thicknesses)
            
            for insert_pos in range(1, len(current_materials)):
                for new_mat in ["L", "H"]:
                    new_materials = current_materials[:insert_pos] + [new_mat] + current_materials[insert_pos:]
                    n_idx = problem["materials"][new_mat]
                    
                    for frac in [0.25, 0.35, 0.5, 0.65, 0.75]:
                        new_thickness = center_wl * frac / n_idx
                        if new_thickness > problem["max_layer_nm"]:
                            continue
                        if sum(current_thicknesses) + new_thickness > problem["max_total_nm"]:
                            continue
                        
                        new_thicknesses = current_thicknesses[:insert_pos] + [new_thickness] + current_thicknesses[insert_pos:]
                        new_score = _mean_reflectance(new_materials, new_thicknesses, problem)
                        
                        if new_score < best_insert_score - 1e-10:
                            best_insert_score = new_score
                            best_insert_materials = new_materials
                            best_insert_thicknesses = new_thicknesses
            
            if best_insert_score < current_score - 1e-10:
                current_materials = best_insert_materials
                current_thicknesses = best_insert_thicknesses
                current_score = best_insert_score
                if current_score < best_score:
                    best_score = current_score
                    best_materials = list(current_materials)
                    best_thicknesses = list(current_thicknesses)
            else:
                break
    
    # Nelder-Mead optimization on the current best
    if time.time() - start_time < time_limit - 40:
        nm_materials, nm_thicknesses, nm_score = _nelder_mead(
            best_materials, best_thicknesses, problem, time_limit - 40
        )
        if nm_score < best_score:
            best_materials = nm_materials
            best_thicknesses = nm_thicknesses
            best_score = nm_score
    
    # Simulated annealing to explore different layer structures
    if time.time() - start_time < time_limit - 20:
        sa_materials, sa_thicknesses, sa_score = _simulated_annealing(
            best_materials, best_thicknesses, problem, time_limit - 20
        )
        if sa_score < best_score:
            best_materials = sa_materials
            best_thicknesses = sa_thicknesses
            best_score = sa_score
    
    # Final high-precision polish
    if time.time() - start_time < time_limit - 5:
        best_materials, best_thicknesses, best_score = _polish_thicknesses(
            best_materials, best_thicknesses, problem, time_limit - 5
        )
    
    # Ensure validity
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [center_wl / (4.0 * L_index)]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
