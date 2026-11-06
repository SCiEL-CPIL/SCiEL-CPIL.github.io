"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_reflectance(materials, thicknesses, n_sub, n_air, wavelengths):
    """Compute reflectance using characteristic matrix method."""
    n = len(materials)
    if n == 0:
        return np.zeros(len(wavelengths))
    
    k = 2 * np.pi / wavelengths
    # Build the matrix for each wavelength
    # M = product of layer matrices
    # For a layer with index n_i and thickness d_i, optical thickness is n_i * d_i
    # Phase delta_i = k * n_i * d_i
    
    # We can vectorize over wavelengths
    # Matrix M_j = [[cos(delta_j), i/sin(delta_j)], [i*n_j*sin(delta_j), cos(delta_j)]]
    
    # Initialize total matrix M_tot = I for each wavelength
    # M_tot shape: (num_wavelengths, 2, 2)
    M_tot = np.zeros((len(wavelengths), 2, 2), dtype=np.complex128)
    M_tot[:, 0, 0] = 1
    M_tot[:, 1, 1] = 1
    
    for i in range(n):
        n_i = materials[i]
        d_i = thicknesses[i]
        delta = k * n_i * d_i
        
        c = np.cos(delta)
        s = np.sin(delta)
        
        # M_j
        M_j = np.zeros((len(wavelengths), 2, 2), dtype=np.complex128)
        M_j[:, 0, 0] = c
        M_j[:, 0, 1] = 1j / s
        M_j[:, 1, 0] = 1j * n_i * s
        M_j[:, 1, 1] = c
        
        # M_tot = M_tot @ M_j
        M_tot = M_tot @ M_j
        
    # Admittance Y = n_air * M11 + i * n_air * M12 ? No.
    # Boundary conditions:
    # E_s = M11 * E_sub + i * M12 * H_sub
    # H_s = i * M21 * E_sub / n_air? No.
    # Let's use the standard formula.
    # Y = (n_air * M11 + i * M12) / (i * n_air * M21 + M22) ?
    # Actually, the admittance Y is given by:
    # Y = (n_air * M11 + i * M12) / (i * n_air * M21 + M22) is incorrect dimensions.
    
    # Standard characteristic matrix method:
    # [E]   [M11 M12] [E]
    # [H] = [M21 M22] [H]
    # At the substrate interface:
    # H_sub = n_sub * E_sub (for normal incidence, H = n E)
    # So:
    # E_s = M11 E_sub + M12 H_sub = (M11 + M12 n_sub) E_sub
    # H_s = M21 E_sub + M22 H_sub = (M21 + M22 n_sub) E_sub
    # Therefore:
    # Y = H_s / E_s = (M21 + M22 n_sub) / (M11 + M12 n_sub)
    
    M11 = M_tot[:, 0, 0]
    M12 = M_tot[:, 0, 1]
    M21 = M_tot[:, 1, 0]
    M22 = M_tot[:, 1, 1]
    
    Y = (M21 + M22 * n_sub) / (M11 + M12 * n_sub)
    
    # Reflectance r = (n_air - Y) / (n_air + Y)
    r = (n_air - Y) / (n_air + Y)
    R = np.abs(r)**2
    return R

def _evaluate_candidate(materials_seq, thickness_indices, problem, wavelengths):
    """Evaluate a candidate design. Returns mean reflectance."""
    n_air = problem.get("ambient_index", 1.0)
    n_sub = problem.get("substrate_index", 1.52)
    mat_map = problem["materials"]
    
    # Map indices to materials and thicknesses
    # We use a fixed set of discrete thicknesses
    # For simplicity in this GA, let's use a small set of discrete thicknesses
    # e.g., multiples of 50 nm up to 400 nm? Or quarter waves?
    # The directive says: "discrete, combinatorial search over a fixed set of quarter-wave optical thicknesses"
    # Let's define a set of discrete optical thicknesses (n*d).
    # Common ones: 100, 150, 200, 250, 300, 350, 400 nm?
    # Or maybe just 50-400 nm in steps of 25 nm?
    # Let's stick to the "quarter-wave" hint.
    # Center wavelength 600 nm.
    # Quarter wave: 150 nm optical thickness.
    # Let's use a set of optical thicknesses: [100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400]
    # This gives 13 discrete options.
    
    discrete_ot = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400])
    
    materials_list = []
    thicknesses_list = []
    
    for mat_label, ot_idx in zip(materials_seq, thickness_indices):
        n_mat = mat_map[mat_label]
        ot = discrete_ot[ot_idx]
        d = ot / n_mat
        if d > problem.get("max_layer_nm", 400):
            return 100.0 # Invalid
        materials_list.append(n_mat)
        thicknesses_list.append(d)
        
    if len(materials_list) > problem.get("max_layers", 20):
        return 100.0
    if sum(thicknesses_list) > problem.get("max_total_nm", 1000):
        return 100.0
        
    R = _compute_reflectance(materials_list, thicknesses_list, n_sub, n_air, wavelengths)
    return np.mean(R) * 100.0

def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    n_air = problem.get("ambient_index", 1.0)
    n_sub = problem.get("substrate_index", 1.52)
    mat_map = problem["materials"]
    band = problem["band_nm"]
    wavelengths = np.linspace(band[0], band[1], 401)
    
    max_layers = problem.get("max_layers", 20)
    max_total = problem.get("max_total_nm", 1000)
    max_layer = problem.get("max_layer_nm", 400)
    
    # Discrete optical thicknesses
    discrete_ot = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400])
    num_ot = len(discrete_ot)
    
    # Genetic Algorithm
    pop_size = 100
    num_generations = 1000
    
    # Initialize population
    # Each individual: (materials_seq, thickness_indices)
    # materials_seq: list of 'L', 'H'
    # thickness_indices: list of int indices into discrete_ot
    
    def random_individual():
        n_layers = np.random.randint(1, max_layers + 1)
        mats = np.random.choice(['L', 'H'], size=n_layers)
        # Bias towards quarter-wave (index 2 is 150nm)
        ot_indices = np.random.choice(num_ot, size=n_layers, p=None)
        # Check validity
        total_ot = sum(discrete_ot[ot_indices])
        # Max physical thickness constraint: sum(d) <= max_total
        # d = ot / n. Max d is when n is min (1.38).
        # So sum(ot / n) <= max_total
        # Rough check: sum(ot) / 1.38 <= max_total => sum(ot) <= max_total * 1.38
        if sum(discrete_ot[ot_indices]) / 1.38 > max_total:
             return None
        # Max layer thickness: ot / n <= max_layer => ot <= max_layer * n
        # Min n is 1.38. So ot <= 400 * 1.38 = 552. All our OTs are <= 400, so this is always valid.
        return (list(mats), list(ot_indices))
    
    population = []
    for _ in range(pop_size):
        ind = random_individual()
        if ind:
            population.append(ind)
    
    # Evaluate initial population
    scores = []
    for ind in population:
        s = _evaluate_candidate(ind[0], ind[1], problem, wavelengths)
        scores.append(s)
    
    best_score = min(scores)
    best_ind = population[np.argmin(scores)]
    
    # GA Loop
    for gen in range(num_generations):
        if time.time() > deadline:
            break
            
        # Sort by score
        indices = np.argsort(scores)
        population = [population[i] for i in indices]
        scores = [scores[i] for i in indices]
        
        best_score = scores[0]
        best_ind = population[0]
        
        # Selection (Tournament)
        def select():
            i1 = np.random.randint(0, pop_size)
            i2 = np.random.randint(0, pop_size)
            if scores[i1] < scores[i2]:
                return population[i1]
            else:
                return population[i2]
        
        new_population = []
        
        # Elitism
        new_population.append(best_ind)
        
        while len(new_population) < pop_size:
            parent1 = select()
            parent2 = select()
            
            # Crossover
            # If lengths differ, truncate or pad
            len1 = len(parent1[0])
            len2 = len(parent2[0])
            max_len = max(len1, len2)
            
            # Random crossover point
            cp = np.random.randint(1, max_len)
            
            child_mats = []
            child_ots = []
            
            for i in range(max_len):
                if i < cp:
                    if i < len1:
                        child_mats.append(parent1[0][i])
                        child_ots.append(parent1[1][i])
                    else:
                        # Pad from parent2 if parent1 is shorter? Or just stop?
                        # Let's just take from parent2
                        if i < len2:
                            child_mats.append(parent2[0][i])
                            child_ots.append(parent2[1][i])
                else:
                    if i < len2:
                        child_mats.append(parent2[0][i])
                        child_ots.append(parent2[1][i])
                    else:
                        if i < len1:
                            child_mats.append(parent1[0][i])
                            child_ots.append(parent1[1][i])
                
            # Remove duplicates? No, layers are distinct.
            # But we might have created a layer with invalid thickness?
            # The OTs are from the discrete set, so they are valid.
            
            # Mutation
            # 1. Change material
            # 2. Change OT
            # 3. Add layer
            # 4. Remove layer
            
            # Material mutation
            for i in range(len(child_mats)):
                if np.random.random() < 0.1:
                    child_mats[i] = 'L' if child_mats[i] == 'H' else 'H'
            
            # OT mutation
            for i in range(len(child_ots)):
                if np.random.random() < 0.2:
                    child_ots[i] = np.random.randint(0, num_ot)
            
            # Add layer
            if np.random.random() < 0.1 and len(child_mats) < max_layers:
                child_mats.append(np.random.choice(['L', 'H']))
                child_ots.append(np.random.randint(0, num_ot))
            
            # Remove layer
            if np.random.random() < 0.1 and len(child_mats) > 1:
                idx = np.random.randint(0, len(child_mats))
                child_mats.pop(idx)
                child_ots.pop(idx)
            
            # Validate
            if not child_mats:
                continue
                
            total_ot = sum(discrete_ot[ot] for ot in child_ots)
            if total_ot / 1.38 > max_total:
                continue
                
            new_population.append((child_mats, child_ots))
            
        # Evaluate new population
        new_scores = []
        for ind in new_population:
            s = _evaluate_candidate(ind[0], ind[1], problem, wavelengths)
            new_scores.append(s)
            
        population = new_population
        scores = new_scores
        
        # Update best
        if min(scores) < best_score:
            best_score = min(scores)
            best_ind = population[np.argmin(scores)]
            
    # Convert best_ind to output format
    materials_labels = best_ind[0]
    thickness_indices = best_ind[1]
    
    thicknesses = []
    for mat_label, ot_idx in zip(materials_labels, thickness_indices):
        n_mat = mat_map[mat_label]
        ot = discrete_ot[ot_idx]
        d = ot / n_mat
        thicknesses.append(d)
        
    return materials_labels, thicknesses
# EVOLVE-BLOCK-END
