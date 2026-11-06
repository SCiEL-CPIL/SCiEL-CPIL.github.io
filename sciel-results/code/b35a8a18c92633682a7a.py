<<<<<<< SEARCH
def _crossover(ind1, ind2, materials, max_layers, max_total_nm, max_layer_nm, thickness_candidates):
    """
    Simple crossover: take a prefix from one parent and suffix from the other,
    then repair to satisfy constraints.
    """
    labels1, th1 = ind1
    labels2, th2 = ind2
    
    if len(labels1) == 0:
        return ind2
    if len(labels2) == 0:
        return ind1
    
    # Random crossover point
    point = random.randint(1, min(len(labels1), len(labels2)) - 1)
    
    new_labels = labels1[:point] + labels2[point:]
    new_th = th1[:point] + th2[:point]
    
    # Truncate to max_layers
    if len(new_labels) > max_layers:
        new_labels = new_labels[:max_layers]
        new_th = new_th[:max_layers]
    
    # Repair: ensure total thickness <= max_total_nm
    total = sum(new_th)
    if total > max_total_nm:
        scale = max_total_nm / total
        new_th = [t * scale for t in new_th]
        # Re-quantize to nearest candidate
        for i in range(len(new_th)):
            nearest = min(thickness_candidates, key=lambda c: abs(c - new_th[i]))
            new_th[i] = nearest
    
    # Ensure individual layer constraints
    for i in range(len(new_th)):
        if new_th[i] > max_layer_nm:
            new_th[i] = max_layer_nm
    
    # Final check: if still over, zero out some layers
    total = sum(new_th)
    while total > max_total_nm and len(new_th) > 1:
        # Remove the thickest layer
        max_idx = np.argmax(new_th)
        new_th[max_idx] = 0.0
        total -= new_th[max_idx]
    
    return new_labels, new_th

def _mutate(ind, materials, max_layers, max_total_nm, max_layer_nm, thickness_candidates, mutation_rate=0.3):
    """
    Mutate an individual with some probability.
    """
    labels, thicknesses = ind
    labels = list(labels)
    thicknesses = list(thicknesses)
    
    # Maybe change a layer's index
    if labels and random.random() < mutation_rate:
        idx = random.randint(0, len(labels) - 1)
        labels[idx] = random.choice(['L', 'H'])
    
    # Maybe change a thickness
    if thicknesses and random.random() < mutation_rate:
        idx = random.randint(0, len(thicknesses) - 1)
        thicknesses[idx] = random.choice(thickness_candidates)
    
    # Maybe add a layer
    if len(labels) < max_layers and random.random() < mutation_rate * 0.5:
        labels.append(random.choice(['L', 'H']))
        thicknesses.append(random.choice(thickness_candidates))
    
    # Maybe remove a layer
    if len(labels) > 1 and random.random() < mutation_rate * 0.5:
        idx = random.randint(0, len(labels) - 1)
        labels.pop(idx)
        thicknesses.pop(idx)
    
    # Repair constraints
    total = sum(thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]
        for i in range(len(thicknesses)):
            nearest = min(thickness_candidates, key=lambda c: abs(c - thicknesses[i]))
            thicknesses[i] = nearest
    
    for i in range(len(thicknesses)):
        if thicknesses[i] > max_layer_nm:
            thicknesses[i] = max_layer_nm
    
    return labels, thicknesses

def _genetic_search(problem, deadline):
    """
    Run a genetic algorithm to find a good AR coating design.
    """
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Generate candidate thicknesses
    thickness_candidates = _generate_quarter_wave_thickesses(materials, band_nm, max_layer_nm, max_total_nm)
    
    # Initialize population
    pop_size = 50
    population = []
    for _ in range(pop_size):
        ind = _random_individual(materials, max_layers, max_total_nm, max_layer_nm, thickness_candidates)
        population.append(ind)
    
    # Evaluate initial population
    scores = []
    for ind in population:
        labels, thicknesses = ind
        indices = [materials[label] for label in labels]
        score = _objective(labels, thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx,
                          max_layer_nm, max_total_nm)
        scores.append(score)
    
    # Find initial best
    best_idx = np.argmin(scores)
    best_ind = population[best_idx]
    best_score = scores[best_idx]
    
    # Genetic algorithm loop
    start_time = time.time()
    generation = 0
    no_improve_generations = 0
    max_no_improve = 15
    
    while time.time() < deadline - 10:
        generation += 1
        no_improve_generations += 1
        
        # Sort population by score
        sorted_indices = np.argsort(scores)
        population = [population[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # Elitism: keep top 10%
        elite_count = max(1, pop_size // 10)
        new_population = population[:elite_count]
        
        # Generate new individuals
        while len(new_population) < pop_size:
            if random.random() < 0.7:
                # Crossover
                p1 = random.choice(population[:pop_size // 2])
                p2 = random.choice(population[:pop_size // 2])
                child = _crossover(p1, p2, materials, max_layers, max_total_nm, max_layer_nm, thickness_candidates)
            else:
                # Mutation of a random individual
                parent = random.choice(population)
                child = _mutate(parent, materials, max_layers, max_total_nm, max_layer_nm, thickness_candidates)
            
            new_population.append(child)
        
        # Evaluate new population
        new_scores = []
        for ind in new_population:
            labels, thicknesses = ind
            indices = [materials[label] for label in labels]
            score = _objective(labels, thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx,
                              max_layer_nm, max_total_nm)
            new_scores.append(score)
        
        # Combine and select
        all_inds = new_population
        all_scores = new_scores
        
        # Update best
        best_idx_new = np.argmin(all_scores)
        if all_scores[best_idx_new] < best_score:
            best_ind = all_inds[best_idx_new]
            best_score = all_scores[best_idx_new]
            no_improve_generations = 0
        
        # Prepare next generation: keep best individuals
        sorted_new = np.argsort(all_scores)
        population = [all_inds[i] for i in sorted_new]
        scores = [all_scores[i] for i in sorted_new]
        
        # Early stop if no improvement for many generations
        if no_improve_generations >= max_no_improve:
            break
    
    return best_ind, best_score

def _local_refine(labels, thicknesses, problem, deadline):
    """
    Local refinement using coordinate descent on the best individual.
    """
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    indices = [materials[label] for label in labels]
    n_layers = len(labels)
    
    x = np.array(thicknesses, dtype=float)
    f_current = _objective(labels, x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                          max_layer_nm, max_total_nm)
    
    step_sizes = [20, 10, 5, 2, 1, 0.5, 0.2, 0.1]
    
    for step in step_sizes:
        if time.time() > deadline - 5:
            break
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(n_layers):
                # Try increasing
                x_trial = x.copy()
                x_trial[i] += step
                if x_trial[i] <= max_layer_nm and np.sum(x_trial) <= max_total_nm:
                    f_trial = _objective(labels, x_trial, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                        max_layer_nm, max_total_nm)
                    if f_trial < f_current:
                        x = x_trial
                        f_current = f_trial
                        improved = True
                
                # Try decreasing
                x_trial = x.copy()
                x_trial[i] -= step
                if x_trial[i] >= 0:
                    f_trial = _objective(labels, x_trial, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                        max_layer_nm, max_total_nm)
                    if f_trial < f_current:
                        x = x_trial
                        f_current = f_trial
                        improved = True
    
    return labels, x.tolist(), f_current
=======
def _differential_evolution_search(problem, deadline):
    """
    Run Differential Evolution on fixed material sequences.
    """
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Define a few good candidate sequences
    # Alternating L/H is a standard broadband AR structure
    seqs = [
        ['L', 'H'] * (max_layers // 2),
        ['H', 'L'] * (max_layers // 2),
        ['L'] * max_layers,
        ['H'] * max_layers,
    ]
    # Add some random sequences
    for _ in range(5):
        seqs.append([random.choice(['L', 'H']) for _ in range(max_layers)])
    
    best_labels = None
    best_thick = None
    best_score = float('inf')
    
    pop_size = 30
    max_generations = 50
    F = 0.8  # Differential weight
    CR = 0.9  # Crossover rate
    
    for seq in seqs:
        if time.time() > deadline - 10:
            break
            
        indices = [materials[label] for label in seq]
        
        # Initialize population of thicknesses
        # Start with random thicknesses
        population = []
        for _ in range(pop_size):
            # Random thicknesses, biased towards quarter wave
            th = np.random.uniform(0, max_layer_nm, max_layers)
            # Ensure total sum constraint
            if np.sum(th) > max_total_nm:
                th *= max_total_nm / np.sum(th)
            population.append(th)
        
        # Evaluate initial population
        scores = []
        for th in population:
            score = _objective(seq, th, indices, wavelengths_nm, ambient_idx, substrate_idx,
                              max_layer_nm, max_total_nm)
            scores.append(score)
        
        # DE Loop
        for gen in range(max_generations):
            if time.time() > deadline - 10:
                break
            
            new_population = []
            new_scores = []
            
            for i in range(pop_size):
                # Select three distinct random individuals
                r1, r2, r3 = random.sample(range(pop_size), 3)
                
                # Mutation
                mutant = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                trial = np.zeros_like(population[i])
                for j in range(max_layers):
                    if random.random() < CR or j == random.randint(0, max_layers - 1):
                        trial[j] = mutant[j]
                    else:
                        trial[j] = population[i][j]
                
                # Bounds
                trial = np.clip(trial, 0, max_layer_nm)
                
                # Constraint: total thickness
                total = np.sum(trial)
                if total > max_total_nm:
                    if total > 0:
                        trial *= max_total_nm / total
                    else:
                        trial = np.zeros(max_layers)
                
                # Evaluate
                score = _objective(seq, trial, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                  max_layer_nm, max_total_nm)
                
                # Selection
                if score < scores[i]:
                    new_population.append(trial)
                    new_scores.append(score)
                else:
                    new_population.append(population[i])
                    new_scores.append(scores[i])
            
            population = new_population
            scores = new_scores
            
            # Track best
            current_best_idx = np.argmin(scores)
            if scores[current_best_idx] < best_score:
                best_score = scores[current_best_idx]
                best_thick = population[current_best_idx]
                best_labels = list(seq)
    
    return best_labels, best_thick, best_score
>>>>>>> REPLACE

Now I need to update `design` to use the new search function and handle the return types correctly. The `_differential_evolution_search` returns `(labels, thicknesses, score)`. The `thicknesses` is a numpy array. I need to convert it to a list and ensure it's valid.

