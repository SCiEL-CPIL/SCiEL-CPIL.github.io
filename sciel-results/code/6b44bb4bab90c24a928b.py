<<<<<<< SEARCH
    # Define bounds for optimization
    # The tubes can be shifted. A range of [-1, 1] is reasonable to keep the box small.
    bounds = [(-1.0, 1.0)] * dim
    
    # Initial population guess: use the baseline Keich construction as a seed for one individual?
    # DE doesn't take a single seed, but we can initialize the population around the baseline.
    # However, standard DE with random init is robust. 
    # Let's try standard DE first.
    
    # To speed up convergence, we can provide a custom initial population.
    # Generate baseline positions
    pos_int = int(np.log2(N))
    j_terms = np.arange(1, pos_int + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_int - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_int)
        
    base_x = keich[i_flat]
    base_y = keich[j_flat]
    baseline_pos = np.stack([base_x, base_y], axis=1).flatten()
    
    # Create initial population
    # 1 individual: baseline
    # Rest: random
    popsize = 30
    init_pop = np.random.uniform(-1.0, 1.0, size=(popsize, dim))
    init_pop[0] = baseline_pos
    
    # Run Differential Evolution
    # maxiter and popsize are tuned for speed (< 600s)
    # 500k points * 64 tubes check is fast in numpy (~10-50ms per eval).
    # 30 pop * 50 iter = 1500 evals. 1500 * 0.05s = 75s. Safe.
    
    result = differential_evolution(
        calculate_volume, 
        bounds, 
        seed=42,
        maxiter=50,
        popsize=popsize,
        tol=0.01,
        mutation=(0.5, 1.0),
        recombination=0.7,
        init=init_pop,
        polish=False  # Polish can be slow and DE is usually good enough
    )
=======
    # Define bounds for optimization
    # The optimal positions are likely near 0 for all tubes (to minimize bounding box).
    # Allow some range for exploration.
    bounds = [(-0.5, 0.5)] * dim
    
    # Create initial population with diverse starting points
    # Include: all-zeros, small random perturbations, and a structured seed
    popsize = 40
    init_pop = np.zeros((popsize, dim))
    
    # Individual 0: all zeros (optimal baseline guess)
    init_pop[0] = 0.0
    
    # Individual 1: small random perturbations
    init_pop[1] = np.random.uniform(-0.05, 0.05, size=dim)
    
    # Individual 2: medium random
    init_pop[2] = np.random.uniform(-0.1, 0.1, size=dim)
    
    # Individual 3: larger random
    init_pop[3] = np.random.uniform(-0.2, 0.2, size=dim)
    
    # Rest: random within bounds
    init_pop[4:] = np.random.uniform(-0.5, 0.5, size=(popsize - 4, dim))
    
    # Run Differential Evolution with more iterations for better convergence
    # 500k points * 64 tubes check is fast in numpy (~30-50ms per eval).
    # 40 pop * 150 iter = 6000 evals. 6000 * 0.04s = 240s. Safe within 600s.
    
    result = differential_evolution(
        calculate_volume, 
        bounds, 
        seed=42,
        maxiter=150,
        popsize=popsize,
        tol=0.001,
        mutation=(0.3, 0.8),
        recombination=0.8,
        init=init_pop,
        polish=False,
        updating='deferred'
    )
>>>>>>> REPLACE

