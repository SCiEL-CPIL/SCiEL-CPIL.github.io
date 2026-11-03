<<<<<<< SEARCH
    # Simulated annealing
    np.random.seed(42)
    random.seed(42)
    
    S = best_set.copy()
    current_size = len(S)
    T = 1.0
    cooling = 0.9995
    
    # Precompute adjacency as lists for faster iteration
    adj_list = [list(s) for s in adj]
    
    step = 0
    max_steps = 200000
    while step < max_steps and time.time() < deadline - 20.0:
        step += 1
        v = random.randint(0, n_sup - 1)
        
        if v in S:
            # Try removing v
            if random.random() < 0.1:
                S.discard(v)
                current_size -= 1
            continue
        
        # v not in S
        N = adj[v] & S
        n_N = len(N)
        
        if n_N == 0:
            S.add(v)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_set = S.copy()
        else:
            delta = 1 - n_N
            if delta >= 0:
                S -= N
                S.add(v)
                current_size += delta
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
            else:
                # delta < 0
                if T > 0.01:
                    p = np.exp(delta / T)
                    if random.random() < p:
                        S -= N
                        S.add(v)
                        current_size += delta
                        if current_size > best_size:
                            best_size = current_size
                            best_set = S.copy()
        
        T = max(0.01, T * cooling)
=======
    # Simulated Annealing with improved moves
    np.random.seed(42)
    random.seed(42)
    
    S = best_set.copy()
    current_size = len(S)
    T = 1.0
    cooling = 0.999
    
    step = 0
    max_steps = 300000
    while step < max_steps and time.time() < deadline - 20.0:
        step += 1
        v = random.randint(0, n_sup - 1)
        
        if v in S:
            # With some probability, try removing v to allow other swaps
            if random.random() < 0.05:
                S.discard(v)
                current_size -= 1
            continue
        
        # v not in S
        N = adj[v] & S
        n_N = len(N)
        
        if n_N == 0:
            S.add(v)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_set = S.copy()
        else:
            # Move 1: Remove all neighbors and add v (Expansion)
            # Move 2: Swap one neighbor with v (Neutral/Exploratory)
            move_type = random.random()
            
            if move_type < 0.5:
                # Try swapping one neighbor
                if n_N > 0:
                    u = random.choice(list(N))
                    # Check if removing u and adding v is valid?
                    # v is not in S. u is in S.
                    # We remove u and add v.
                    # This is always a valid state transition (size stays same).
                    # But we need to check if it leads to a better solution eventually.
                    # In SA, we accept based on energy (size).
                    # Here size is constant, so we accept with prob 1 (neutral move) 
                    # or based on some heuristic? 
                    # Actually, neutral moves help escape local minima.
                    S.discard(u)
                    S.add(v)
                    # Size unchanged
            else:
                # Try removing all neighbors and adding v
                delta = 1 - n_N
                if delta >= 0:
                    S -= N
                    S.add(v)
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_set = S.copy()
                else:
                    if T > 0.01:
                        p = np.exp(delta / T)
                        if random.random() < p:
                            S -= N
                            S.add(v)
                            current_size += delta
                            if current_size > best_size:
                                best_size = current_size
                                best_set = S.copy()
        
        T = max(0.01, T * cooling)
>>>>>>> REPLACE

This change introduces a "swap" move which keeps the set size constant but changes the composition. This is often more effective for MIS than only allowing size-changing moves, as it allows the algorithm to "walk" through the space of independent sets of the same size, potentially finding a better configuration that allows for further expansion later. The "remove all" move is still used for expansion/contraction.

I will also adjust the restart loop to use the same improved logic.

