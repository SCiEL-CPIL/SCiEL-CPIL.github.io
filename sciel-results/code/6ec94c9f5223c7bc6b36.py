<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    seeds = [
        _initial_fcc_like(),
        _initial_bcc(),
        _initial_grid(),
        _initial_random()
    ]
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time to each seed
    total_time = deadline - start_time
    time_per_seed = total_time / len(seeds)
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = seed_start + time_per_seed * (1.0 + 0.1 * i)
        
        # Stage 1: Repulsion annealing
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, r_target=best_r + 0.005, max_iters=3000)
        
        # Stage 2: Lloyd's algorithm
        c_lloyd, r_lloyd = _lloyds_algorithm(c_anneal, seed_deadline, max_iters=100)
        
        # Stage 3: Softmin polish
        c_polish, r_polish = _polish(c_lloyd, seed_deadline, t_start=0.05, t_end=0.001)
        
        # Stage 4: Epigraph polish
        c_epi, r_epi = _epigraph_polish(c_polish, seed_deadline)
        
        # Stage 5: Local Search
        c_ls, r_ls = _local_search(c_epi, seed_deadline, max_iters=200)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
        if time.time() > deadline - 15.0:
            break
            
    # Final high-precision polish on the global best if time remains
    if time.time() < deadline - 15.0:
        final_deadline = deadline - 3.0
        c_final, r_final = _polish(best_c, final_deadline, t_start=0.01, t_end=0.00005)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        if time.time() < final_deadline - 3.0:
            c_epi, r_epi = _epigraph_polish(best_c, final_deadline)
            if r_epi > best_r:
                best_r = r_epi
                best_c = c_epi.copy()
            
        if time.time() < final_deadline - 3.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=500)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
        if time.time() < final_deadline - 2.0:
            c_final2, r_final2 = _polish(best_c, final_deadline - 1.0, t_start=0.001, t_end=0.00001)
            if r_final2 > best_r:
                best_r = r_final2
                best_c = c_final2.copy()
                
    # Ensure all points are strictly inside the cube
    best_c = np.clip(best_c, -0.4999999, 0.4999999)
    
    return best_c
=======
def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin
    
    # Initial configurations
    seeds = [
        _initial_fcc_like(),
        _initial_bcc(),
        _initial_grid(),
        _initial_random()
    ]
    
    best_c = seeds[0].copy()
    best_r = _compute_radius(best_c)
    
    # Allocate time: 3 seeds get main budget, 4th gets remainder
    # First 3 seeds: ~35s each, 4th seed: remaining
    seed_time = 35.0
    remaining_time = deadline - start_time
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        if i < 3:
            seed_deadline = seed_start + seed_time
        else:
            # Use remaining time minus final polish budget
            seed_deadline = deadline - 20.0
        
        if seed_deadline <= seed_start:
            break
        
        # Stage 1: Repulsion annealing
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, r_target=best_r + 0.005, max_iters=4000)
        
        # Stage 2: Metropolis MC to escape local optima
        c_mc, r_mc = _metropolis_mc(c_anneal, seed_deadline, max_iters=3000, r_target=best_r + 0.003)
        
        # Stage 3: Lloyd's algorithm
        c_lloyd, r_lloyd = _lloyds_algorithm(c_mc, seed_deadline, max_iters=80)
        
        # Stage 4: Softmin polish
        c_polish, r_polish = _polish(c_lloyd, seed_deadline, t_start=0.05, t_end=0.0005)
        
        # Stage 5: Epigraph polish
        c_epi, r_epi = _epigraph_polish(c_polish, seed_deadline)
        
        # Stage 6: Local Search
        c_ls, r_ls = _local_search(c_epi, seed_deadline, max_iters=300)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
        
        # Check if we should stop early
        if time.time() > deadline - 25.0:
            break
        
    # Final high-precision polish on the global best if time remains
    final_start = time.time()
    if final_start < deadline - 20.0:
        final_deadline = deadline - 3.0
        
        # Multi-round final polish
        for round_i in range(3):
            if time.time() > final_deadline - 5.0:
                break
            
            # Softmin polish with decreasing temperature
            t_start = 0.01 / (round_i + 1)
            t_end = 0.00001 / (round_i + 1)
            c_final, r_final = _polish(best_c, final_deadline, t_start=t_start, t_end=t_end)
            if r_final > best_r:
                best_r = r_final
                best_c = c_final.copy()
            
            # Epigraph polish
            if time.time() < final_deadline - 5.0:
                c_epi, r_epi = _epigraph_polish(best_c, final_deadline)
                if r_epi > best_r:
                    best_r = r_epi
                    best_c = c_epi.copy()
            
            # Local search
            if time.time() < final_deadline - 5.0:
                c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=400)
                if r_ls > best_r:
                    best_r = r_ls
                    best_c = c_ls.copy()
                
            # Metropolis with very small steps for final refinement
            if time.time() < final_deadline - 5.0:
                c_mc, r_mc = _metropolis_mc(best_c, final_deadline, max_iters=2000, r_target=best_r + 0.001)
                if r_mc > best_r:
                    best_r = r_mc
                    best_c = c_mc.copy()
            
            # Final softmin with very small t
            if time.time() < final_deadline - 3.0:
                c_fin, r_fin = _polish(best_c, final_deadline - 1.0, t_start=0.001, t_end=0.000001)
                if r_fin > best_r:
                    best_r = r_fin
                    best_c = c_fin.copy()
    
    # Ensure all points are strictly inside the cube
    best_c = np.clip(best_c, -0.4999999, 0.4999999)
    
    return best_c
>>>>>>> REPLACE

