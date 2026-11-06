<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin: finish before 90s limit
    
    # Step 1: Construct initial network
    initial = _construct_initial_network()
    current = initial[:]
    best_network = initial[:]
    best_len = len(current)
    
    # Step 2: SAT-based optimization (new structural approach)
    if _HAS_PYSAT and time.time() < deadline - 30.0:
        sat_result = _sat_based_optimization(current, deadline - 30.0)
        if sat_result is not None and len(sat_result) < best_len:
            best_network = sat_result[:]
            best_len = len(best_network)
            current = sat_result[:]
    
    # Step 3: Aggressive systematic deletion passes
    for pass_num in range(30):
        if time.time() > deadline - 25.0:
            break
        
        progress = False
        i = 0
        while i < len(current):
            if time.time() > deadline - 25.0:
                break
            
            success, new_current = _try_delete(current, i, deadline - 25.0)
            if success:
                current = new_current
                if len(current) < best_len:
                    best_network = current[:]
                    best_len = len(current)
                progress = True
                continue
            i += 1
        
        if not progress:
            break
    
    # Step 4: Genetic algorithm (structural change)
    rng = random.Random(12345)
    if time.time() < deadline - 15.0:
        ga_comps, ga_layers = _genetic_algorithm(current, deadline - 15.0, rng)
        if ga_comps is not None:
            if is_sorted_network_full(ga_comps):
                if len(ga_comps) < best_len:
                    best_network = ga_comps[:]
                    best_len = len(ga_comps)
                    current = ga_comps[:]
    
    # Step 5: More deletion after GA
    if time.time() < deadline - 10.0:
        for pass_num in range(20):
            if time.time() > deadline - 10.0:
                break
            progress = False
            i = 0
            while i < len(current):
                if time.time() > deadline - 10.0:
                    break
                success, new_current = _try_delete(current, i, deadline - 10.0)
                if success:
                    current = new_current
                    if len(current) < best_len:
                        best_network = current[:]
                        best_len = len(current)
                    progress = True
                    continue
                i += 1
            if not progress:
                break
    
    # Step 6: Local search
    if time.time() < deadline - 5.0:
        current = _local_search(current, deadline - 5.0, rng)
        if len(current) < best_len:
            best_network = current[:]
            best_len = len(current)
    
    # Step 7: Final deletion pass
    if time.time() < deadline - 2.0:
        i = 0
        while i < len(current) and time.time() < deadline - 2.0:
            success, new_current = _try_delete(current, i, deadline - 2.0)
            if success:
                current = new_current
                if len(current) < best_len:
                    best_network = current[:]
                    best_len = len(current)
                continue
            i += 1
    
    # Final validation
    if not is_sorted_network_full(best_network):
        if is_sorted_network_full(initial):
            return initial
        return _odd_even_transposition(N)
    
    return best_network
=======
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin: finish before 90s limit
    
    # Step 1: Construct initial network (prefer odd-even merge sort)
    initial = _construct_initial_network()
    current = initial[:]
    best_network = initial[:]
    best_len = len(current)
    
    rng = random.Random(12345)
    
    # Step 2: Aggressive systematic deletion passes
    for pass_num in range(50):
        if time.time() > deadline - 10.0:
            break
        
        progress = False
        i = 0
        while i < len(current):
            if time.time() > deadline - 10.0:
                break
            
            success, new_current = _try_delete_and_reorder(current, i, deadline - 10.0, rng)
            if success:
                current = new_current
                if len(current) < best_len:
                    best_network = current[:]
                    best_len = len(current)
                progress = True
                continue
            i += 1
        
        if not progress:
            break
    
    # Step 3: Iterative local search
    if time.time() < deadline - 5.0:
        current = _iterative_local_search(current, deadline - 5.0, rng)
        if is_sorted_network_full(current):
            if len(current) < best_len:
                best_network = current[:]
                best_len = len(current)
    
    # Step 4: Final deletion pass
    if time.time() < deadline - 2.0:
        i = 0
        while i < len(current) and time.time() < deadline - 2.0:
            success, new_current = _try_delete(current, i, deadline - 2.0)
            if success:
                current = new_current
                if len(current) < best_len:
                    best_network = current[:]
                    best_len = len(current)
                continue
            i += 1
    
    # Final validation
    if not is_sorted_network_full(best_network):
        if is_sorted_network_full(initial):
            return initial
        return _odd_even_transposition(N)
    
    return best_network
>>>>>>> REPLACE
