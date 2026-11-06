import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all configurations (sets of 6 points partitioned into 3 pairs)
    # A configuration is "satisfied" (violating) if all 8 transversal triples are present.
    # We maintain config_load[c] = number of transversals present for config c.
    # A triple can be added only if for all configs containing it, config_load[c] < 7.
    
    # Hardcoded 15 pairings of 6 indices into 3 pairs
    pairings = [
        [(0,1),(2,3),(4,5)],
        [(0,1),(2,4),(3,5)],
        [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)],
        [(0,2),(1,4),(3,5)],
        [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)],
        [(0,3),(1,4),(2,5)],
        [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)],
        [(0,4),(1,3),(2,5)],
        [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)],
        [(0,5),(1,3),(2,4)],
        [(0,5),(1,4),(2,3)]
    ]
    
    # triple_configs[i] = list of config indices that contain triple i
    triple_configs = [[] for _ in range(num_triples)]
    
    # Count total configs
    num_configs = 0
    for vertices in itertools.combinations(range(N), 6):
        for p in pairings:
            num_configs += 1
            
    config_loads = [0] * num_configs
    config_idx = 0
    
    # Populate triple_configs
    for vertices in itertools.combinations(range(N), 6):
        for p in pairings:
            a1, a2 = vertices[p[0][0]], vertices[p[0][1]]
            b1, b2 = vertices[p[1][0]], vertices[p[1][1]]
            c1, c2 = vertices[p[2][0]], vertices[p[2][1]]
            
            # The 8 transversal triples
            t_indices = []
            for a in (a1, a2):
                for b in (b1, b2):
                    for c in (c1, c2):
                        t = tuple(sorted((a, b, c)))
                        idx = triple_to_idx[t]
                        t_indices.append(idx)
                        
            for idx in t_indices:
                triple_configs[idx].append(config_idx)
            
            config_idx += 1

    # Helper functions
    def can_add(i, loads):
        # Check if adding triple i would cause any config to reach 8
        for c in triple_configs[i]:
            if loads[c] == 7:
                return False
        return True

    def do_add(i, loads):
        for c in triple_configs[i]:
            loads[c] += 1

    def do_remove(i, loads):
        for c in triple_configs[i]:
            loads[c] -= 1

    def greedy_init(seed):
        random.seed(seed)
        in_set = [False] * num_triples
        loads = [0] * num_configs
        # Weighted greedy: prefer triples with fewer configs (less constraining)
        # Pre-compute degrees (len of triple_configs)
        degrees = [len(cfgs) for cfgs in triple_configs]
        indices = list(range(num_triples))
        # Sort by degree ascending to pick "safe" ones first
        # But randomize slightly to avoid deterministic bias? 
        # Let's just shuffle and iterate, but maybe pick low degree first?
        # Actually, simple random shuffle is often better for diversity in restarts.
        random.shuffle(indices)
        for i in indices:
            if can_add(i, loads):
                in_set[i] = True
                do_add(i, loads)
        return in_set, loads

    def simulated_annealing(in_set, loads, time_limit):
        current_size = sum(in_set)
        best_size = current_size
        best_in_set = in_set[:]
        best_loads = loads[:]
        
        # Temperature schedule
        # Start high enough to accept bad moves, end low
        T = 2.0 
        T_min = 0.1
        cooling_rate = 0.995 # Per step? No, per time unit or step count.
        
        # We don't know step count in advance, so use time-based decay or step-based.
        # Let's use step-based for consistency in logic, but check time.
        # Approx steps per second is ~1000-5000. 
        # Let's just decay T every 1000 steps.
        
        end_time = time.time() + time_limit
        steps = 0
        
        while time.time() < end_time:
            steps += 1
            if steps % 1000 == 0:
                T *= cooling_rate
                if T < T_min:
                    T = T_min
            
            # Choose move type
            # Probabilities: 
            # If we are small, try to add.
            # If we are stuck (no adds possible), try to remove/swap.
            
            # Simple heuristic: 
            # 1. Try to add a random inactive triple
            # 2. If fail, try to swap (remove active, add inactive)
            # 3. If fail, try to remove (to escape local max)
            
            action = random.random()
            
            # Find a valid add
            # Optimization: Instead of scanning all 560, keep a list of "potentially addable"?
            # But validity changes with loads. Scanning 560 is cheap (~50us).
            
            # 1. Try ADD
            if action < 0.5:
                # Pick random inactive
                inactive_idx = [i for i in range(num_triples) if not in_set[i]]
                if inactive_idx:
                    i = random.choice(inactive_idx)
                    if can_add(i, loads):
                        in_set[i] = True
                        do_add(i, loads)
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_in_set = in_set[:]
                            best_loads = loads[:]
                        # Move accepted (improvement)
                        continue
            
            # 2. Try SWAP (Remove 1, Add 1)
            elif action < 0.8:
                active_idx = [i for i in range(num_triples) if in_set[i]]
                inactive_idx = [i for i in range(num_triples) if not in_set[i]]
                if active_idx and inactive_idx:
                    i_out = random.choice(active_idx)
                    i_in = random.choice(inactive_idx)
                    
                    # Tentatively remove i_out
                    old_loads = loads[:] # Expensive copy! 
                    # Better: modify, check, revert if needed.
                    
                    do_remove(i_out, loads)
                    in_set[i_out] = False
                    temp_size = current_size - 1
                    
                    if can_add(i_in, loads):
                        do_add(i_in, loads)
                        in_set[i_in] = True
                        # Size remains same (removed 1, added 1)
                        # This is a "neutral" move. Accept to explore.
                        # But we should only keep it if it leads to improvement later?
                        # For now, accept neutral moves with probability 0.5 to shuffle structure.
                        if random.random() < 0.5:
                            current_size = temp_size + 1 # Still same size
                            # Note: best_size not updated as size didn't change
                        else:
                            # Revert
                            do_remove(i_in, loads)
                            in_set[i_in] = False
                            do_add(i_out, loads)
                            in_set[i_out] = True
                            current_size = temp_size + 1
                    else:
                        # Revert
                        do_add(i_out, loads)
                        in_set[i_out] = True
                        current_size = temp_size + 1
            else:
                # 3. Try REMOVE (Worsening move, accepted via SA)
                active_idx = [i for i in range(num_triples) if in_set[i]]
                if active_idx:
                    i = random.choice(active_idx)
                    # Energy change: Size decreases by 1. Delta E = +1 (in minimization of -Size)
                    # Prob = exp(-1 / T)
                    if random.random() < math.exp(-1.0 / T):
                        in_set[i] = False
                        do_remove(i, loads)
                        current_size -= 1
                        # best_size not updated
                    # Else reject removal, state unchanged

        return best_in_set, best_loads, best_size

    # Main loop: Random Restarts
    global_best_size = 0
    global_best_state = None
    global_best_counts = None
    
    # Number of restarts depends on time. 
    # Each restart takes ~10-20s? 
    # Let's try 5 restarts with ~25s each.
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        seed = 1000 + r
        in_set, loads = greedy_init(seed)
        
        remaining = deadline - time.time()
        # Leave 1s for final conversion
        search_time = min(time_per_restart, remaining - 1.0)
        
        if search_time <= 0:
            break
            
        best_state, best_counts, best_size = simulated_annealing(in_set, loads, search_time)
        
        if best_size > global_best_size:
            global_best_size = best_size
            global_best_state = best_state
            global_best_counts = best_counts

    if global_best_state is None:
        in_set, loads = greedy_init(42)
        global_best_state = in_set
        global_best_counts = loads
        global_best_size = sum(in_set)

    return [list(all_triples[i]) for i in range(num_triples) if global_best_state[i]]
