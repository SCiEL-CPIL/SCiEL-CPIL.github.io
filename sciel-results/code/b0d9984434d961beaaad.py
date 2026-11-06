"""Max K(2,2,2)-free triple system on 16 points via precomputed config counts + local search."""
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

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
    
    # Count total configs first
    num_configs = 0
    for vertices in itertools.combinations(range(N), 6):
        for p in pairings:
            num_configs += 1
            
    config_idx = 0
    
    # Populate triple_configs
    for vertices in itertools.combinations(range(N), 6):
        for p in pairings:
            a1, a2 = vertices[p[0][0]], vertices[p[0][1]]
            b1, b2 = vertices[p[1][0]], vertices[p[1][1]]
            c1, c2 = vertices[p[2][0]], vertices[p[2][1]]
            
            for a in (a1, a2):
                for b in (b1, b2):
                    for c in (c1, c2):
                        t = tuple(sorted((a, b, c)))
                        idx = triple_to_idx[t]
                        triple_configs[idx].append(config_idx)
            
            config_idx += 1
    
    # Helper functions
    def can_add(i, config_counts):
        for c in triple_configs[i]:
            if config_counts[c] == 7:
                return False
        return True

    def do_add(i, config_counts):
        for c in triple_configs[i]:
            config_counts[c] += 1

    def do_remove(i, config_counts):
        for c in triple_configs[i]:
            config_counts[c] -= 1

    def greedy_init(seed):
        random.seed(seed)
        in_set = [False] * num_triples
        config_counts = [0] * num_configs
        indices = list(range(num_triples))
        random.shuffle(indices)
        for i in indices:
            if can_add(i, config_counts):
                in_set[i] = True
                do_add(i, config_counts)
        return in_set, config_counts

    def local_search(in_set, config_counts, time_limit):
        current_size = sum(in_set)
        best_size = current_size
        best_in_set = in_set[:]
        best_config_counts = config_counts[:]
        
        end_time = time.time() + time_limit
        stuck_counter = 0
        max_stuck = 50
        
        while time.time() < end_time:
            # Try to add a triple
            improved = False
            addable = []
            for i in range(num_triples):
                if not in_set[i] and can_add(i, config_counts):
                    addable.append(i)
            
            if addable:
                i = random.choice(addable)
                in_set[i] = True
                do_add(i, config_counts)
                current_size += 1
                improved = True
                stuck_counter = 0
                
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set[:]
                    best_config_counts = config_counts[:]
            else:
                stuck_counter += 1
                if stuck_counter > max_stuck:
                    # Perturbation: Remove k random present triples
                    present = [i for i in range(num_triples) if in_set[i]]
                    if present:
                        k = random.randint(3, 8)
                        if len(present) >= k:
                            to_remove = random.sample(present, k)
                            old_in_set = in_set[:]
                            old_config_counts = config_counts[:]
                            old_size = current_size
                            
                            for i in to_remove:
                                in_set[i] = False
                                do_remove(i, config_counts)
                                current_size -= 1
                                
                            # Greedily add as many as possible
                            missing = [i for i in range(num_triples) if not in_set[i]]
                            random.shuffle(missing)
                            for i in missing:
                                if can_add(i, config_counts):
                                    in_set[i] = True
                                    do_add(i, config_counts)
                                    current_size += 1
                                    
                            if current_size > best_size:
                                best_size = current_size
                                best_in_set = in_set[:]
                                best_config_counts = config_counts[:]
                                stuck_counter = 0
                            else:
                                # Revert
                                in_set = old_in_set
                                config_counts = old_config_counts
                                current_size = old_size
                                stuck_counter = 0
                else:
                    # Try a 1-swap: remove one, try to add another
                    present = [i for i in range(num_triples) if in_set[i]]
                    if present:
                        i_out = random.choice(present)
                        old_in_set = in_set[:]
                        old_config_counts = config_counts[:]
                        old_size = current_size
                        
                        in_set[i_out] = False
                        do_remove(i_out, config_counts)
                        current_size -= 1
                        
                        added = False
                        for i in range(num_triples):
                            if not in_set[i] and can_add(i, config_counts):
                                in_set[i] = True
                                do_add(i, config_counts)
                                current_size += 1
                                added = True
                                break
                        
                        if added and current_size > best_size:
                            best_size = current_size
                            best_in_set = in_set[:]
                            best_config_counts = config_counts[:]
                            stuck_counter = 0
                        else:
                            in_set = old_in_set
                            config_counts = old_config_counts
                            current_size = old_size
                            stuck_counter += 1

        return best_in_set, best_config_counts, best_size

    # Main loop: Random Restarts
    global_best_size = 0
    global_best_state = None
    global_best_counts = None
    
    # Use more restarts with faster per-step logic
    num_restarts = 500
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        seed = 42 + r
        in_set, config_counts = greedy_init(seed)
        
        # Adjust time for local search to not exceed global deadline
        remaining = deadline - time.time()
        search_time = min(time_per_restart, remaining - 1)
        
        if search_time <= 0:
            break
            
        best_state, best_counts, best_size = local_search(in_set, config_counts, search_time)
        
        if best_size > global_best_size:
            global_best_size = best_size
            global_best_state = best_state
            global_best_counts = best_counts

    if global_best_state is None:
        in_set, config_counts = greedy_init(42)
        global_best_state = in_set
        global_best_counts = config_counts
        global_best_size = sum(in_set)

    return [list(all_triples[i]) for i in range(num_triples) if global_best_state[i]]
