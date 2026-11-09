import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)  # 560
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Enumerate all K(2,2,2) forbidden configurations
    # Each config is a set of 8 triples
    forbidden_configs = []
    
    for six_points in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six_points
        # 15 ways to partition 6 points into 3 pairs
        pairings = [
            [(p0,p1),(p2,p3),(p4,p5)],
            [(p0,p1),(p2,p4),(p3,p5)],
            [(p0,p1),(p2,p5),(p3,p4)],
            [(p0,p2),(p1,p3),(p4,p5)],
            [(p0,p2),(p1,p4),(p3,p5)],
            [(p0,p2),(p1,p5),(p3,p4)],
            [(p0,p3),(p1,p2),(p4,p5)],
            [(p0,p3),(p1,p4),(p2,p5)],
            [(p0,p3),(p1,p5),(p2,p4)],
            [(p0,p4),(p1,p2),(p3,p5)],
            [(p0,p4),(p1,p3),(p2,p5)],
            [(p0,p4),(p1,p5),(p2,p3)],
            [(p0,p5),(p1,p2),(p3,p4)],
            [(p0,p5),(p1,p3),(p2,p4)],
            [(p0,p5),(p1,p4),(p2,p3)],
        ]
        
        for pairs in pairings:
            (a1,b1), (a2,b2), (a3,b3) = pairs
            transversals = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        t = tuple(sorted([(a1,b1)[i], (a2,b2)[j], (a3,b3)[k]]))
                        if t in triple_index:
                            transversals.append(triple_index[t])
            
            if len(transversals) == 8:
                forbidden_configs.append(frozenset(transversals))
    
    forbidden_configs = list(set(forbidden_configs))
    num_configs = len(forbidden_configs)
    
    # Build incidence structure
    triple_to_configs = [[] for _ in range(num_triples)]
    config_to_triples = [list(c) for c in forbidden_configs]
    for ci, config in enumerate(forbidden_configs):
        for t in config:
            triple_to_configs[t].append(ci)
    
    # Greedy construction with random ordering
    def greedy_construct():
        present = [False] * num_triples
        config_count = [0] * num_configs
        
        # Random ordering of triples
        order = list(range(num_triples))
        random.shuffle(order)
        
        for t in order:
            if time.time() > deadline - 5:
                break
            # Check if adding t violates any constraint
            can_add = True
            for ci in triple_to_configs[t]:
                if config_count[ci] >= 7:
                    can_add = False
                    break
            if can_add:
                present[t] = True
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        
        return present, config_count
    
    # Local search with add, delete, and swap moves
    def local_search(present, config_count):
        improved = True
        stall_count = 0
        
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Try to add a triple
            added = False
            for _ in range(50):  # Try up to 50 random candidates
                t = random.randint(0, num_triples - 1)
                if present[t]:
                    continue
                # Check if we can add t
                can_add = True
                for ci in triple_to_configs[t]:
                    if config_count[ci] >= 7:
                        can_add = False
                        break
                if can_add:
                    present[t] = True
                    for ci in triple_to_configs[t]:
                        config_count[ci] += 1
                    improved = True
                    added = True
                    break
            
            if added:
                continue
            
            # Try a swap: remove one triple, add another
            swapped = False
            for _ in range(50):  # Try up to 50 random swap candidates
                t_remove = random.randint(0, num_triples - 1)
                if not present[t_remove]:
                    continue
                
                # Find which configs are satisfied by removing t_remove
                # After removing t_remove, some configs go from 7 to 6, becoming "slack"
                # We need to find a triple t_add that fits in all its configs after removal
                
                # First, check which configs become available
                slack_configs = set()
                for ci in triple_to_configs[t_remove]:
                    if config_count[ci] == 7:
                        slack_configs.add(ci)
                
                if not slack_configs:
                    continue
                
                t_add = random.randint(0, num_triples - 1)
                if present[t_add]:
                    continue
                
                # Check if t_add can be added after removing t_remove
                can_add = True
                for ci in triple_to_configs[t_add]:
                    if ci in slack_configs:
                        # After removing t_remove, this config has 6, so adding t_add makes 7 - OK
                        continue
                    if config_count[ci] >= 7:
                        can_add = False
                        break
                
                if can_add:
                    # Perform the swap
                    present[t_remove] = False
                    for ci in triple_to_configs[t_remove]:
                        config_count[ci] -= 1
                    present[t_add] = True
                    for ci in triple_to_configs[t_add]:
                        config_count[ci] += 1
                    improved = True
                    swapped = True
                    break
            
            if swapped:
                stall_count = 0
                continue
            
            # No improvement found
            stall_count += 1
            if stall_count > 20:
                break
        
        return present, config_count
    
    # Simulated annealing style search
    def annealing_search(present, config_count):
        temp = 1.0
        cooling = 0.995
        best_present = present[:]
        best_config_count = config_count[:]
        best_count = sum(present)
        
        while time.time() < deadline - 15 and temp > 0.1:
            # Random perturbation: remove a random present triple
            present_list = [i for i in range(num_triples) if present[i]]
            if not present_list:
                break
            
            t_remove = random.choice(present_list)
            present[t_remove] = False
            for ci in triple_to_configs[t_remove]:
                config_count[ci] -= 1
            
            # Try to add some triples
            for _ in range(30):
                t = random.randint(0, num_triples - 1)
                if present[t]:
                    continue
                can_add = True
                for ci in triple_to_configs[t]:
                    if config_count[ci] >= 7:
                        can_add = False
                        break
                if can_add:
                    present[t] = True
                    for ci in triple_to_configs[t]:
                        config_count[ci] += 1
            
            # Run local search to improve
            present, config_count = local_search(present, config_count)
            
            current_count = sum(present)
            if current_count > best_count:
                best_count = current_count
                best_present = present[:]
                best_config_count = config_count[:]
            
            temp *= cooling
        
        return best_present, best_config_count
    
    best_present = None
    best_count = 0
    best_config_count = None
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 25:
        present, config_count = greedy_construct()
        present, config_count = local_search(present, config_count)
        
        count = sum(present)
        if count > best_count:
            best_count = count
            best_present = present[:]
            best_config_count = config_count[:]
        
        num_restarts += 1
        
        # Occasionally try annealing from the best solution
        if num_restarts % 5 == 0 and best_present is not None and time.time() < deadline - 40:
            anneal_present, anneal_count = annealing_search(best_present[:], best_config_count[:])
            anneal_sum = sum(anneal_present)
            if anneal_sum > best_count:
                best_count = anneal_sum
                best_present = anneal_present[:]
                best_config_count = anneal_count[:]
        
        if num_restarts > 500:  # Safety cap
            break
    
    if best_present is None:
        return []

    return [list(all_triples[i]) for i in range(num_triples) if best_present[i]]
