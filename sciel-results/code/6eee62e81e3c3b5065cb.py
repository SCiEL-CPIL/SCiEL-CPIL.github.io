import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Enumerate all K(2,2,2) forbidden configurations
    forbidden_configs = []
    
    for six_points in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six_points
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
    
    triple_to_configs = [[] for _ in range(num_triples)]
    config_to_triples = [list(c) for c in forbidden_configs]
    for ci, config in enumerate(forbidden_configs):
        for t in config:
            triple_to_configs[t].append(ci)
    
    def greedy_delete():
        present = [True] * num_triples
        config_count = [8] * num_configs
        active = [True] * num_configs
        
        triple_active_count = [0] * num_triples
        for t in range(num_triples):
            for ci in triple_to_configs[t]:
                if active[ci]:
                    triple_active_count[t] += 1
        
        num_active = num_configs
        
        while num_active > 0:
            if time.time() > deadline - 10:
                break
            
            max_count = -1
            candidates = []
            for t in range(num_triples):
                if present[t]:
                    if triple_active_count[t] > max_count:
                        max_count = triple_active_count[t]
                        candidates = [t]
                    elif triple_active_count[t] == max_count:
                        candidates.append(t)
            
            if not candidates:
                break
                
            best_t = random.choice(candidates)
            
            # Remove best_t
            present[best_t] = False
            
            # Update counts
            for ci in triple_to_configs[best_t]:
                if active[ci]:
                    active[ci] = False
                    num_active -= 1
                    
                    for t_other in config_to_triples[ci]:
                        if t_other != best_t and present[t_other]:
                            triple_active_count[t_other] -= 1
            
        return present

    def local_search_add(present):
        # Reconstruct config counts
        config_count = [0] * num_configs
        for t in range(num_triples):
            if present[t]:
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        
        # Initialize unsafe_count
        # unsafe_count[t] is number of configs c containing t such that config_count[c] >= 7
        unsafe_count = [0] * num_triples
        for t in range(num_triples):
            if not present[t]:
                for ci in triple_to_configs[t]:
                    if config_count[ci] >= 7:
                        unsafe_count[t] += 1
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Find a safe triple
            candidates = []
            for t in range(num_triples):
                if not present[t] and unsafe_count[t] == 0:
                    candidates.append(t)
            
            if candidates:
                t = random.choice(candidates)
                present[t] = True
                improved = True
                
                # Update unsafe_count for neighbors
                for ci in triple_to_configs[t]:
                    old_count = config_count[ci]
                    config_count[ci] = old_count + 1
                    
                    if old_count == 6:
                        # Config is now critical (7 present). 
                        # Any other triple in this config becomes unsafe.
                        for t_other in config_to_triples[ci]:
                            if t_other != t and not present[t_other]:
                                unsafe_count[t_other] += 1
        
        return present

    best_present = None
    best_count = 0
    
    # Run multiple instances of Greedy Delete + Local Search
    num_restarts = 0
    while time.time() < deadline - 15:
        present = greedy_delete()
        present = local_search_add(present)
        
        count = sum(present)
        if count > best_count:
            best_count = count
            best_present = present
        
        num_restarts += 1
        if num_restarts > 100: # Safety cap
            break
    
    if best_present is None:
        # Fallback
        return []

    return [list(all_triples[i]) for i in range(num_triples) if best_present[i]]
