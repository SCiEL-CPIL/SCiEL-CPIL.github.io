import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Enumerate all K(2,2,2) forbidden configurations
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
            if time.time() > deadline - 15:
                break
            
            # Find triple with max active config count
            max_count = -1
            best_t = -1
            for t in range(num_triples):
                if present[t]:
                    if triple_active_count[t] > max_count:
                        max_count = triple_active_count[t]
                        best_t = t
            
            if best_t == -1:
                break
                
            present[best_t] = False
            
            for ci in triple_to_configs[best_t]:
                if active[ci]:
                    active[ci] = False
                    num_active -= 1
                    for t_other in config_to_triples[ci]:
                        if t_other != best_t and present[t_other]:
                            triple_active_count[t_other] -= 1
            
        return present

    def local_search(present):
        # Reconstruct config counts
        config_count = [0] * num_configs
        for t in range(num_triples):
            if present[t]:
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # 1. Try to add safe triples
            # unsafe_count[t] = number of configs c containing t with config_count[c] >= 7
            unsafe_count = [0] * num_triples
            for t in range(num_triples):
                if not present[t]:
                    for ci in triple_to_configs[t]:
                        if config_count[ci] >= 7:
                            unsafe_count[t] += 1
            
            safe_triples = [t for t in range(num_triples) if not present[t] and unsafe_count[t] == 0]
            
            if safe_triples:
                t = random.choice(safe_triples)
                present[t] = True
                improved = True
                
                # Update unsafe_count
                for ci in triple_to_configs[t]:
                    old_count = config_count[ci]
                    config_count[ci] = old_count + 1
                    
                    if old_count == 6:
                        # Config becomes critical (7).
                        # Other missing triples in this config become unsafe.
                        for t_other in config_to_triples[ci]:
                            if t_other != t and not present[t_other]:
                                unsafe_count[t_other] += 1
                continue
            
            # 2. Try 1-remove, 2-add swap
            # Identify critical configs (count == 7)
            critical_configs = [ci for ci in range(num_configs) if config_count[ci] == 7]
            
            if not critical_configs:
                break
            
            # For each present triple t, see if removing it unblocks >= 2 missing triples
            # unblocked[t] = list of missing triples m such that:
            #   m is not present
            #   ALL critical configs containing m also contain t
            #   (And m is not blocked by any non-critical config? No, if count < 7, adding m makes it <= 7, which is safe. 
            #    Wait, if count is 6, adding m makes it 7. Safe.
            #    If count is 7, adding m makes it 8. Unsafe.
            #    So m is blocked IF AND ONLY IF there exists a critical config C containing m.
            #    So the condition "ALL critical configs containing m also contain t" is correct.
            #    If we remove t, those critical configs become count 6.
            #    Then m can be added, making them 7. Safe.
            #    We need to add TWO such m's.
            #    We must ensure that adding m1 and m2 doesn't create a NEW critical config issue?
            #    No, we are adding to configs that are now 6. They become 7.
            #    Are there other configs?
            #    If m1 is in a non-critical config C2 (count <= 6), adding m1 makes it count+1 <= 7. Safe.
            #    So the only blockers are the critical configs.
            #    
            #    However, we must ensure that m1 and m2 are distinct and not present.
            #    Also, we must ensure that m1 and m2 don't share a critical config that DOES NOT contain t?
            #    No, the condition "ALL critical configs containing m1 contain t" ensures that.
            
            # Optimization: Instead of checking all t, we can iterate over critical configs.
            # But let's stick to checking t if the list of present triples is small.
            # num_triples is 560. Present is ~260.
            
            swap_found = False
            for t in range(num_triples):
                if not present[t]:
                    continue
                
                unblocked = []
                for m in range(num_triples):
                    if present[m]:
                        continue
                    # Check if m is blocked ONLY by critical configs that contain t
                    is_blocked_by_other = False
                    for ci in triple_to_configs[m]:
                        if config_count[ci] == 7:
                            # This is a critical config blocking m
                            if t not in config_to_triples[ci]:
                                is_blocked_by_other = True
                                break
                    if not is_blocked_by_other:
                        unblocked.append(m)
                
                if len(unblocked) >= 2:
                    # Perform swap
                    m1, m2 = random.sample(unblocked, 2)
                    
                    # Remove t
                    present[t] = False
                    for ci in triple_to_configs[t]:
                        config_count[ci] -= 1
                    
                    # Add m1
                    present[m1] = True
                    for ci in triple_to_configs[m1]:
                        config_count[ci] += 1
                        
                    # Add m2
                    present[m2] = True
                    for ci in triple_to_configs[m2]:
                        config_count[ci] += 1
                        
                    improved = True
                    swap_found = True
                    break
            
            if swap_found:
                continue
                
            # If no simple add or swap, break and let outer loop perturb
            break
            
        return present

    best_present = None
    best_count = 0
    
    # Run multiple instances
    num_restarts = 0
    while time.time() < deadline - 10:
        present = greedy_delete()
        present = local_search(present)
        
        count = sum(present)
        if count > best_count:
            best_count = count
            best_present = present
        
        num_restarts += 1
        
        # If we are stuck, we could perturb, but greedy_delete is randomized enough
        # with random.choice in the max_count tie-breaking? 
        # Actually, greedy_delete picks the FIRST max. We should randomize it.
        
        if num_restarts > 200:
            break

    if best_present is None:
        return []

    return [list(all_triples[i]) for i in range(num_triples) if best_present[i]]
