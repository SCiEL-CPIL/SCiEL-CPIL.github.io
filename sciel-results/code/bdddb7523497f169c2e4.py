import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)  # 560
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute configs
    # A config is a 6-set and a pairing.
    # We identify a config by (six_set, partner_idx, remaining_pairing_idx)
    # But simpler: just generate all 120,120 and map them.
    
    triple_to_configs = defaultdict(list)
    config_to_triples = []
    config_id = 0
    
    # To speed up generation, we can iterate 6-sets and pairings
    for six_set in itertools.combinations(range(n), 6):
        a, b, c, d, e, f = six_set
        # 15 pairings
        # 1. (a,b)(c,d)(e,f)
        # 2. (a,b)(c,e)(d,f)
        # ...
        # Let's use a generic generator for pairings of 6 elements
        # Fixed point a. Partner can be b,c,d,e,f.
        for partner_idx in range(1, 6):
            partner = six_set[partner_idx]
            remaining = [six_set[i] for i in range(6) if i != 0 and i != partner_idx]
            r0, r1, r2, r3 = remaining
            pairs_list = [
                [(a, partner), (r0, r1), (r2, r3)],
                [(a, partner), (r0, r2), (r1, r3)],
                [(a, partner), (r0, r3), (r1, r2)],
            ]
            for pairs in pairs_list:
                p1, p2, p3 = pairs
                transversals = []
                for x in p1:
                    for y in p2:
                        for z in p3:
                            t = tuple(sorted((x, y, z)))
                            transversals.append(t)
                
                # Store mapping
                config_to_triples.append([triple_index[t] for t in transversals])
                for t_idx in config_to_triples[-1]:
                    triple_to_configs[t_idx].append(config_id)
                config_id += 1
                
    num_configs = config_id
    
    def solve_one(seed):
        random.seed(seed)
        selected = [False] * num_triples
        config_count = [0] * num_configs
        available = set(range(num_triples))
        
        # We will process triples one by one.
        # To improve packing, we can prioritize triples that are "popular" or just random.
        # Random with pruning is a good baseline.
        
        while available:
            if time.time() > deadline - 5:
                break
            
            # Pick a random triple from available
            t = random.choice(list(available))
            available.remove(t)
            
            # Check if t is still valid
            # t is valid if for all cid in triple_to_configs[t], config_count[cid] < 7
            valid = True
            for cid in triple_to_configs[t]:
                if config_count[cid] >= 7:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Add t
            selected[t] = True
            
            # Update counts and prune
            for cid in triple_to_configs[t]:
                config_count[cid] += 1
                if config_count[cid] == 7:
                    # This config is now saturated to 7.
                    # Any other triple in this config that is still available must be removed.
                    for t_other in config_to_triples[cid]:
                        if t_other != t and t_other in available:
                            available.remove(t_other)
                            
        return sum(1 for s in selected if s)

    best_count = 0
    seed = 0
    # Run multiple restarts
    # 560 triples, pruning is fast.
    # Each run is O(N * (degree)) roughly.
    # Let's run as many as we can.
    
    while time.time() < deadline - 10:
        count = solve_one(seed)
        if count > best_count:
            best_count = count
            best_selected = [False] * num_triples
            # Re-run to get the actual set? Or store it?
            # Storing 560 bools is cheap.
            # But we need to reconstruct the list of triples.
            # It's better to just return the count if we only care about score,
            # but the task requires returning the list of triples.
            # So we need to keep the best `selected` array.
            
            # Optimization: We can just re-run the best seed at the end?
            # Or store the selected array.
            # Let's store the selected array for the best run.
            
        seed += 1

    # We need to return the list of triples for the best run.
    # The loop above didn't store the `selected` array for the best run.
    # Let's refactor to store it.
    
    # Refactored loop
    best_count = 0
    best_selected = None
    seed = 0
    
    while time.time() < deadline - 10:
        random.seed(seed)
        selected = [False] * num_triples
        config_count = [0] * num_configs
        available = set(range(num_triples))
        
        while available:
            if time.time() > deadline - 5:
                break
            t = random.choice(list(available))
            available.remove(t)
            
            valid = True
            for cid in triple_to_configs[t]:
                if config_count[cid] >= 7:
                    valid = False
                    break
            
            if not valid:
                continue
            
            selected[t] = True
            
            for cid in triple_to_configs[t]:
                config_count[cid] += 1
                if config_count[cid] == 7:
                    for t_other in config_to_triples[cid]:
                        if t_other != t and t_other in available:
                            available.remove(t_other)
                            
        count = sum(1 for s in selected if s)
        if count > best_count:
            best_count = count
            best_selected = selected[:]
        seed += 1

    if best_selected is None:
        # Fallback: return empty or a small valid set
        # An empty set is valid.
        return []
        
    result = [list(all_triples[i]) for i in range(num_triples) if best_selected[i]]
    return result

