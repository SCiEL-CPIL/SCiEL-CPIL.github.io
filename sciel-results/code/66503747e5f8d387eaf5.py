import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)

    # Precompute all K(2,2,2) configurations (sets of 8 transversal triples)
    # We store them as lists of triple indices
    k222_configs = []
    seen_configs = set()
    
    # Optimization: Iterate over 6 points
    for six in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six
        # There are 15 ways to partition 6 items into 3 pairs
        # We generate the 8 transversal triples for each partition
        # To avoid duplicates in the global list, we can just add them.
        # Duplicates in k222_configs don't hurt correctness, just slightly slow down checks.
        # But we can use a set of frozensets to be safe.
        
        pairs_list = [
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
        
        for pairs in pairs_list:
            (a,b),(c,d),(e,f) = pairs
            transversal = [
                tuple(sorted((a,c,e))), tuple(sorted((a,c,f))),
                tuple(sorted((a,d,e))), tuple(sorted((a,d,f))),
                tuple(sorted((b,c,e))), tuple(sorted((b,c,f))),
                tuple(sorted((b,d,e))), tuple(sorted((b,d,f))),
            ]
            # Convert to indices
            t_indices = [triple_index[t] for t in transversal]
            cfg_key = frozenset(t_indices)
            if cfg_key not in seen_configs:
                seen_configs.add(cfg_key)
                k222_configs.append(list(cfg_key))

    num_configs = len(k222_configs)
    
    # Map triple -> list of config indices it belongs to
    triple_to_configs = [[] for _ in range(num_triples)]
    for ci, cfg in enumerate(k222_configs):
        for ti in cfg:
            triple_to_configs[ti].append(ci)

    # Data structures for fast checking
    # in_selected[ti] = 1 if triple ti is in solution
    # config_count[ci] = number of triples from config ci in solution
    # A config is "full" if config_count[ci] == 8
    
    def fast_greedy(seed_offset=0):
        rng = random.Random(42 + seed_offset)
        in_selected = [0] * num_triples
        config_count = [0] * num_configs
        selected = []
        
        # Shuffle triples
        items = list(range(num_triples))
        rng.shuffle(items)
        
        for ti in items:
            # Check if adding ti is safe
            safe = True
            for ci in triple_to_configs[ti]:
                if config_count[ci] == 7:
                    safe = False
                    break
            
            if safe:
                in_selected[ti] = 1
                selected.append(ti)
                for ci in triple_to_configs[ti]:
                    config_count[ci] += 1
                    
        return selected, in_selected, config_count

    def local_search(initial_selected, max_steps=5000, seed_offset=0):
        rng = random.Random(2000 + seed_offset)
        in_selected = [0] * num_triples
        config_count = [0] * num_configs
        selected = list(initial_selected)
        
        for ti in selected:
            in_selected[ti] = 1
            for ci in triple_to_configs[ti]:
                config_count[ci] += 1
        
        current_size = len(selected)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Try to add a random unselected triple
            # We need a list of unselected triples for efficient random picking
            # Rebuilding this list every step is O(N). 
            # Alternative: Pick random ti, check if not selected.
            
            ti = rng.randrange(num_triples)
            if in_selected[ti]:
                continue
            
            # Check if adding ti is safe
            blocking_configs = []
            safe = True
            for ci in triple_to_configs[ti]:
                if config_count[ci] == 7:
                    blocking_configs.append(ci)
                    safe = False
                # If config_count[ci] < 7, it's safe for now
            
            if safe:
                # Add it
                in_selected[ti] = 1
                selected.append(ti)
                current_size += 1
                for ci in triple_to_configs[ti]:
                    config_count[ci] += 1
                continue
            
            # If not safe, we must remove one triple from one of the blocking configs
            # to make room for ti.
            # Pick a random blocking config
            if not blocking_configs:
                continue
                
            ci = rng.choice(blocking_configs)
            # Get the 7 other triples in this config that are selected
            # We need to find which triples in k222_configs[ci] are currently selected
            # and are NOT ti.
            candidates_to_remove = []
            for other_ti in k222_configs[ci]:
                if other_ti != ti and in_selected[other_ti]:
                    candidates_to_remove.append(other_ti)
            
            if not candidates_to_remove:
                continue
                
            # Pick one to remove
            remove_ti = rng.choice(candidates_to_remove)
            
            # Check if removing remove_ti and adding ti is a valid swap
            # 1. Remove remove_ti
            in_selected[remove_ti] = 0
            selected.remove(remove_ti) # O(N) but N is small-ish (560)
            current_size -= 1
            for c in triple_to_configs[remove_ti]:
                config_count[c] -= 1
            
            # 2. Try to add ti
            # Re-check safety for ti after removal
            safe_to_add = True
            for c in triple_to_configs[ti]:
                if config_count[c] == 7:
                    safe_to_add = False
                    break
            
            if safe_to_add:
                in_selected[ti] = 1
                selected.append(ti)
                current_size += 1
                for c in triple_to_configs[ti]:
                    config_count[c] += 1
            else:
                # Rollback
                in_selected[remove_ti] = 1
                selected.append(remove_ti)
                current_size += 1
                for c in triple_to_configs[remove_ti]:
                    config_count[c] += 1
                # ti was not added
                
        return selected

    # Main loop
    best_selected = []
    best_size = 0
    
    # Phase 1: Multiple Greedy Runs
    num_greedy = 50
    for i in range(num_greedy):
        if time.time() > deadline - 30:
            break
        sel, _, _ = fast_greedy(seed_offset=i)
        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = sel
            
    # Phase 2: Local Search on Best and Random Starts
    while time.time() < deadline - 10:
        if time.time() < deadline - 40 and random.random() < 0.3:
            start_sel, _, _ = fast_greedy(seed_offset=random.randint(0, 10000))
        else:
            start_sel = best_selected
            
        ls_sel = local_search(start_sel, max_steps=2000, seed_offset=random.randint(0, 10000))
        
        if len(ls_sel) > best_size:
            best_size = len(ls_sel)
            best_selected = ls_sel

    # Final conversion
    result = [list(all_triples[ti]) for ti in sorted(best_selected)]
    return result
