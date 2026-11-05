"""
Maximize number of triples in a 3-uniform hypergraph on 16 vertices
such that no three disjoint pairs have all 8 transversal triples present.

Strategy:
1. Precompute all K(2,2,2) configurations (each is a set of 8 triples).
2. Use local search with efficient add/swap moves to maximize
   the number of triples while keeping all K(2,2,2) constraints satisfied.
3. Use greedy initialization with randomization for diverse starting points.
4. Time-bounded with multiple restarts.
"""

import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_total = len(all_triples)  # 560
    
    # Precompute all K(2,2,2) configurations
    k222_configs = []
    triple_to_k222 = [[] for _ in range(num_total)]
    
    for six_vertices in itertools.combinations(range(N), 6):
        a, b, c, d, e, f = six_vertices
        partitions = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for p1, p2, p3 in partitions:
            transversals = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        transversals.append(t)
            trans_indices = frozenset(triple_to_idx[t] for t in transversals)
            k222_configs.append(trans_indices)
            config_idx = len(k222_configs) - 1
            for ti in trans_indices:
                triple_to_k222[ti].append(config_idx)
    
    num_k222 = len(k222_configs)
    
    def time_check():
        return time.time() >= deadline
    
    best_result = []
    best_count = 0
    
    # Greedy initialization
    def greedy_init(rng):
        present = [False] * num_total
        k222_count = [0] * num_k222
        count = 0
        
        order = list(range(num_total))
        rng.shuffle(order)
        
        for ti in order:
            can_add = True
            for ci in triple_to_k222[ti]:
                if k222_count[ci] == 7:
                    can_add = False
                    break
            if can_add:
                present[ti] = True
                count += 1
                for ci in triple_to_k222[ti]:
                    k222_count[ci] += 1
        return present, k222_count, count
    
    # Local search with efficient moves
    def local_search(present, k222_count, count, rng, max_steps=500000):
        steps = 0
        no_improve = 0
        max_no_improve = 200
        
        # List of triples that can be added
        # We will maintain this dynamically or recompute if it becomes empty
        
        while steps < max_steps and no_improve < max_no_improve:
            if time_check():
                break
            
            # Phase 1: Try to add triples
            # Find all addable triples
            addable = []
            for ti in range(num_total):
                if present[ti]:
                    continue
                can_add = True
                for ci in triple_to_k222[ti]:
                    if k222_count[ci] == 7:
                        can_add = False
                        break
                if can_add:
                    addable.append(ti)
            
            if addable:
                # Add as many as possible (they don't conflict with each other? 
                # Actually, adding one might block another. 
                # But if we just added one, the k222 counts changed.
                # So we should add one, update, and re-check?
                # Or add all if we are sure they don't conflict?
                # They might conflict. E.g. t1 and t2 share a k222 config where count is 6.
                # Adding t1 makes count 7, blocking t2.
                # So we should add them one by one, re-evaluating availability.
                # However, for speed, we can just pick one random addable triple, add it, and loop.
                # This is O(1) per add.
                ti = rng.choice(addable)
                present[ti] = True
                count += 1
                for ci in triple_to_k222[ti]:
                    k222_count[ci] += 1
                no_improve = 0
                steps += 1
                continue
            
            # Phase 2: No addable triples. We are saturated.
            # We need to remove a triple to unlock others.
            # Strategy: For each present triple, calculate how many missing triples it uniquely blocks.
            # If removing t_pres unlocks k missing triples, and we can add all of them, net gain is k-1.
            # We want to maximize k.
            
            # To do this efficiently:
            # 1. Identify all missing triples.
            # 2. For each missing triple t_miss, find the set of present triples that block it.
            #    A present triple t_pres blocks t_miss if they share a k222 config ci where k222_count[ci] == 7 
            #    AND t_pres is in that config.
            #    Actually, simpler: t_miss is blocked by t_pres if they share a config ci 
            #    such that k222_count[ci] == 7 and t_pres is in config ci.
            #    Note: Since t_miss is missing, it's not in the count.
            #    If we remove t_pres, count[ci] becomes 6. Then t_miss can be added if no OTHER present 
            #    triple blocks it via a DIFFERENT config, or the SAME config if count was >7? No, count is 7.
            #    So t_miss becomes unblocked by t_pres.
            #    If t_miss is blocked by ONLY t_pres (among all present triples), then removing t_pres 
            #    makes t_miss addable.
            
            # Let's build a map: blocker_count[t_pres] = number of missing triples uniquely blocked by t_pres.
            # And maybe a list of those triples?
            
            # Optimization: 
            # Iterate through missing triples.
            # For each missing triple, find all present triples that block it.
            # If exactly 1, increment score for that present triple.
            
            # This might be slow if done naively for every step.
            # But we only do it when saturated.
            
            blocker_scores = [0] * num_total
            # We don't strictly need the list of unblocked triples, just the count for the score.
            # But we DO need to know WHICH triples to add after removal.
            # So we need to map t_pres -> list of t_miss it uniquely blocks.
            
            unique_blockers = [[] for _ in range(num_total)]
            
            # Iterate missing triples
            for ti_miss in range(num_total):
                if present[ti_miss]:
                    continue
                
                # Find present triples blocking ti_miss
                blockers = []
                for ci in triple_to_k222[ti_miss]:
                    if k222_count[ci] == 7:
                        # This config is full.
                        # Which present triples are in this config?
                        # We need to know which present triples are in k222_configs[ci].
                        # k222_configs[ci] is a frozenset of indices.
                        # We can intersect with present set?
                        # Or iterate through the config and check present[]?
                        
                        # Since config size is 8, this is fast.
                        for t_in_config in k222_configs[ci]:
                            if present[t_in_config]:
                                blockers.append(t_in_config)
                
                # Remove duplicates in blockers (a triple might block via multiple configs? 
                # No, if t_pres and t_miss share two configs, and both are full, 
                # t_pres is still just one blocker. But we just need to know if it's UNIQUE.)
                unique_blockers_set = set(blockers)
                
                if len(unique_blockers_set) == 1:
                    t_pres = list(unique_blockers_set)[0]
                    unique_blockers[t_pres].append(ti_miss)
                    blocker_scores[t_pres] += 1
            
            # Find the present triple with max score
            best_remove = -1
            max_score = 0
            for ti in range(num_total):
                if present[ti] and blocker_scores[ti] > max_score:
                    max_score = blocker_scores[ti]
                    best_remove = ti
            
            if best_remove == -1:
                # No single triple uniquely blocks any missing triple.
                # This means every missing triple is blocked by >= 2 present triples.
                # Removing one won't make any addable immediately.
                # We might need a 2-swap (remove 2, add 3+).
                # For now, consider this a local optimum for 1-swap.
                # Try a random removal to escape?
                # Or break?
                # Let's try a random removal of a present triple that blocks the MOST triples (not necessarily uniquely).
                
                # Calculate total blocks (not unique)
                total_block_scores = [0] * num_total
                for ti_miss in range(num_total):
                    if present[ti_miss]:
                        continue
                    blockers = set()
                    for ci in triple_to_k222[ti_miss]:
                        if k222_count[ci] == 7:
                            for t_in_config in k222_configs[ci]:
                                if present[t_in_config]:
                                    blockers.add(t_in_config)
                    for b in blockers:
                        total_block_scores[b] += 1
                
                best_remove = -1
                max_score = 0
                for ti in range(num_total):
                    if present[ti] and total_block_scores[ti] > max_score:
                        max_score = total_block_scores[ti]
                        best_remove = ti
                
                if best_remove == -1:
                    # Truly stuck?
                    no_improve += 1
                    steps += 1
                    continue
            
            # Perform the swap
            ti_remove = best_remove
            for ci in triple_to_k222[ti_remove]:
                k222_count[ci] -= 1
            present[ti_remove] = False
            count -= 1
            
            # Now add all triples that were uniquely blocked by ti_remove
            # Note: Some of these might now be blocked by OTHER triples?
            # No, they were uniquely blocked by ti_remove. 
            # But adding one might block another?
            # If t1 and t2 were both uniquely blocked by t_remove, 
            # do they conflict?
            # If they share a config, and that config was full (7), 
            # and t_remove was the only present one in it...
            # Wait, if t1 and t2 share a config C, and C was full (7 present), 
            # and t_remove is in C.
            # If t_remove is the ONLY present triple in C that blocks t1?
            # That implies all other present triples in C do NOT block t1?
            # But if they are in C, they DO block t1 (since C is full).
            # So if t1 and t2 share C, and C is full, then ALL present triples in C block t1 and t2.
            # So if there is more than 1 present triple in C, t1 is not uniquely blocked.
            # Thus, if t1 and t2 are uniquely blocked by t_remove, they cannot share a full config 
            # with any other present triple.
            # Can they share a full config with EACH OTHER? No, they are missing.
            # So they don't conflict with each other via that config.
            # Do they conflict via another config?
            # If they share another config D, and D is full, then t_remove must be the unique blocker in D too?
            # If so, they are both in D.
            # This logic holds: if they are uniquely blocked by t_remove, they are safe to add simultaneously 
            # relative to each other?
            # Actually, adding t1 increases counts in its configs.
            # If t1 and t2 share a config D where count was 6 (not full), adding t1 makes it 7.
            # Then t2 (if in D) becomes blocked.
            # So we should add them one by one, checking validity.
            
            added_count = 0
            for ti_add in unique_blockers[ti_remove]:
                if present[ti_add]: # Should be false, but safe
                    continue
                can_add = True
                for ci in triple_to_k222[ti_add]:
                    if k222_count[ci] == 7:
                        can_add = False
                        break
                if can_add:
                    present[ti_add] = True
                    count += 1
                    added_count += 1
                    for ci in triple_to_k222[ti_add]:
                        k222_count[ci] += 1
            
            if added_count > 0:
                no_improve = 0
            else:
                no_improve += 1
            
            steps += 1
        
        return present, count
    
    # Main loop with multiple restarts
    rng = random.Random(42)
    
    num_restarts = 0
    while not time_check() and num_restarts < 500:
        num_restarts += 1
        
        # Greedy init with randomization
        present, k222_count, count = greedy_init(rng)
        
        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]
        
        # Local search
        present, count = local_search(present, k222_count, count, rng)
        
        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]
        
        if time_check():
            break
    
    if not best_result:
        # Fallback: return a valid simple solution
        best_result = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_result
