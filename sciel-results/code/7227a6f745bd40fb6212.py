import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # 1. Precompute all K(2,2,2) configurations
    # A K(2,2,2) is defined by 3 disjoint pairs.
    # We iterate over all sets of 6 points, then all pairings of those 6 points.
    # Number of configs: C(16,6) * 15 = 120120
    
    configs = [] # List of lists of 8 triple indices
    # To save memory/time, we might not need to store all if we can generate on fly, 
    # but 120k configs * 8 ints is small.
    
    # Map from triple index to list of config indices containing it
    triple_to_configs = [[] for _ in range(num_triples)]
    
    # We need a fast way to get the 8 triples for a given pairing of 6 points.
    # Let the 6 points be p0..p5.
    # Pairings are permutations of (0,1,2,3,4,5) modulo symmetries? 
    # There are 15 distinct pairings of 6 items.
    # We can hardcode the 15 pairings of indices 0..5.
    
    pairings = [
        ((0,1),(2,3),(4,5)),
        ((0,1),(2,4),(3,5)),
        ((0,1),(2,5),(3,4)),
        ((0,2),(1,3),(4,5)),
        ((0,2),(1,4),(3,5)),
        ((0,2),(1,5),(3,4)),
        ((0,3),(1,2),(4,5)),
        ((0,3),(1,4),(2,5)),
        ((0,3),(1,5),(2,4)),
        ((0,4),(1,2),(3,5)),
        ((0,4),(1,3),(2,5)),
        ((0,4),(1,5),(2,3)),
        ((0,5),(1,2),(3,4)),
        ((0,5),(1,3),(2,4)),
        ((0,5),(1,4),(2,3))
    ]
    
    # Generate configs
    for six_points in combinations(range(n), 6):
        # six_points is a tuple of 6 distinct points
        # Map index 0-5 to actual points
        pts = six_points
        for pairing in pairings:
            p1 = (pts[pairing[0][0]], pts[pairing[0][1]])
            p2 = (pts[pairing[1][0]], pts[pairing[1][1]])
            p3 = (pts[pairing[2][0]], pts[pairing[2][1]])
            
            # Generate the 8 transversals
            transversals = []
            for a in p1:
                for b in p2:
                    for c in p3:
                        # Ensure sorted tuple for lookup
                        t = tuple(sorted((a, b, c)))
                        transversals.append(t)
            
            # Convert to indices
            t_indices = [triple_to_idx[t] for t in transversals]
            
            # Store config
            config_idx = len(configs)
            configs.append(t_indices)
            
            # Update reverse mapping
            for t_idx in t_indices:
                triple_to_configs[t_idx].append(config_idx)
    
    # 2. Greedy Construction
    def greedy_construct(seed):
        random.seed(seed)
        selected = set()
        forbidden = set()
        counts = [0] * len(configs)
        
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        
        for t_idx in candidates:
            if time.time() > deadline - 2:
                break
                
            if t_idx in forbidden:
                continue
            
            selected.add(t_idx)
            
            # Update counts for all configs containing t_idx
            for k in triple_to_configs[t_idx]:
                counts[k] += 1
                if counts[k] == 7:
                    # Find the missing triple in this config
                    for t_candidate in configs[k]:
                        if t_candidate not in selected:
                            forbidden.add(t_candidate)
                            break
                            
        return selected, forbidden, counts

    def local_search(selected, forbidden, counts, seed, time_budget=5.0):
        """Perform swap-based local search to escape local minima."""
        random.seed(seed)
        start_time = time.time()
        improved = True
        
        while improved and (time.time() - start_time) < time_budget:
            improved = False
            
            # Try to find a beneficial swap: remove one selected, add two new ones
            # Or simpler: try to add any non-forbidden, non-selected triple
            # First pass: try to add any available triple
            unselected = [t for t in range(num_triples) if t not in selected and t not in forbidden]
            random.shuffle(unselected)
            
            added_any = False
            for t_idx in unselected:
                if time.time() > deadline - 2:
                    break
                if t_idx in forbidden or t_idx in selected:
                    continue
                selected.add(t_idx)
                added_any = True
                for k in triple_to_configs[t_idx]:
                    counts[k] += 1
                    if counts[k] == 7:
                        for t_candidate in configs[k]:
                            if t_candidate not in selected:
                                forbidden.add(t_candidate)
                                break
            
            if added_any:
                improved = True
                continue
            
            # Second pass: try swaps - remove one, add one or more
            # For each selected triple, try removing it and see if we can add 2+ new ones
            selected_list = list(selected)
            random.shuffle(selected_list)
            
            for t_remove in selected_list:
                if time.time() > deadline - 2:
                    break
                if t_remove not in selected:
                    continue
                
                # Tentatively remove t_remove
                selected.discard(t_remove)
                for k in triple_to_configs[t_remove]:
                    counts[k] -= 1
                
                # Save forbidden state
                old_forbidden = set(forbidden)
                # After removal, some previously forbidden may now be available
                # Recompute: a triple is forbidden if it's the 8th in any config with 7 selected
                # But we need to be careful - let's just try to add any non-selected triple
                added_count = 0
                for t_add in range(num_triples):
                    if time.time() > deadline - 2:
                        break
                    if t_add in selected or t_add == t_remove:
                        continue
                    # Check if adding t_add would violate
                    can_add = True
                    for k in triple_to_configs[t_add]:
                        if counts[k] >= 7:
                            can_add = False
                            break
                    if can_add:
                        selected.add(t_add)
                        added_count += 1
                        for k in triple_to_configs[t_add]:
                            counts[k] += 1
                            if counts[k] == 7:
                                for t_candidate in configs[k]:
                                    if t_candidate not in selected:
                                        forbidden.add(t_candidate)
                                        break
                
                # Only accept if we gained net triples
                if added_count > 1:
                    improved = True
                else:
                    # Revert
                    selected.discard(t_remove)
                    for k in triple_to_configs[t_remove]:
                        counts[k] += 1
                    # Restore forbidden
                    forbidden.clear()
                    forbidden.update(old_forbidden)
                    # Also restore any newly forbidden from the failed attempt
                    for t_add in list(selected):
                        if t_add != t_remove:
                            for k in triple_to_configs[t_add]:
                                if counts[k] == 7:
                                    for t_candidate in configs[k]:
                                        if t_candidate not in selected:
                                            forbidden.add(t_candidate)
                                            break
                    break  # Move to next remove candidate
        
        return selected, forbidden, counts

    # Run multiple restarts with local search
    best_result = set()
    best_size = 0
    num_restarts = 15  # Adjust based on speed
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        selected, forbidden, counts = greedy_construct(seed=i)
        
        # Apply local search if we have time
        remaining = deadline - time.time()
        if remaining > 8:
            selected, forbidden, counts = local_search(selected, forbidden, counts, seed=i, time_budget=min(4.0, remaining - 5))
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = set(selected)
            
    # Convert indices back to triples
    return [list(all_triples[idx]) for idx in best_result]
