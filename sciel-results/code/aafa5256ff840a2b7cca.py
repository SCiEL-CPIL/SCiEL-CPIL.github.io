import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Extended deadline for more search time
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute for each triple, how many K(2,2,2) configs it belongs to
    # This helps with greedy ordering
    
    # Precompute K(2,2,2) configurations
    # A K(2,2,2) is formed by 3 disjoint pairs.
    # There are 15 ways to partition 6 items into 3 pairs.
    # Total configs: C(16,6) * 15
    k222_list = []
    triple_to_k222s = [[] for _ in range(num_triples)]
    config_triples = [] # List of lists of triple indices
    
    for points in combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = points
        # Generate 15 pairings
        pairings = [
            ((p0,p1),(p2,p3),(p4,p5)),
            ((p0,p1),(p2,p4),(p3,p5)),
            ((p0,p1),(p2,p5),(p3,p4)),
            ((p0,p2),(p1,p3),(p4,p5)),
            ((p0,p2),(p1,p4),(p3,p5)),
            ((p0,p2),(p1,p5),(p3,p4)),
            ((p0,p3),(p1,p2),(p4,p5)),
            ((p0,p3),(p1,p4),(p2,p5)),
            ((p0,p3),(p1,p5),(p2,p4)),
            ((p0,p4),(p1,p2),(p3,p5)),
            ((p0,p4),(p1,p3),(p2,p5)),
            ((p0,p4),(p1,p5),(p2,p3)),
            ((p0,p5),(p1,p2),(p3,p4)),
            ((p0,p5),(p1,p3),(p2,p4)),
            ((p0,p5),(p1,p4),(p2,p3)),
        ]
        
        for pair1, pair2, pair3 in pairings:
            transversals = []
            for x in pair1:
                for y in pair2:
                    for z in pair3:
                        transversals.append((x, y, z))
            
            trans_sorted = [tuple(sorted(t)) for t in transversals]
            trans_indices = [triple_to_idx[t] for t in trans_sorted]
            
            k_idx = len(k222_list)
            k222_list.append(set(trans_indices))
            config_triples.append(trans_indices)
            
            for idx in trans_indices:
                triple_to_k222s[idx].append(k_idx)
    
    num_k222s = len(k222_list)
    
    def time_left():
        return deadline - time.time()

    def greedy_init(strategy="random"):
        active = [False] * num_triples
        counts = [0] * num_k222s
        current_size = 0
        
        if strategy == "random":
            order = list(range(num_triples))
            random.shuffle(order)
        elif strategy == "degree":
            # Sort by number of K(2,2,2) configs (fewer constraints first)
            order = sorted(range(num_triples), key=lambda i: len(triple_to_k222s[i]))
            random.shuffle(order)  # Break ties randomly
        else:
            order = list(range(num_triples))
            random.shuffle(order)
        
        for t_idx in order:
            valid = True
            for k_idx in triple_to_k222s[t_idx]:
                if counts[k_idx] == 7:
                    valid = False
                    break
            if valid:
                active[t_idx] = True
                current_size += 1
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] += 1
        return active, counts, current_size

    def is_addable(t_idx, counts):
        for k_idx in triple_to_k222s[t_idx]:
            if counts[k_idx] == 7:
                return False
        return True

    def get_blockers(t_idx, counts):
        """Returns list of config indices c where counts[c] == 7"""
        blockers = []
        for k_idx in triple_to_k222s[t_idx]:
            if counts[k_idx] == 7:
                blockers.append(k_idx)
        return blockers

    def local_search(active, counts, current_size):
        improved = True
        addable_cache = []  # Cache of currently addable edges
        
        while improved and time_left() > 2:
            improved = False
            
            # 1. Try to add any edge - use cached list if available, else rebuild
            if not addable_cache:
                addable_cache = [i for i in range(num_triples) if not active[i] and is_addable(i, counts)]
            
            if addable_cache:
                # Pick a random addable edge
                t_idx = random.choice(addable_cache)
                active[t_idx] = True
                current_size += 1
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] += 1
                # Invalidate cache since counts changed
                addable_cache = []
                improved = True
                continue
            
            # 2. If no addable, try swaps
            inactive_edges = [i for i in range(num_triples) if not active[i]]
            if not inactive_edges:
                break # Full set (unlikely)
                
            # Try swaps with a more systematic approach
            found_swap = False
            num_swap_attempts = 150
            for _ in range(num_swap_attempts):
                if time_left() < 1: break
                t_idx = random.choice(inactive_edges)
                
                # Check if t_idx is blocked
                blockers = get_blockers(t_idx, counts)
                if not blockers:
                    # Should have been added in step 1, but just in case
                    active[t_idx] = True
                    current_size += 1
                    for k_idx in triple_to_k222s[t_idx]:
                        counts[k_idx] += 1
                    improved = True
                    found_swap = True
                    break
                
                # Strategy: For each blocker config, try removing an active triple that
                # minimizes the impact (i.e., belongs to few other blocked configs)
                random.shuffle(blockers)
                swap_success = False
                for c_idx in blockers:
                    candidates = []
                    for t_rem in config_triples[c_idx]:
                        if active[t_rem]:
                            # Score: how many other blocked configs does t_rem belong to?
                            # Fewer is better (less collateral damage)
                            score = 0
                            for k2 in triple_to_k222s[t_rem]:
                                if k2 != c_idx and counts[k2] == 7:
                                    score += 1
                            candidates.append((score, t_rem))
                    
                    if not candidates:
                        continue
                    
                    # Sort by score (fewer collateral blocked configs first)
                    candidates.sort()
                    for _, t_rem in candidates:
                        # Temporarily remove t_rem
                        active[t_rem] = False
                        for k_idx in triple_to_k222s[t_rem]:
                            counts[k_idx] -= 1
                        
                        # Check if t_idx is now addable
                        if is_addable(t_idx, counts):
                            # Perform swap
                            active[t_idx] = True
                            for k_idx in triple_to_k222s[t_idx]:
                                counts[k_idx] += 1
                            improved = True
                            found_swap = True
                            swap_success = True
                            break
                        else:
                            # Restore t_rem
                            active[t_rem] = True
                            for k_idx in triple_to_k222s[t_rem]:
                                counts[k_idx] += 1
                        
                        if swap_success:
                            break
                    
                    if swap_success:
                        break
                
                if found_swap:
                    break
                    
            if not found_swap:
                break

        return active, counts, current_size

    best_size = 0
    best_active = None
    best_counts = None
    
    # Try multiple initialization strategies
    strategies = ["random", "degree", "random"]
    for strat in strategies:
        if time_left() < 10:
            break
        active, counts, current_size = greedy_init(strategy=strat)
        active, counts, current_size = local_search(active, counts, current_size)
        
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]
    
    # Use the best solution as starting point for ILS
    active = best_active[:]
    counts = best_counts[:]
    current_size = best_size
    
    # 2. Iterated Local Search with adaptive perturbation
    no_improve_count = 0
    while time_left() > 5:
        # Adaptive perturbation: increase when stuck, decrease when improving
        if no_improve_count > 5:
            k = random.randint(15, 30)  # Larger perturbation when stuck
        elif no_improve_count > 2:
            k = random.randint(10, 20)
        else:
            k = random.randint(5, 15)  # Smaller perturbation when making progress
        
        active_edges = [i for i in range(num_triples) if active[i]]
        if len(active_edges) < k:
            k = len(active_edges)
            
        # Shuffle active edges and remove first k
        random.shuffle(active_edges)
        for i in range(k):
            t_idx = active_edges[i]
            active[t_idx] = False
            current_size -= 1
            for k_idx in triple_to_k222s[t_idx]:
                counts[k_idx] -= 1
                
        # Local Search on perturbed solution
        active, counts, current_size = local_search(active, counts, current_size)
        
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Revert to best solution if perturbation didn't help
            active = best_active[:]
            counts = best_counts[:]
            current_size = best_size
            
    # Construct result
    if best_active is None:
        active, counts, current_size = greedy_init()
        best_active = active
        
    result = []
    for i in range(num_triples):
        if best_active[i]:
            result.append(list(all_triples[i]))
            
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")
