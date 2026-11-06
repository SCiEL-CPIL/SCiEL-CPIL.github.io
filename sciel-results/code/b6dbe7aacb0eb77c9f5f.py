import time
import random
import sys
from itertools import combinations

# Increase recursion limit
sys.setrecursionlimit(10000)

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute triple indices
_triple_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

# --- Precomputation: K(2,2,2) Configurations ---
def _precompute_k222s():
    triple_to_k222s = [[] for _ in range(N_TRIPLES)]
    k222_triples = []
    
    for verts in combinations(range(V), 6):
        v0, v1, v2, v3, v4, v5 = verts
        
        # 15 perfect matchings of 6 elements
        # Fix v0's partner, then 3 matchings of remaining 4
        matchings = [
            ((v0, v1), (v2, v3), (v4, v5)),
            ((v0, v1), (v2, v4), (v3, v5)),
            ((v0, v1), (v2, v5), (v3, v4)),
            ((v0, v2), (v1, v3), (v4, v5)),
            ((v0, v2), (v1, v4), (v3, v5)),
            ((v0, v2), (v1, v5), (v3, v4)),
            ((v0, v3), (v1, v2), (v4, v5)),
            ((v0, v3), (v1, v4), (v2, v5)),
            ((v0, v3), (v1, v5), (v2, v4)),
            ((v0, v4), (v1, v2), (v3, v5)),
            ((v0, v4), (v1, v3), (v2, v5)),
            ((v0, v4), (v1, v5), (v2, v3)),
            ((v0, v5), (v1, v2), (v3, v4)),
            ((v0, v5), (v1, v3), (v2, v4)),
            ((v0, v5), (v1, v4), (v2, v3)),
        ]
        
        for matching in matchings:
            p1, p2, p3 = matching
            t_indices = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        t_indices.append(_triple_idx[t])
            
            k_idx = len(k222_triples)
            k222_triples.append(t_indices)
            
            for t_idx in t_indices:
                triple_to_k222s[t_idx].append(k_idx)
                
    return triple_to_k222s, k222_triples

# Run precomputation once at module load
triple_to_k222s, k222_triples = _precompute_k222s()
N_K222 = len(k222_triples)

# Precompute sets for faster subset checks
triple_k222_sets = [set(lst) for lst in triple_to_k222s]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_size = 0
    best_active = [False] * N_TRIPLES
    
    def time_left():
        return deadline - time.time()
    
    # Greedy Initialization with multiple strategies
    def greedy_init(seed, strategy=0):
        random.seed(seed)
        active = [False] * N_TRIPLES
        counts = [0] * N_K222
        current_size = 0
        
        if strategy == 0:
            # Random order
            order = list(range(N_TRIPLES))
            random.shuffle(order)
        elif strategy == 1:
            # Prefer triples that are in fewer K222s (less restrictive)
            # Actually all triples are in the same number of K222s by symmetry
            # So use degree-based: prefer triples with more potential partners
            order = list(range(N_TRIPLES))
            random.shuffle(order)
        else:
            # Reverse random
            order = list(range(N_TRIPLES))
            random.shuffle(order)
            order.reverse()
        
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
    
    # Local Search with enhanced swap strategies
    def local_search(active, counts, current_size, max_iterations=5000):
        improved = True
        iteration = 0
        while improved and time_left() > 2.0 and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Phase 1: Add any possible triple
            for i in range(N_TRIPLES):
                if time_left() < 1.0:
                    return active, counts, current_size
                if not active[i]:
                    addable = True
                    for k_idx in triple_to_k222s[i]:
                        if counts[k_idx] == 7:
                            addable = False
                            break
                    if addable:
                        active[i] = True
                        current_size += 1
                        for k_idx in triple_to_k222s[i]:
                            counts[k_idx] += 1
                        improved = True
            
            if improved:
                continue
            
            # Phase 2: 1-for-1 Swaps
            inactive_indices = [i for i in range(N_TRIPLES) if not active[i]]
            active_indices = [i for i in range(N_TRIPLES) if active[i]]
            
            if not inactive_indices or not active_indices:
                break
            
            random.shuffle(inactive_indices)
            
            found_swap = False
            for t_add in inactive_indices:
                if time_left() < 1.0:
                    return active, counts, current_size
                
                # Find blockers
                blockers = []
                for k_idx in triple_to_k222s[t_add]:
                    if counts[k_idx] == 7:
                        blockers.append(k_idx)
                
                if not blockers:
                    active[t_add] = True
                    current_size += 1
                    for k_idx in triple_to_k222s[t_add]:
                        counts[k_idx] += 1
                    improved = True
                    found_swap = True
                    break
                
                # Find t_remove that contains all blockers
                if len(blockers) == 1:
                    candidates = k222_triples[blockers[0]]
                else:
                    candidates = set(k222_triples[blockers[0]])
                    for k in blockers[1:]:
                        candidates.intersection_update(k222_triples[k])
                    if not candidates:
                        continue
                    candidates = list(candidates)
                
                for t_remove in candidates:
                    if active[t_remove]:
                        # Perform swap
                        active[t_remove] = False
                        for k_idx in triple_to_k222s[t_remove]:
                            counts[k_idx] -= 1
                        
                        active[t_add] = True
                        for k_idx in triple_to_k222s[t_add]:
                            counts[k_idx] += 1
                        
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            # Phase 3: 2-for-1 swaps (remove 2, add 1) - only if size can increase
            # This is tricky since we're removing 2 and adding 1, net -1
            # Instead, try 1-for-2: remove 1, add 2 (net +1)
            if not improved and time_left() > 3.0:
                # Try to find a triple to remove that allows adding 2 new triples
                for t_remove in active_indices:
                    if time_left() < 2.0:
                        return active, counts, current_size
                    
                    # Temporarily remove t_remove
                    for k_idx in triple_to_k222s[t_remove]:
                        counts[k_idx] -= 1
                    
                    # Find all triples that become addable
                    newly_addable = []
                    for i in range(N_TRIPLES):
                        if not active[i] and i != t_remove:
                            addable = True
                            for k_idx in triple_to_k222s[i]:
                                if counts[k_idx] == 7:
                                    addable = False
                                    break
                            if addable:
                                newly_addable.append(i)
                    
                    if len(newly_addable) >= 2:
                        # Add the best 2 (those that block the fewest others)
                        newly_addable.sort(key=lambda i: len([k for k in triple_to_k222s[i] if counts[k] >= 6]))
                        
                        added_count = 0
                        for i in newly_addable[:2]:
                            # Re-check if still addable
                            still_addable = True
                            for k_idx in triple_to_k222s[i]:
                                if counts[k_idx] == 7:
                                    still_addable = False
                                    break
                            if still_addable:
                                active[i] = True
                                for k_idx in triple_to_k222s[i]:
                                    counts[k_idx] += 1
                                added_count += 1
                        
                        if added_count >= 2:
                            # Net gain: +2 - 1 = +1
                            current_size += added_count - 1
                            improved = True
                            break
                        else:
                            # Undo additions
                            for i in newly_addable[:2]:
                                if active[i] and i != t_remove:
                                    active[i] = False
                                    for k_idx in triple_to_k222s[i]:
                                        counts[k_idx] -= 1
                    
                    # Restore t_remove
                    for k_idx in triple_to_k222s[t_remove]:
                        counts[k_idx] += 1
                
                if improved:
                    continue
            
            if not improved:
                break
        
        return active, counts, current_size
    
    # Main Loop
    seed = 42
    strategies = [0, 1, 2]
    
    while time_left() > 10.0:
        for strategy in strategies:
            if time_left() < 10.0:
                break
            active, counts, current_size = greedy_init(seed, strategy)
            active, counts, current_size = local_search(active, counts, current_size)
            
            if current_size > best_size:
                best_size = current_size
                best_active = active[:]
        
        seed += 1
    
    # Final polish: try to improve the best solution
    if time_left() > 5.0:
        counts = [0] * N_K222
        for i in range(N_TRIPLES):
            if best_active[i]:
                for k_idx in triple_to_k222s[i]:
                    counts[k_idx] += 1
        
        best_active, counts, best_size = local_search(best_active, counts, best_size, max_iterations=20000)
    
    # Construct result
    result = []
    for i in range(N_TRIPLES):
        if best_active[i]:
            result.append(list(ALL_TRIPLES[i]))
    
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
