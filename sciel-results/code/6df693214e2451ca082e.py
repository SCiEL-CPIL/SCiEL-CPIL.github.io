import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute pairs for each triple
    triple_pairs = []
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        triple_pairs.append(pairs)
        
    # Precompute subsets of size 7 and 6
    subsets_7 = list(combinations(points, 7))
    subsets_6 = list(combinations(points, 6))
    n_subsets_7 = len(subsets_7)
    n_subsets_6 = len(subsets_6)
    
    # Map subset to index
    subset_7_index = {s: i for i, s in enumerate(subsets_7)}
    subset_6_index = {s: i for i, s in enumerate(subsets_6)}
    
    # For each triple, find which subsets contain it
    # triple_to_subsets_7[t] = list of indices in subsets_7 that contain t
    triple_to_subsets_7 = {t: [] for t in all_triples}
    triple_to_subsets_6 = {t: [] for t in all_triples}
    
    # This precomputation might take a bit, but 560 * (11440 + 8008) is ~ 11M ops.
    # We can optimize by iterating subsets and finding triples?
    # No, for each triple, generate subsets containing it.
    # For triple t, pick 4 points from remaining 13.
    remaining = [p for p in points if p not in t]
    for t in all_triples:
        remaining = [p for p in points if p not in t]
        # Subsets of size 7 containing t: t + 4 from remaining
        for s4 in combinations(remaining, 4):
            s7 = tuple(sorted(t + s4))
            idx = subset_7_index[s7]
            triple_to_subsets_7[t].append(idx)
            
        # Subsets of size 6 containing t: t + 3 from remaining
        for s3 in combinations(remaining, 3):
            s6 = tuple(sorted(t + s3))
            idx = subset_6_index[s6]
            triple_to_subsets_6[t].append(idx)
            
    # State variables
    curr_triples = []
    curr_set = set()
    pair_used = set()
    
    count_7 = np.zeros(n_subsets_7, dtype=np.int32)
    count_6 = np.zeros(n_subsets_6, dtype=np.int32)
    
    N_ind_7 = n_subsets_7
    N_ind_6 = n_subsets_6
    
    best_triples = []
    best_alpha = 16
    best_N_ind_7 = n_subsets_7
    best_N_ind_6 = n_subsets_6
    
    def add_triple(t):
        nonlocal N_ind_7, N_ind_6
        curr_triples.append(t)
        curr_set.add(t)
        for p in triple_pairs[all_triples.index(t)]:
            pair_used.add(p)
        
        idx_t = all_triples.index(t)
        # Update counts for 7-subsets
        for idx in triple_to_subsets_7[t]:
            if count_7[idx] == 0:
                N_ind_7 -= 1
            count_7[idx] += 1
            
        # Update counts for 6-subsets
        for idx in triple_to_subsets_6[t]:
            if count_6[idx] == 0:
                N_ind_6 -= 1
            count_6[idx] += 1
                
    def remove_triple(t):
        nonlocal N_ind_7, N_ind_6
        curr_triples.remove(t)
        curr_set.remove(t)
        for p in triple_pairs[all_triples.index(t)]:
            pair_used.remove(p)
            
        for idx in triple_to_subsets_7[t]:
            count_7[idx] -= 1
            if count_7[idx] == 0:
                N_ind_7 += 1
                
        for idx in triple_to_subsets_6[t]:
            count_6[idx] -= 1
            if count_6[idx] == 0:
                N_ind_6 += 1

    # Initialize with a greedy packing
    random.shuffle(all_triples)
    for t in all_triples:
        pairs = triple_pairs[all_triples.index(t)]
        if not any(p in pair_used for p in pairs):
            add_triple(t)
            
    # Calculate initial alpha
    # If N_ind_7 == 0, alpha <= 6.
    # If N_ind_6 == 0, alpha <= 5.
    # We can determine alpha exactly if needed, but for SA we use N_ind as proxy.
    # However, to report best_alpha, we need the exact value.
    # We can calculate exact alpha at the end or periodically.
    
    def get_exact_alpha(triples):
        if not triples:
            return 16
        masks = []
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            masks.append(mask)
            
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                is_ind = True
                for m in masks:
                    if (m & mask) == m:
                        is_ind = False
                        break
                if is_ind:
                    return k
        return 0

    curr_alpha = get_exact_alpha(curr_triples)
    best_triples = list(curr_triples)
    best_alpha = curr_alpha
    best_N_ind_7 = N_ind_7
    best_N_ind_6 = N_ind_6
    
    # Simulated Annealing
    temperature = 1.0
    cooling_rate = 0.995
    min_temp = 0.01
    
    # Objective: Minimize N_ind_6 (stronger constraint) then N_ind_7
    # Actually, minimizing N_ind_6 is sufficient for alpha <= 5.
    # If we minimize N_ind_7, we ensure alpha <= 6.
    # Let's minimize a combined score: 100 * N_ind_6 + N_ind_7
    
    def get_score():
        return 100 * N_ind_6 + N_ind_7

    curr_score = get_score()
    best_score = curr_score
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        if not curr_triples:
            break
            
        # Pick a triple to remove
        t_remove = random.choice(curr_triples)
        remove_triple(t_remove)
        
        # Pick a triple to add
        # Must be compatible with current set
        added = False
        for _ in range(50): # Try 50 random candidates
            t_add = random.choice(all_triples)
            if t_add in curr_set:
                continue
            pairs = triple_pairs[all_triples.index(t_add)]
            if not any(p in pair_used for p in pairs):
                # Check if adding improves score
                old_score = get_score()
                add_triple(t_add)
                new_score = get_score()
                
                delta = new_score - old_score
                if delta < 0 or random.random() < np.exp(-delta / temperature):
                    added = True
                    curr_score = new_score
                    if curr_score < best_score:
                        best_score = curr_score
                        best_N_ind_7 = N_ind_7
                        best_N_ind_6 = N_ind_6
                        # Update best_triples and alpha periodically or if score is 0
                        if best_score == 0:
                             # Found alpha <= 5?
                             pass
                        else:
                             # Only calculate exact alpha if we improved significantly or periodically
                             if steps % 100 == 0 or delta < -10:
                                 curr_alpha = get_exact_alpha(curr_triples)
                                 if curr_alpha < best_alpha:
                                     best_alpha = curr_alpha
                                     best_triples = list(curr_triples)
                else:
                    # Reject
                    remove_triple(t_add)
                break
        if not added:
            # Restore
            add_triple(t_remove)
            
        temperature *= cooling_rate
        if temperature < min_temp:
            temperature = min_temp
            
    # Final check
    if best_triples:
        final_alpha = get_exact_alpha(best_triples)
        if final_alpha < best_alpha:
            best_alpha = final_alpha
            
    return best_triples
