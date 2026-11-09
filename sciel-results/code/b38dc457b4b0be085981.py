import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    all_triples = []
    triple_masks = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                all_triples.append((i, j, k))
                mask = (1 << i) | (1 << j) | (1 << k)
                triple_masks.append(mask)
                
    num_triples = len(all_triples)
    
    def count_bits(x):
        return bin(x).count('1')
        
    def is_compatible(new_idx, current_indices, current_masks):
        new_mask = triple_masks[new_idx]
        for idx in current_indices:
            if count_bits(new_mask & current_masks[idx]) > 1:
                return False
        return True

    def compute_alpha(triples_list):
        if not triples_list:
            return n
            
        triples_masks_local = [triple_masks[i] for i in triples_list]
        
        # Check from largest size down
        for k in range(n, 0, -1):
            # If k is large, C(n, k) is small.
            # If k is small (e.g. 5), C(16, 5) = 4368.
            for comb in combinations(range(n), k):
                mask = 0
                for p in comb:
                    mask |= (1 << p)
                # Check if any triple is subset of mask
                indep = True
                for tm in triples_masks_local:
                    if (mask & tm) == tm:
                        indep = False
                        break
                if indep:
                    return k
        return 0

    best_alpha = n
    best_system = []
    
    # Use a seeded random for reproducibility in debugging, but random for diversity
    # random.seed(42) 
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
            
        current_triples = []
        current_indices = []
        current_masks = []
        
        # Randomized greedy
        # To improve coverage, we can bias towards points with low degree?
        # Let's just shuffle and pick.
        indices = list(range(num_triples))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            if is_compatible(idx, current_indices, current_masks):
                current_triples.append(all_triples[idx])
                current_indices.append(idx)
                current_masks.append(triple_masks[idx])
                
        if not current_triples:
            continue
            
        # Quick check: if number of triples is very small, alpha is likely high.
        # But let's compute it.
        alpha = compute_alpha(current_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = current_triples
            
        if best_alpha <= 4:
            break

    if not best_system:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return best_system
