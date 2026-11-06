import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute masks for all triples
    triple_masks = []
    for t in all_triples:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(m)
        
    # Precompute compatibility
    # compat[i] is a set of indices j such that triple i and j are compatible
    # Or just check on the fly? 
    # 560 triples. 560*560/2 ~ 156k checks. Fast.
    # Let's precompute a list of compatible partners for each triple to speed up greedy.
    compat = [set() for _ in range(len(all_triples))]
    for i in range(len(all_triples)):
        mi = triple_masks[i]
        for j in range(i+1, len(all_triples)):
            mj = triple_masks[j]
            if (mi & mj).bit_count() <= 1:
                compat[i].add(j)
                compat[j].add(i)
                
    best_system = None
    best_alpha = 16
    
    # Function to compute independence number
    def get_alpha(system_indices):
        if not system_indices:
            return 16
        
        # Create a set of masks for fast lookup? 
        # Or just iterate.
        sys_masks = [triple_masks[i] for i in system_indices]
        
        # Check subsets of size k from 16 down to 1
        # We can prune: if we find an independent set of size k, we don't need to check k+1?
        # No, we want the LARGEST.
        # So we check k=16, 15, ... until we find one.
        # But usually alpha is small (6-8).
        # It is faster to check from small k up? No, we want max.
        # If we check from 16 down, we stop as soon as we find one.
        # But if alpha is 6, we check 16, 15, ..., 7 (fail), 6 (success).
        # Number of subsets: C(16,16)+...+C(16,7).
        # C(16,7) = 11440.
        # Total checks ~ 20k.
        # Each check iterates over system_indices (max ~35).
        # 20k * 35 = 700k ops.
        
        for k in range(16, 0, -1):
            for subset in itertools.combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                
                is_independent = True
                for sm in sys_masks:
                    if (mask & sm) == sm:
                        is_independent = False
                        break
                
                if is_independent:
                    return k
        return 0

    # Random Greedy Search
    num_triples = len(all_triples)
    indices = list(range(num_triples))
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        random.shuffle(indices)
        
        current_system = []
        used_mask = 0 # Not strictly needed if we check compat, but good for quick reject?
        # Actually, compat check is O(1) if we use bitsets for compat too.
        # Let's stick to the set of indices.
        
        for idx in indices:
            # Check if idx is compatible with all in current_system
            compatible = True
            for existing_idx in current_system:
                if idx not in compat[existing_idx]:
                    compatible = False
                    break
            
            if compatible:
                current_system.append(idx)
        
        alpha = get_alpha(current_system)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = current_system[:]
            
        # If we hit the theoretical lower bound (likely 5 or 6), we can stop or continue to verify.
        if best_alpha <= 6:
            # We can break early if we are confident, but let's run a bit longer to be sure?
            # Or just break. 6 is likely the min.
            # Let's break to save time for safety.
            break
            
    if best_system is None:
        # Fallback: return empty or a trivial system
        return []
        
    # Convert indices back to triples
    return [list(all_triples[i]) for i in best_system]
