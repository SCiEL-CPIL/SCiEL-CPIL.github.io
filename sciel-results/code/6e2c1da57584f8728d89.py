import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for bit in combo:
            mask |= (1 << bit)
        all_sets.append(mask)
    
    n_total = len(all_sets)
    # Map mask to index for quick lookup? 
    # We just need to iterate.
    
    best_family = []
    best_size = 0
    
    for restart in range(100): # Max restarts, will break on time
        if time.time() > deadline:
            break
            
        # valid[i] is True if all_sets[i] is not in family and doesn't conflict
        # Actually, we just need to know if it's a candidate.
        # We maintain a list of valid indices.
        # Initially all are valid.
        valid_indices = list(range(n_total))
        random.shuffle(valid_indices)
        
        current_family = []
        # To speed up "find a valid candidate", we can just pop from valid_indices
        # But we need to remove indices that become invalid.
        # Using a set or list with removals is tricky.
        # Better: maintain a boolean array `is_valid` and a list of active candidates.
        
        is_valid = [True] * n_total
        # We'll use a list `candidates` that we iterate through.
        # When we invalidate some, we can just leave them in the list and skip them?
        # Or filter the list. Filtering is O(N).
        
        # Let's just iterate through the shuffled list of all sets.
        # If a set is still valid, add it.
        # To check validity, we need to know if it has been invalidated by a previous pair.
        # So we maintain `is_valid`.
        
        for s_idx in valid_indices:
            if time.time() > deadline:
                break
                
            if not is_valid[s_idx]:
                continue
                
            s = all_sets[s_idx]
            
            # Add s to family
            current_family.append(s)
            is_valid[s_idx] = False # It's in the family now, so not a candidate
            
            # Invalidate candidates that form sunflower with (s, a) for a in current_family[:-1]
            # Note: current_family[:-1] is the family before adding s.
            # We need to check pairs (s, a) where a is in the old family.
            for a in current_family[:-1]:
                # Compute forbidden candidates for pair (s, a)
                I = s & a
                # R = (s | a) ^ I  (elements in s or a but not in I)
                # C must contain I and be disjoint from R.
                # C is a 4-set.
                # Let's iterate through all candidates and check
                # This is O(N) per pair.
                # Optimization: Precompute for each mask its bits?
                # Or just loop.
                
                # To avoid O(N) loop per pair, we can precompute the list of candidates
                # that are disjoint from a given mask and contain a given mask?
                # That's complex. Let's try the O(N) loop first and see speed.
                
                # We can break early if we find we've invalidated many? No.
                
                # Let's optimize the inner loop by using bitmasks.
                # Condition: (c & I) == I and (c & R) == 0
                # c & I == I  <=>  (c | I) == c  <=>  ~c & I == 0
                # c & R == 0  <=>  ~c & R == R? No. c & R == 0.
                
                # We can precompute for each candidate c, its mask.
                # We already have all_sets.
                
                # Let's just loop.
                for c_idx in range(n_total):
                    if not is_valid[c_idx]:
                        continue
                    c = all_sets[c_idx]
                    if (c & I) == I and (c & ((s | a) ^ I)) == 0:
                        is_valid[c_idx] = False

        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family
            
    # Convert best_family bitmasks back to sorted lists of 1-based integers
    result = []
    for mask in best_family:
        elements = []
        for bit in range(16):
            if mask & (1 << bit):
                elements.append(bit + 1)
        result.append(sorted(elements))
        
    return result
