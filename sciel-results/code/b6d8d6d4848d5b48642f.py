import numpy as np
import itertools
import time

def construct(q, n, radius):
    # This task is specifically for q=3, n=8, radius=1
    # But the interface is general.
    # For other parameters, we fall back to a simple greedy or the seed logic.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to the seed program's logic for other cases
        # The seed program used tetracode for n>=4
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Small n, use full space or simple greedy
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific optimization for K_3(8,1)
    total_points = q ** n  # 6561
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for all 6561 words
    # Each word has 1 + n*(q-1) = 1 + 8*2 = 17 neighbors (including itself)
    neighbors = np.zeros((total_points, 17), dtype=np.int32)
    
    all_words = [idx_to_word(i) for i in range(total_points)]
    
    for i in range(total_points):
        w = all_words[i]
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                new_idx = word_to_idx(w)
                neighbors[i, count] = new_idx
                count += 1
            w[pos] = orig_val
    
    # Build a reverse mapping: for each point, which codeword indices cover it
    # This is expensive, so we'll use a different approach.
    
    # Greedy Set Cover with bitmask acceleration
    # Use numpy arrays for fast operations
    covered = np.zeros(total_points, dtype=bool)
    available = np.ones(total_points, dtype=bool)
    chosen_indices = []
    
    start_time = time.time()
    time_limit = 105.0  # Keep 15s buffer for local search and safety
    
    while True:
        if covered.all():
            break
        if time.time() - start_time > time_limit:
            break
        
        # For each available word, count how many of its neighbors are NOT covered
        avail_indices = np.where(available)[0]
        if len(avail_indices) == 0:
            break
        
        # Vectorized calculation: for each available word, count newly covered points
        neigh_avail = neighbors[avail_indices]  # shape: (num_avail, 17)
        is_covered = covered[neigh_avail]  # shape: (num_avail, 17)
        newly_covered_count = (~is_covered).sum(axis=1)
        
        max_count = np.max(newly_covered_count)
        if max_count == 0:
            break
        
        # Pick the word with max newly covered points
        best_local_idx = np.argmax(newly_covered_count)
        best_global_idx = avail_indices[best_local_idx]
        
        chosen_indices.append(best_global_idx)
        covered[neighbors[best_global_idx]] = True
        available[best_global_idx] = False
    
    # Local search: try to remove redundant codewords
    # A codeword is redundant if all points it covers are also covered by other chosen codewords
    if time.time() - start_time < time_limit - 5.0:
        chosen_set = set(chosen_indices)
        # For each chosen codeword, check if removing it still covers all points
        # We need to track coverage counts
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        
        removable = []
        for i, ci in enumerate(chosen_indices):
            # Check if all points covered by ci are covered by at least one other codeword
            # i.e., cover_count[neighbors[ci]] >= 2 for all neighbors
            if np.all(cover_count[neighbors[ci]] >= 2):
                removable.append(i)
        
        # Try removing in order of least unique coverage
        for i in removable:
            if time.time() - start_time > time_limit - 5.0:
                break
            ci = chosen_indices[i]
            # Temporarily remove and check
            cover_count[neighbors[ci]] -= 1
            if np.all(cover_count[neighbors[ci]] >= 1):
                # Safe to remove
                cover_count[neighbors[ci]] += 1
                # Actually remove
                cover_count[neighbors[ci]] -= 1
                chosen_indices.pop(i)
                chosen_set.discard(ci)
            else:
                # Not safe, restore
                cover_count[neighbors[ci]] += 1
    
    # Second pass: try to swap codewords to reduce size
    if time.time() - start_time < time_limit - 3.0:
        # Try swapping: for each chosen codeword, see if replacing it with another
        # allows removal of another codeword
        for pass_num in range(3):
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            cover_count = np.zeros(total_points, dtype=np.int32)
            for ci in chosen_indices:
                cover_count[neighbors[ci]] += 1
            
            for i in range(len(chosen_indices)):
                if time.time() - start_time > time_limit - 3.0:
                    break
                ci = chosen_indices[i]
                # Try removing ci temporarily
                cover_count[neighbors[ci]] -= 1
                # Find uncovered points
                uncovered = np.where(cover_count == 0)[0]
                if len(uncovered) == 0:
                    # Can remove ci entirely
                    cover_count[neighbors[ci]] += 1
                    chosen_indices.pop(i)
                    improved = True
                    # Restart this pass
                    cover_count = np.zeros(total_points, dtype=np.int32)
                    for ci2 in chosen_indices:
                        cover_count[neighbors[ci2]] += 1
                    break
                else:
                    # Try to find a replacement that covers all uncovered points
                    # and allows removal of another codeword
                    # This is expensive, so skip for now
                    cover_count[neighbors[ci]] += 1
            
            if not improved:
                break
    
    # Convert indices to words
    result = [all_words[i] for i in chosen_indices]
    
    return result
