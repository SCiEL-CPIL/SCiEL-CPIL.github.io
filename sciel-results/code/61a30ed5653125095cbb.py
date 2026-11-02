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
    total_points = q ** n
    # Map word (list of ints) to index (int)
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
    # neighbors[i] is a list of indices of words within distance 1 from word i
    neighbors = np.zeros((total_points, 17), dtype=np.int32)
    
    # Generate all words and their indices
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
                # Create new word
                w[pos] = new_val
                new_idx = word_to_idx(w)
                neighbors[i, count] = new_idx
                count += 1
            w[pos] = orig_val
        
    # Greedy Set Cover
    covered = np.zeros(total_points, dtype=bool)
    chosen_indices = []
    available = np.ones(total_points, dtype=bool)
    
    start_time = time.time()
    time_limit = 110.0 # Keep 10s buffer
    
    while True:
        if covered.all():
            break
        if time.time() - start_time > time_limit:
            break
            
        # For each available word, count how many of its neighbors are NOT covered
        # We want to maximize this count
        # Mask available words
        # neighbors[available] shape: (num_avail, 17)
        # covered[neighbors[available]] shape: (num_avail, 17)
        # ~covered[...] gives True for uncovered
        # sum along axis 1 gives count of newly covered points
        
        avail_indices = np.where(available)[0]
        if len(avail_indices) == 0:
            break
            
        # Vectorized calculation
        # This might be memory heavy if avail_indices is large initially (6561 * 17 is small)
        neigh_avail = neighbors[avail_indices]
        # Check coverage
        is_covered = covered[neigh_avail]
        newly_covered_count = (~is_covered).sum(axis=1)
        
        # Find the word with max newly covered points
        # Tie-breaker: random or first found
        max_count = np.max(newly_covered_count)
        if max_count == 0:
             # Should not happen if not all covered, but safety
             break
             
        # Pick the first one with max count
        best_local_idx = np.argmax(newly_covered_count)
        best_global_idx = avail_indices[best_local_idx]
        
        chosen_indices.append(best_global_idx)
        # Update covered
        covered[neighbors[best_global_idx]] = True
        available[best_global_idx] = False
        
    # Convert indices to words
    result = [all_words[i] for i in chosen_indices]
    
    # Optional: Local search to reduce size further?
    # Given time constraints and that greedy is quite good, 
    # and 486 is the target, greedy might land around 500-550.
    # A simple swap heuristic could help.
    
    # Let's try a simple improvement loop if time permits
    # Try to remove a word if its coverage is fully covered by others?
    # Or try to swap a word with one that allows removal of another?
    
    # For now, return the greedy result.
    # To beat 729, greedy is sufficient.
    # To beat 500, we need the specific 486 structure.
    
    # Let's check if we can force the 486 structure.
    # The 486 code is known.
    # If we can't guarantee 486, greedy is the safest bet for a significant improvement over 729.
    
    return result
