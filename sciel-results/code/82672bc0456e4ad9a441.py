import time
import random
import numpy as np

def construct(q, n, radius):
    # This task is specifically for q=3, n=7, radius=1.
    # We handle the general case by falling back to a simple construction if parameters differ,
    # but optimize for the specific case.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Direct sum of tetracode if n >= 4 and q=3
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            import itertools
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: all-zero codeword covers only 1 word, so we need many.
            # Just return a large set to be safe, though score will be bad.
            return [[0] * n]

    # Specific optimization for K_3(7, 1)
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    total_words = 3 ** n
    # Map word (tuple of ints) to index 0..total_words-1
    # We use base-3 representation
    
    def to_index(word):
        idx = 0
        for i in range(n):
            idx = idx * 3 + word[i]
        return idx

    def from_index(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word

    # Precompute spheres of radius 1 for all indices
    # Sphere of idx: idx itself and all indices differing in exactly 1 position
    spheres = []
    indices = list(range(total_words))
    words_list = [from_index(i) for i in indices]
    
    # Optimization: Precompute the list of neighbors for each index
    # For n=7, radius=1, sphere size is 1 + 7*2 = 15
    for idx in indices:
        word = words_list[idx]
        neigh = [idx]
        for pos in range(n):
            original_val = word[pos]
            for delta in [1, 2]:
                new_val = (original_val + delta) % 3
                # Calculate new index efficiently? 
                # Or just construct the word and to_index? 
                # to_index is fast enough for 2187 * 14 ops.
                new_word = word[:]
                new_word[pos] = new_val
                neigh.append(to_index(new_word))
        spheres.append(neigh)

    # 1. Greedy Construction
    uncovered = np.ones(total_words, dtype=bool)
    code_indices = []
    
    # Precompute sphere arrays for fast numpy operations?
    # spheres is a list of lists. We can convert to a numpy array of shape (total_words, sphere_size) 
    # but sphere sizes are uniform (15), so yes.
    sphere_size = len(spheres[0])
    spheres_np = np.array(spheres, dtype=np.int32)
    
    while np.any(uncovered):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Find the index that covers the most uncovered words
        # For each candidate center, count how many of its neighbors are uncovered
        # uncovered is a boolean array.
        # We want to maximize sum(uncovered[spheres_np[i, :]] for i in range(total_words))
        
        # Efficient way:
        # counts[i] = sum(uncovered[spheres_np[i]])
        # This can be done via matrix multiplication if we view uncovered as a vector?
        # Or just loop. 2187 * 15 is small.
        
        best_idx = -1
        max_count = -1
        
        # Optimization: Only consider centers that have at least one uncovered neighbor?
        # Actually, if a center has 0 uncovered neighbors, it's useless.
        # We can just iterate all.
        
        for i in range(total_words):
            # Count uncovered in sphere
            # Using numpy slicing is fast
            c = np.sum(uncovered[spheres_np[i]])
            if c > max_count:
                max_count = c
                best_idx = i
                if c == sphere_size: # Perfect coverage of sphere
                    # Can't get better than sphere_size
                    pass 
        
        if best_idx == -1 or max_count == 0:
            # Should not happen if uncovered is not empty
            break
            
        code_indices.append(best_idx)
        uncovered[spheres_np[best_idx]] = False

    # 2. Deletion Heuristic (Local Search)
    # Convert code_indices to a set for fast lookup if needed, but we work with list
    # We need to maintain cover counts.
    
    def try_deletion_pass(code_list):
        """
        Attempts to remove redundant codewords from code_list.
        Returns a new list with redundant codewords removed.
        """
        if not code_list:
            return code_list
            
        # Build cover counts
        cover_count = np.zeros(total_words, dtype=np.int32)
        for idx in code_list:
            cover_count[spheres_np[idx]] += 1
            
        # Shuffle to avoid bias
        random.shuffle(code_list)
        
        removed = 0
        new_code = []
        for idx in code_list:
            # Tentatively remove
            cover_count[spheres_np[idx]] -= 1
            if np.min(cover_count) < 1:
                # Restore
                cover_count[spheres_np[idx]] += 1
                new_code.append(idx)
            else:
                # Keep removed
                removed += 1
                
        return new_code, removed

    # Run deletion passes
    current_code = code_indices
    while time.time() - start_time < time_limit:
        new_code, removed = try_deletion_pass(current_code)
        if removed == 0:
            # Try a few more random shuffles to escape local minima
            for _ in range(5):
                if time.time() - start_time > time_limit:
                    break
                new_code, removed = try_deletion_pass(current_code)
                if removed > 0:
                    current_code = new_code
                    break
            if removed == 0:
                break
        else:
            current_code = new_code

    # Convert indices back to words
    return [words_list[i] for i in current_code]
