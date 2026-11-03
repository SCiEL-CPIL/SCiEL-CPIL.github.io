import numpy as np
import time
import itertools

def construct(q, n, radius):
    """
    Finds a covering code of length n over alphabet size q with covering radius r.
    Uses a greedy set cover approach.
    """
    start_time = time.time()
    deadline = 110.0 # Safety margin for 120s limit
    
    # Only handle the specific case for efficiency and correctness in this context
    # The general greedy works, but let's ensure it handles the specific parameters
    if q != 3 or n != 6 or radius != 1:
        # Fallback to a simple direct product if parameters are different (though task specifies 3,6,1)
        # This branch is unlikely to be hit based on the task description
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
             # Just return all words if n is small
             return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimized path for q=3, n=6, r=1
    total_words = 3**6
    # Map integer to list of digits for final output
    def int_to_list(x):
        return [int(d) for d in np.base_repr(x, 3).zfill(6)]
    
    # Precompute neighbors: for each word, list of words within distance 1
    # A word x in base 3 can be represented as an integer.
    # Neighbors are x itself, and x +/- 3^i (with carry handling? No, just digit change)
    # Easier: generate all words as lists, compute neighbors, map to ints.
    
    # Generate all words as tuples
    all_words = list(itertools.product(range(3), repeat=6))
    # Map tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor indices for each word
    neighbors = []
    for w in all_words:
        w_list = list(w)
        nbs = [w]
        for i in range(6):
            orig = w_list[i]
            for val in range(3):
                if val == orig: continue
                w_list[i] = val
                nbs.append(tuple(w_list))
            w_list[i] = orig
        neighbors.append([word_to_idx[nb] for nb in nbs])
    
    # Convert to numpy array for fast indexing
    neighbors_arr = np.array(neighbors, dtype=np.int32)
    
    # Coverage status: 0 for uncovered, 1 for covered
    covered = np.zeros(total_words, dtype=np.uint8)
    
    code = []
    
    while True:
        if time.time() > deadline:
            break
            
        # Count how many uncovered neighbors each candidate word has
        # candidates: 0..728
        # For each candidate c, sum(covered[neighbors[c]] == 0)
        
        # Vectorized approach:
        # Get all neighbors for all candidates: shape (729, 19)
        # Check if they are uncovered: shape (729, 19) boolean
        # Sum along axis 1: shape (729,)
        
        # This creates a large intermediate array (729*19 ~ 14k elements), which is fine.
        # covered[neighbors_arr] -> shape (729, 19)
        # ~0 -> boolean
        # sum(axis=1)
        
        uncovered_counts = np.sum(covered[neighbors_arr] == 0, axis=1)
        
        # Find the best candidate
        best_idx = np.argmax(uncovered_counts)
        
        # If no uncovered words remain (or best count is 0), we are done
        if uncovered_counts[best_idx] == 0:
            break
            
        # Add best candidate to code
        code.append(best_idx)
        
        # Mark its neighbors as covered
        covered[neighbors_arr[best_idx]] = 1
        
        # Optimization: If we've covered everything, break early
        if np.sum(covered) == total_words:
            break
            
    # Convert integer indices back to lists of digits
    result = [int_to_list(i) for i in code]
    
    # Ensure distinctness (greedy ensures distinct candidates if we don't re-pick, 
    # but argmax might pick a covered word if counts are equal? 
    # Actually, if a word is already in the code, its neighbors are covered.
    # If we pick a word already in the code, its uncovered count is 0.
    # Since we break if max count is 0, and we only add if count > 0, 
    # and we mark neighbors as covered, a previously added word will have count 0 
    # because all its neighbors are covered. So no duplicates.)
    
    return result

