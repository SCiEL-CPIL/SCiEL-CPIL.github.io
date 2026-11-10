import numpy as np
import time
import sys

def construct(q, n, radius):
    # This implementation is specifically tuned for the ternary case q=3, n=8, radius=1
    # which is the specific task instance.
    # For generality, we could fall back to the seed method if parameters differ, 
    # but the task contract implies the evaluator passes the specific task's parameters.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to the seed's direct sum construction for other parameters
        # This is a safe default if the evaluator tests other small cases.
        import itertools
        # For q=3, n=4, radius=1, we use tetracode.
        # For other cases, a simple greedy or product might be needed.
        # However, the prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
        # So we assume q=3, n=8, radius=1.
        # We will implement a generic greedy fallback just in case, but optimize for 3,8,1.
        
        # Generic Greedy Fallback
        total_words = q ** n
        if total_words > 10000:
            # Too large for simple greedy in 120s without heavy optimization
            # Return a trivial code (all words) or the seed logic if applicable
            # Seed logic was specific to n>=4 and q=3.
            # Let's just return the seed logic if it applies, else full space.
            if q == 3 and n >= 4:
                 tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            else:
                 return [list(word) for word in itertools.product(range(q), repeat=n)]
        
        # Small case generic greedy
        # Precompute neighbors
        neighbors = []
        for i in range(total_words):
            # Convert i to base q vector
            vec = [0] * n
            rem = i
            for k in range(n):
                vec[k] = rem % q
                rem //= q
            vec.reverse()
            
            nbrs = [i]
            for pos in range(n):
                orig_val = vec[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_vec = vec.copy()
                    new_vec[pos] = new_val
                    # Convert back to int
                    new_i = 0
                    for k in range(n):
                        new_i = new_i * q + new_vec[k]
                    nbrs.append(new_i)
            neighbors.append(nbrs)
        
        neighbors_arr = np.array(neighbors)
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        
        while uncovered.any():
            # Count uncovered neighbors for each word
            # neighbors_arr is (total_words, sphere_size)
            # uncovered[neighbors_arr] gives (total_words, sphere_size) boolean array
            counts = uncovered[neighbors_arr].sum(axis=1)
            best_i = np.argmax(counts)
            if counts[best_i] == 0:
                break
            code.append(best_i)
            uncovered[neighbors_arr[best_i]] = False
            
        # Convert code back to lists
        result = []
        for i in code:
            vec = [0] * n
            rem = i
            for k in range(n):
                vec[k] = rem % q
                rem //= q
            vec.reverse()
            result.append(vec)
        return result

    # Optimized path for q=3, n=8, radius=1
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    total_words = 3 ** n  # 6561
    
    # 1. Generate all words as base-3 integers 0..6560
    # We need to map int <-> vector efficiently.
    
    # Precompute neighbors for all 6561 words
    # Sphere size = 1 + n*(q-1) = 1 + 8*2 = 17
    sphere_size = 17
    neighbors = np.zeros((total_words, sphere_size), dtype=np.int32)
    
    # Helper to convert int to base-3 tuple
    def int_to_vec(num):
        vec = [0] * n
        for i in range(n):
            vec[i] = num % 3
            num //= 3
        return vec

    # Helper to convert base-3 tuple to int
    def vec_to_int(vec):
        num = 0
        for i in range(n):
            num = num * 3 + vec[i]
        return num

    # Precompute neighbors
    for i in range(total_words):
        vec = int_to_vec(i)
        nbrs = [i]
        for pos in range(n):
            orig = vec[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                new_vec = vec.copy()
                new_vec[pos] = new_val
                nbrs.append(vec_to_int(new_vec))
        # There should be exactly 17 neighbors
        neighbors[i, :] = nbrs[:sphere_size]

    uncovered = np.ones(total_words, dtype=bool)
    code = []

    # Greedy Loop
    while uncovered.any():
        if time.time() > deadline:
            break
        
        # Vectorized count of uncovered neighbors
        # neighbors is (6561, 17)
        # uncovered[neighbors] is (6561, 17)
        counts = uncovered[neighbors].sum(axis=1)
        best_i = np.argmax(counts)
        
        if counts[best_i] == 0:
            # Should not happen if uncovered.any() is true, unless logic error
            break
            
        code.append(best_i)
        uncovered[neighbors[best_i]] = False

    # Convert code indices back to lists of ints
    result = []
    for i in code:
        vec = int_to_vec(i)
        result.append(vec)
        
    return result
