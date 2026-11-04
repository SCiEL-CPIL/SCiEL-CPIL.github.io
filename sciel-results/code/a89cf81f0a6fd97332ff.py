import numpy as np
import itertools
import time

D = 11

def generate_vectors(norm_sq):
    """Generate all integer vectors with squared norm equal to norm_sq."""
    vectors = []
    # We need sum of squares = norm_sq
    # Since D=11, max value is sqrt(norm_sq)
    max_val = int(np.sqrt(norm_sq)) + 1
    
    # Recursive generation or itertools.product
    # Product is too slow for large ranges, but norm_sq is small.
    # Range -max_val to max_val.
    # 2*max_val+1 choices per dimension.
    # If norm_sq=2, max_val=2. 5^11 = 48M. Too slow.
    
    # Better: generate partitions of norm_sq into squares.
    # Then permute and assign signs.
    
    def gen_squares(n, max_n):
        if n == 0:
            yield []
            return
        for i in range(1, max_n + 1):
            sq = i * i
            if sq > n:
                continue
            for rest in gen_squares(n - sq, i):
                yield [i] + rest
                
    # Get all multisets of integers whose squares sum to norm_sq
    # Then generate unique permutations and sign assignments
    
    # Actually, simpler:
    # Iterate over all possible non-zero coordinate values
    # We can just iterate over all integer vectors in a box?
    # No, 5^11 is too big.
    
    # Let's use the partition approach.
    parts = list(gen_squares(norm_sq, int(np.sqrt(norm_sq))))
    
    unique_vectors = set()
    for p in parts:
        # p is a sorted list of non-zero absolute values
        # e.g., [1, 1] for norm_sq=2
        # We need to place these in D dimensions.
        # The remaining dimensions are 0.
        
        # Generate unique permutations of p + [0]*(D-len(p))
        # Then assign signs to non-zero elements.
        
        base = p + [0] * (D - len(p))
        
        # Get unique permutations of base
        seen_perms = set()
        for perm in itertools.permutations(base):
            if perm in seen_perms:
                continue
            seen_perms.add(perm)
            
            # Assign signs to non-zero elements
            non_zero_indices = [i for i, x in enumerate(perm) if x != 0]
            num_non_zero = len(non_zero_indices)
            
            for signs in itertools.product([1, -1], repeat=num_non_zero):
                v = list(perm)
                for k, idx in enumerate(non_zero_indices):
                    v[idx] = v[idx] * signs[k]
                unique_vectors.add(tuple(v))
                
    return list(unique_vectors)

def max_independent_set_heuristic(vectors, norm_sq):
    """Find a large subset of vectors with pairwise dot product <= norm_sq // 2."""
    n = len(vectors)
    if n == 0:
        return []
    
    # Precompute dot products
    # This is O(n^2 * D). n can be ~1000. 1000^2 * 11 = 11M. OK.
    V = np.array(vectors)
    dots = V @ V.T
    
    threshold = norm_sq // 2
    # Conflict if dot > threshold
    # We want max independent set in graph where edge exists if dot > threshold
    
    # Greedy initialization
    # Sort by degree (number of conflicts)
    degrees = np.sum(dots > threshold, axis=1)
    
    best_set = []
    best_size = 0
    
    start_time = time.time()
    deadline = start_time + 30 # 30s per norm
    
    # Multiple restarts with random ordering
    for _ in range(50):
        if time.time() > deadline:
            break
            
        # Randomize order, but prioritize low degree
        # Actually, just random shuffle
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        selected = []
        remaining = set(indices)
        
        for idx in indices:
            if idx not in remaining:
                continue
            
            # Check if idx conflicts with any in selected
            # We can maintain a set of conflicts
            # Simpler: check against selected
            ok = True
            for s in selected:
                if dots[idx, s] > threshold:
                    ok = False
                    break
            
            if ok:
                selected.append(idx)
                # Remove neighbors from remaining?
                # This is O(n^2) per step if done naively.
                # n=1000, steps=1000. 1M ops. OK.
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
        # Local search: try to swap
        # Not implemented for now, greedy might be enough to beat 220?
        
    return [vectors[i] for i in best_set]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_vectors = None
    best_size = 0
    
    # Try different norms
    # norm_sq = 2 is the seed (220)
    # norm_sq = 5
    # norm_sq = 10
    
    for norm_sq in [2, 5, 10, 13, 17, 25]:
        if time.time() > deadline - 10:
            break
            
        vectors = generate_vectors(norm_sq)
        if not vectors:
            continue
            
        # If the total number of vectors is less than best_size, skip
        # (We can't do better than the total count)
        if len(vectors) <= best_size:
            continue
            
        result = max_independent_set_heuristic(vectors, norm_sq)
        
        if len(result) > best_size:
            best_size = len(result)
            best_vectors = result
            
    if best_vectors is None:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        best_vectors = vectors
        best_size = len(vectors)
        
    return np.array(best_vectors, dtype=np.int64)
