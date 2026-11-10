import time
import random
import numpy as np

# Precompute neighbors for q=3, n=8
Q = 3
N = 8
SIZE = Q ** N

# Convert index to tuple of digits
def idx_to_tuple(idx):
    t = []
    for _ in range(N):
        t.append(idx % Q)
        idx //= Q
    return t

# Convert tuple of digits to index
def tuple_to_idx(t):
    idx = 0
    for i in range(N):
        idx = idx * Q + t[i]
    return idx

# Precompute neighbors
# neighbors[i] is a list of indices j such that Hamming distance(i, j) <= 1
neighbors = []
for i in range(SIZE):
    t = idx_to_tuple(i)
    nb = [i]
    for pos in range(N):
        orig = t[pos]
        for val in range(Q):
            if val == orig:
                continue
            t[pos] = val
            nb.append(tuple_to_idx(t))
        t[pos] = orig
    neighbors.append(nb)

def greedy_construct(seed):
    rng = random.Random(seed)
    covered = np.zeros(SIZE, dtype=bool)
    C = []
    
    # To speed up finding uncovered, we can keep a list of uncovered indices
    # But removing from a list is slow.
    # Since SIZE is small (6561), scanning is O(6561).
    # Number of steps is ~400. Total ops ~2.6M. Acceptable.
    
    while not covered.all():
        # Find an uncovered index
        # Using numpy argwhere might be slow due to overhead.
        # Let's find the first False.
        # np.flatnonzero(~covered) returns indices.
        uncovered_indices = np.flatnonzero(~covered)
        if len(uncovered_indices) == 0:
            break
        # Pick a random one from the uncovered
        # rng.choice(uncovered_indices) is O(len(uncovered))
        # Or just pick the first one for determinism? 
        # Random choice might lead to better packing.
        c = int(rng.choice(uncovered_indices))
        C.append(c)
        # Mark neighbors as covered
        for w in neighbors[c]:
            covered[w] = True
            
    return C

def remove_redundant(C):
    # C is a list of indices
    # cover_count[w] = number of codewords in C that cover w
    cover_count = np.zeros(SIZE, dtype=int)
    for c in C:
        for w in neighbors[c]:
            cover_count[w] += 1
            
    changed = True
    while changed:
        changed = False
        new_C = []
        for c in C:
            # Check if c is redundant
            # c is redundant if for all w in neighbors[c], cover_count[w] > 1
            # Note: cover_count includes c itself.
            redundant = True
            for w in neighbors[c]:
                if cover_count[w] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove c
                for w in neighbors[c]:
                    cover_count[w] -= 1
                changed = True
            else:
                new_C.append(c)
        C = new_C
    return C

def construct(q, n, radius):
    # Verify parameters match precomputation
    # The task says q, n, radius are passed.
    # We precomputed for q=3, n=8, radius=1.
    # If parameters differ, we need to handle it.
    # But the task is specific to this instance.
    # "The evaluator passes this task's alphabet size q, length n and covering radius."
    # It implies q=3, n=8, radius=1.
    # If they are different, the precomputed neighbors are invalid.
    # We should check and potentially recompute or raise an error/fallback.
    # Given the context "Football pool problem for 8 matches", it's fixed.
    # But to be safe, if q != 3 or n != 8 or radius != 1, we can't use the precomputed data.
    # However, the prompt says "Keep the function names...".
    # I will assume q=3, n=8, radius=1.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to simple direct sum if params don't match (unlikely)
        # Just return a valid code for general case?
        # For now, assume it's the specific case.
        pass

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_code = None
    best_size = float('inf')
    
    # Run multiple iterations with different seeds
    # Each iteration is very fast.
    for seed in range(100000):
        if time.time() > deadline:
            break
            
        C = greedy_construct(seed)
        C = remove_redundant(C)
        
        if len(C) < best_size:
            best_size = len(C)
            best_code = C
            # If we reach the theoretical lower bound or a known good value, we can stop early?
            # Best known is 486. Lower bound 403.
            # If we hit 486, it's good. If we hit < 486, it's better.
            # We don't know the absolute minimum, so keep searching.
            
        # Optional: If best_size is very small, we might spend more time polishing?
        # The remove_redundant is already a polish.
        
    if best_code is None:
        # Should not happen, but fallback to direct sum
        # Direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        best_code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                best_code.append(tuple_to_idx(h + list(tail)))
                
    # Convert indices to lists of integers
    result = [idx_to_tuple(idx) for idx in best_code]
    return result

