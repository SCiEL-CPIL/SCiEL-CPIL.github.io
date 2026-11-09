"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START
# Precomputation
ALL_TRIPLES = list(itertools.combinations(range(16), 3))
TRIPLE_TO_IDX = {t: i for i, t in enumerate(ALL_TRIPLES)}
N_TRIPLES = len(ALL_TRIPLES) # 560

# Generate all K(2,2,2) structures
# A K(2,2,2) is defined by 3 disjoint pairs.
# We iterate over all sets of 6 distinct points.
# For each set of 6, there are 15 ways to partition into 3 pairs.
K222_STRUCTURES = []

# Helper to get the 8 transversals given 3 pairs (a,a'), (b,b'), (c,c')
def get_transversals(pairs):
    # pairs = [(a, a'), (b, b'), (c, c')]
    a, ap = pairs[0]
    b, bp = pairs[1]
    c, cp = pairs[2]
    
    # The 8 transversals are combinations of picking one from each pair
    # (a,b,c), (a,b,cp), (a,bp,c), (a,bp,cp), (ap,b,c), (ap,b,cp), (ap,bp,c), (ap,bp,cp)
    transversals = [
        tuple(sorted((a, b, c))),
        tuple(sorted((a, b, cp))),
        tuple(sorted((a, bp, c))),
        tuple(sorted((a, bp, cp))),
        tuple(sorted((ap, b, c))),
        tuple(sorted((ap, b, cp))),
        tuple(sorted((ap, bp, c))),
        tuple(sorted((ap, bp, cp)))
    ]
    return transversals

# Precompute pairings of 6 items into 3 pairs
# There are 15 such pairings.
# We can hardcode them or generate them.
# Generating:
def generate_pairings(points):
    # points is a list of 6 elements
    # Fix point 0. Pair it with 1, 2, 3, 4, 5.
    # If paired with 1, remaining 4 have 3 pairings.
    # Total 5 * 3 = 15.
    p = points
    pairings = []
    for i in range(1, 6):
        # Pair p[0] with p[i]
        pair1 = (p[0], p[i])
        remaining = [p[j] for j in range(1, 6) if j != i]
        # Generate pairings for remaining 4
        # Fix remaining[0]. Pair with remaining[1], remaining[2], remaining[3].
        for j in range(1, 4):
            pair2 = (remaining[0], remaining[j])
            rem2 = [remaining[k] for k in range(1, 4) if k != j]
            pair3 = (rem2[0], rem2[1])
            pairings.append([pair1, pair2, pair3])
    return pairings

# Build K222_STRUCTURES
for points in itertools.combinations(range(16), 6):
    points_list = list(points)
    for pairs in generate_pairings(points_list):
        transversals = get_transversals(pairs)
        # Store as tuple of indices
        idxs = tuple(TRIPLE_TO_IDX[t] for t in transversals)
        K222_STRUCTURES.append(idxs)

# Build conflict masks
# conflict_masks[i] is a list of masks. Each mask represents the other 7 triples in a K(2,2,2) containing i.
conflict_masks = [[] for _ in range(N_TRIPLES)]

for k222 in K222_STRUCTURES:
    # k222 is a tuple of 8 indices
    for i in range(8):
        # Create mask of all 8
        # We want mask of the other 7
        mask = 0
        for j in range(8):
            if j != i:
                mask |= (1 << k222[j])
        conflict_masks[k222[i]].append(mask)

# Build NumPy representation for fast vectorized checks
NUM_WORDS = 9 # 560 bits / 64 bits per word = 8.75 -> 9
conflict_masks_np = []
for masks in conflict_masks:
    if not masks:
        conflict_masks_np.append(np.empty((0, NUM_WORDS), dtype=np.uint64))
    else:
        arr = np.empty((len(masks), NUM_WORDS), dtype=np.uint64)
        for i, m in enumerate(masks):
            for k in range(NUM_WORDS):
                arr[i, k] = (m >> (64 * k)) & 0xFFFFFFFFFFFFFFFF
        conflict_masks_np.append(arr)

def _check_addition(triple_idx, S_int):
    """
    Check if adding triple_idx to S_int creates a K(2,2,2).
    Returns True if valid, False if conflict.
    Uses NumPy for vectorized conflict checking.
    """
    masks = conflict_masks_np[triple_idx]
    if masks.size == 0:
        return True
    
    # Convert S_int to numpy array of uint64 words
    s_words = np.empty(NUM_WORDS, dtype=np.uint64)
    val = S_int
    for k in range(NUM_WORDS):
        s_words[k] = val & 0xFFFFFFFFFFFFFFFF
        val >>= 64
        
    # Check if any mask is a subset of S_int
    # masks shape: (K, 9), s_words shape: (9,)
    # (masks & s_words[None, :]) == masks checks bitwise subset for each word
    # np.all(..., axis=1) checks if ALL words in the mask are covered
    # np.any(...) checks if ANY mask is fully covered
    is_submask = np.all((masks & s_words[None, :]) == masks, axis=1)
    return not np.any(is_submask)

def _greedy_initial():
    """Generate an initial valid triple set using greedy."""
    indices = list(range(N_TRIPLES))
    random.shuffle(indices)
    
    S_int = 0
    count = 0
    
    for idx in indices:
        if _check_addition(idx, S_int):
            S_int |= (1 << idx)
            count += 1
            
    return S_int, count

def _local_search(S_int, count, time_limit=10.0):
    """Improve the triple set using local search."""
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try to add any missing triple
        # We need to iterate over all triples not in S_int
        # To avoid generating the list of missing triples every time (expensive),
        # we can just iterate over all indices and check if bit is 0.
        # 560 checks is cheap.
        
        # Randomize order to avoid bias
        indices = list(range(N_TRIPLES))
        random.shuffle(indices)
        
        for idx in indices:
            if not (S_int & (1 << idx)): # Not in set
                if _check_addition(idx, S_int):
                    S_int |= (1 << idx)
                    count += 1
                    improved = True
                    break
        
        if not improved:
            # 2. Try a 1-remove, 2-add move (net +1)
            # Pick a random triple to remove
            # We need to pick from present triples.
            # Generate list of present indices.
            present_indices = [i for i in range(N_TRIPLES) if (S_int & (1 << i))]
            if not present_indices:
                break
            random.shuffle(present_indices)
            
            for idx_rem in present_indices[:50]: # Try more removals due to faster checks
                # Remove
                S_int_new = S_int & ~(1 << idx_rem)
                count_new = count - 1
                
                # Try to add as many as possible
                added = 0
                # Shuffle all indices to try adding
                add_candidates = list(range(N_TRIPLES))
                random.shuffle(add_candidates)
                
                for idx_add in add_candidates:
                    if idx_add == idx_rem:
                        continue
                    if not (S_int_new & (1 << idx_add)):
                        if _check_addition(idx_add, S_int_new):
                            S_int_new |= (1 << idx_add)
                            count_new += 1
                            added += 1
                            
                if count_new > count:
                    S_int = S_int_new
                    count = count_new
                    improved = True
                    break
            if not improved:
                break
                
    return S_int, count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_S_int = 0
    best_count = 0
    
    while time.time() < deadline:
        S_int, count = _greedy_initial()
        S_int, count = _local_search(S_int, count, time_limit=5.0)
        
        if count > best_count:
            best_count = count
            best_S_int = S_int
            
    # Convert back to list of triples
    result = []
    for i in range(N_TRIPLES):
        if best_S_int & (1 << i):
            result.append(list(ALL_TRIPLES[i]))
            
    return result
# EVOLVE-BLOCK-END
