"""Rewritten program using bitmask exhaustive search for maximum efficiency."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_min_v = int(P['minimum_vertices'])

# Precompute adjacency bitmasks
_adj_masks = [0] * _N
for a, b in P['edges']:
    _adj_masks[a] |= (1 << b)
    _adj_masks[b] |= (1 << a)

# Precompute popcount for all 17-bit masks? No, just use bin(x).count('1') or a lookup table for bytes.
# Since N=17, max mask is 2^17.
# We can use a simple helper.
def _popcount(x):
    return bin(x).count('1')

# Or use int.bit_count() if Python 3.10+. Let's assume standard compatibility.
# bin(x).count('1') is safe.

_deadline = time.time() + 80.0  # Safety margin

def _score_mask(mask):
    """Calculate number of distinct degrees for the subset represented by mask."""
    if mask == 0:
        return 0
    degs = set()
    m = mask
    while m:
        # Get lowest set bit
        lsb = m & -m
        v = lsb.bit_length() - 1
        # Calculate degree of v in this subset
        d = _popcount(_adj_masks[v] & mask)
        degs.add(d)
        m ^= lsb
    return len(degs)

def construct():
    global _deadline
    best_score = 0
    best_mask = 0
    
    # Iterate through all subsets
    # We can iterate from 0 to 2^N - 1
    # Filter by popcount >= _min_v
    # To speed up, we can iterate by popcount or just check.
    # 2^17 is 131k. Checking popcount for each is fast.
    
    for mask in range(1 << _N):
        if time.time() > _deadline:
            break
            
        # Check size constraint
        if _popcount(mask) < _min_v:
            continue
            
        score = _score_mask(mask)
        if score > best_score:
            best_score = score
            best_mask = mask
            
    # Convert best_mask to list of vertices
    best_subset = []
    for i in range(_N):
        if best_mask & (1 << i):
            best_subset.append(int(i))
            
    # Fallback if no valid subset found (should not happen)
    if not best_subset:
        best_subset = list(range(_min_v))
        
    return best_subset
# EVOLVE-BLOCK-END
