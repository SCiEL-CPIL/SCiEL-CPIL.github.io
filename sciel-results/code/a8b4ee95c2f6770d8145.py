import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Precompute modular inverses of 1..10 modulo 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
    
    # For each residue, store the minimum-size subset that sums to it
    # min_size[r] = smallest k such that some k-subset sums to r mod 101
    # min_mask[r] = bitmask of that subset
    min_size = [11] * 101
    min_mask = [0] * 101
    
    # Enumerate all non-empty subsets by increasing size
    # For each size k, find the first subset (in lex order) that covers each residue
    for k in range(1, 11):
        if time.time() > deadline:
            break
        
        for comb in combinations(range(1, 11), k):
            s = sum(inv[x] for x in comb) % 101
            if min_size[s] > k:
                min_size[s] = k
                mask = 0
                for x in comb:
                    mask |= (1 << (x - 1))
                min_mask[s] = mask
        
        # Early termination: if all residues are covered, we can stop
        # since we're going in increasing order of k
        if all(min_size[r] <= k for r in range(101)):
            break
    
    # Determine the maximum row length (this is our score)
    max_len = max(min_size)
    
    # Build result
    result = []
    for r in range(101):
        mask = min_mask[r]
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        # Fallback: if somehow no subset was found (shouldn't happen), use full set
        if not terms:
            terms = list(range(1, 11))
        result.append(terms)
    
    return result
