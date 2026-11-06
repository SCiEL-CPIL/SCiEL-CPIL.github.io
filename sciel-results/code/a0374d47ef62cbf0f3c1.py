"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def _paley_orientation(p):
    """Return orientation list for Paley tournament on p vertices."""
    squares = {x * x % p for x in range(1, p)}
    n = p
    orient = []
    for i in range(n):
        for j in range(i + 1, n):
            # i -> j iff (j - i) % p is a nonzero square mod p
            if (j - i) % p in squares:
                orient.append(0)
            else:
                orient.append(1)
    return n, orient

def _check_four_dominator_vectorized(n, orient):
    """
    Vectorized check for the Schütte condition (k=4).
    Returns True if every set of 4 vertices has a common dominator.
    """
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orient[idx]
            idx += 1
            if bit == 0:
                adj[i, j] = True
            else:
                adj[j, i] = True

    # Precompute indices for combinations of 3 vertices for each vertex
    # We need to check if there exists a 4-set {a,b,c,d} such that no vertex dominates the other 3.
    # This is equivalent to: For every a, and every pair (b,c) with b,c > a, 
    # and every d > c, the set {a,b,c,d} must be dominated by a, b, c, or d.
    
    # Let's iterate over 'a' and 'b' and vectorize the rest.
    # For a fixed a, b (a < b), we consider all pairs (c, d) with b < c < d.
    # The set is {a, b, c, d}.
    # Dominated by a: adj[a,b] & adj[a,c] & adj[a,d]
    # Dominated by b: adj[b,a] & adj[b,c] & adj[b,d]
    # Dominated by c: adj[c,a] & adj[c,b] & adj[c,d]
    # Dominated by d: adj[d,a] & adj[d,b] & adj[d,c]
    
    # If none of these are True, it's a violation.
    
    # To avoid O(n^4) memory, we loop over a and b, and vectorize c and d.
    # Number of (c,d) pairs for fixed a,b is C(n-b-1, 2).
    # Total iterations is sum_{a=0}^{n-1} sum_{b=a+1}^{n-2} C(n-b-1, 2) = C(n, 4).
    # But the inner loop is vectorized.
    
    for a in range(n - 3):
        for b in range(a + 1, n - 2):
            # Get indices for c and d
            # c ranges from b+1 to n-2
            # d ranges from c+1 to n-1
            # We can create arrays of all valid (c,d) pairs.
            
            # To be efficient, we can just generate the indices.
            # For large n, this might be slow if done naively.
            # Let's try a simpler vectorization:
            # For fixed a, b:
            # Let C be the list of vertices > b.
            # We need to check all pairs (c, d) from C with c < d.
            
            C = np.arange(b + 1, n)
            if len(C) < 2:
                continue
            
            # Generate all pairs (c, d) with c < d from C
            # We can use meshgrid or just loop if C is small?
            # For n=67, max len(C) is 66. Pairs ~ 2000.
            # This is small enough to vectorize fully.
            
            # Create indices for c and d
            # c_idx, d_idx are arrays of indices into C
            # We want all i < j in range(len(C))
            i_idx, j_idx = np.triu_indices(len(C), k=1)
            
            c_vals = C[i_idx]
            d_vals = C[j_idx]
            
            # Check conditions
            # 1. a dominates: adj[a,b] & adj[a,c_vals] & adj[a,d_vals]
            dom_a = adj[a, b] & adj[a, c_vals] & adj[a, d_vals]
            
            # 2. b dominates: adj[b,a] & adj[b,c_vals] & adj[b,d_vals]
            dom_b = adj[b, a] & adj[b, c_vals] & adj[b, d_vals]
            
            # 3. c dominates: adj[c,a] & adj[c,b] & adj[c,d]
            # adj[c_vals, a] is a vector. adj[c_vals, b] is a vector.
            # adj[c_vals, d_vals] is not a simple vector lookup, it's a matrix lookup.
            # We need adj[c_vals[i], d_vals[i]] for each i.
            # This is equivalent to adj[c_vals, d_vals] using advanced indexing.
            dom_c = adj[c_vals, a] & adj[c_vals, b] & adj[c_vals, d_vals]
            
            # 4. d dominates: adj[d,a] & adj[d,b] & adj[d,c]
            dom_d = adj[d_vals, a] & adj[d_vals, b] & adj[d_vals, c_vals]
            
            # If any of these is True for a specific (c,d) pair, it's valid.
            # We check if there is ANY pair where ALL are False.
            valid_pairs = dom_a | dom_b | dom_c | dom_d
            
            if not np.all(valid_pairs):
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # The known minimal solution is n=67 (Paley tournament).
    # We verify it quickly.
    p = 67
    n, orient = _paley_orientation(p)
    
    if _check_four_dominator_vectorized(n, orient):
        return {'n': n, 'orientation': orient}
    
    # If 67 fails (shouldn't, but just in case), try 71, 79, 83
    for p in [71, 79, 83]:
        if time.time() > deadline:
            break
        n, orient = _paley_orientation(p)
        if _check_four_dominator_vectorized(n, orient):
            return {'n': n, 'orientation': orient}
            
    # Fallback: Return the smallest n we could verify or just 83
    # Since 67 is the theoretical minimum known, and we checked it, this block is unlikely.
    n, orient = _paley_orientation(83)
    return {'n': n, 'orientation': orient}

# EVOLVE-BLOCK-END
