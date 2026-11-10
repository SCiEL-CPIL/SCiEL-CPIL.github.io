import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a,b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    """
    N = 255
    deadline = time.time() + 55.0  # Safety margin
    
    # Candidates must share a factor with 255 (3, 5, 17)
    # 255 is always included.
    # We find the max clique in the graph of other candidates U.
    
    # 1. Generate candidates U = {x in [1, 254] | gcd(x, 255) > 1}
    candidates = []
    for x in range(1, N):
        if math.gcd(x, N) > 1:
            candidates.append(x)
            
    # Map values to indices
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    n_cands = len(candidates)
    
    # 2. Build adjacency bitsets
    # adj[i] is a bitmask of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [0] * n_cands
    for i in range(n_cands):
        vi = candidates[i]
        mask = 0
        for j in range(i + 1, n_cands):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                mask |= (1 << j)
                adj[j] |= (1 << i) # symmetric
        adj[i] = mask

    # 3. Find Maximum Clique using Bron-Kerbosch with Pivoting
    # We only need the size, but we also want the set to return.
    # Since n_cands is ~126, recursion depth is manageable.
    
    max_clique_size = 0
    max_clique_set = []
    
    def bron_kerbosch(R, P, X):
        nonlocal max_clique_size, max_clique_set
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > max_clique_size:
                max_clique_size = len(R)
                max_clique_set = R[:]
            return
        
        # Pivot selection: choose u in P union X with max degree in P
        # For simplicity and speed with bitsets, we can pick any u in P if P is not empty, 
        # or u in X if P is empty. 
        # Standard pivot: u in P \cup X that maximizes |P \cap N(u)|
        # To keep it simple and fast, let's just pick the first available or use a simple heuristic.
        # Given the small size, even without perfect pivoting it should be fast.
        
        # Let's use a simple pivot: pick u from P if P exists, else from X.
        # Actually, Bron-Kerbosch with pivoting requires u in P U X.
        # If P is empty, we are at a leaf (handled above).
        # So P is not empty.
        
        # Find u in P U X with max neighbors in P.
        # We can just iterate through P U X.
        # For efficiency, let's just pick the first element in P as a pivot? 
        # No, that's not the standard pivot which reduces branching.
        # Let's implement a basic pivot: pick u from P with max degree in P.
        
        best_u = -1
        max_deg = -1
        # Iterate through P and X to find best pivot
        # P is a bitmask.
        # To iterate bits in a mask:
        def iter_bits(mask):
            while mask:
                lsb = mask & -mask
                yield lsb.bit_length() - 1
                mask ^= lsb

        # Combine P and X for pivot search
        PX = P | X
        for u in iter_bits(PX):
            # Degree of u in P
            # Neighbors of u in P: P & adj[u]
            deg = bin(P & adj[u]).count('1')
            if deg > max_deg:
                max_deg = deg
                best_u = u
        
        if best_u == -1:
            # Should not happen if P is not empty
            return

        # Recurse on P \ N(u)
        # The elements in P that are NOT neighbors of u
        # We iterate over v in P \ N(u)
        # P_not_N_u = P & ~adj[best_u]
        
        P_not_N_u = P & ~adj[best_u]
        
        for v in iter_bits(P_not_N_u):
            # R' = R U {v}
            R.append(candidates[v])
            # P' = P \ N(v) ... wait, standard BK:
            # P' = P \ N(v) ? No.
            # P' = P ∩ N(v)
            # X' = X ∩ N(v)
            
            P_new = P & adj[v]
            X_new = X & adj[v]
            
            bron_kerbosch(R, P_new, X_new)
            
            R.pop()
            
            # Move v from P to X
            P &= ~(1 << v)
            X |= (1 << v)

    # Initial call
    # R = []
    # P = all candidates (mask with n_cands bits set)
    # X = []
    initial_P = (1 << n_cands) - 1
    bron_kerbosch([], initial_P, 0)
    
    # Construct the final set
    # The clique found is in candidates indices.
    # We need to map back to values.
    if max_clique_set:
        # max_clique_set contains values from candidates list
        result = max_clique_set + [N]
    else:
        # Fallback: Multiples of 3
        result = [x for x in range(1, N + 1) if x % 3 == 0]
        
    # Verify validity and size
    # The set must contain 255.
    if N not in result:
        result.append(N)
        
    # Ensure no duplicates (though construction prevents it)
    # result = list(set(result))
    
    return result
