import itertools
import time

def construct():
    # Check Paley tournaments for p = 67, 71, 79, 83
    # p must be 3 mod 4 for Paley tournament to be a tournament (antisymmetric)
    candidates = [67, 71, 79, 83]
    
    for p in candidates:
        if p % 4 != 3:
            continue
        # Construct Paley tournament
        squares = {x * x % p for x in range(1, p)}
        # orientation[i] corresponds to pair (i, i+1) ... wait, the format is specific
        # "one bit per pair (i,j), i<j, of vertices 0..n-1 in lexicographic order"
        # So index k in orientation list corresponds to pair (i, j)
        # We need to generate the list
        
        # First, check validity using bitmasks
        # In[v] is a bitmask of vertices u such that u -> v
        In = [0] * p
        # For u in 0..p-1
        # For v in 0..p-1, v != u
        # If (v - u) % p is in squares, then v -> u?
        # Standard Paley: i -> j iff (j - i) is a quadratic residue mod p.
        # So for a fixed target u (vertex being dominated), the dominators v are those where (u - v) % p is a residue?
        # i -> j means j - i is residue.
        # So v -> u means u - v is residue.
        # Let's build In[u]
        for u in range(p):
            mask = 0
            for v in range(p):
                if v == u:
                    continue
                # v -> u if (u - v) % p is in squares
                if (u - v) % p in squares:
                    mask |= (1 << v)
            In[u] = mask
            
        # Check all 4-sets
        valid = True
        # Optimization: If any 4-set has empty intersection of In, it's invalid.
        # We can iterate through combinations.
        # For n=67, this is fast.
        for S in itertools.combinations(range(p), 4):
            inter = In[S[0]]
            for u in S[1:]:
                inter &= In[u]
                if inter == 0:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            # Construct the orientation list
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    # i -> j if (j - i) % p in squares
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': p, 'orientation': orientation}
            
    # Fallback to 83 if something went wrong (should not happen)
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
