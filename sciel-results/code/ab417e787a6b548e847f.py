"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build neighbor masks
    neigh_mask = [0] * n
    for u, v in edges:
        neigh_mask[u] |= (1 << v)
        neigh_mask[v] |= (1 << u)
        
    # Precompute validity of all subsets
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_valid = [False] * (1 << n)
    
    is_clique[0] = True
    is_indep[0] = True
    is_valid[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (neigh_mask[v] & rest) == rest:
            is_clique[mask] = True
        if is_indep[rest] and (neigh_mask[v] & rest) == 0:
            is_indep[mask] = True
        if is_clique[mask] or is_indep[mask]:
            is_valid[mask] = True
            
    # Group valid masks by the lowest vertex they contain? 
    # Or just by vertex they contain.
    # For the search, we pick the lowest set bit 'v' in 'rem'.
    # We need valid masks 'm' such that 'm' is subset of 'rem' and 'v' in 'm'.
    # Let's precompute valid_masks_containing[v] for all v.
    valid_masks_containing = [[] for _ in range(n)]
    for m in range(1, 1 << n):
        if is_valid[m]:
            # Add to all vertices in m? No, just iterate bits.
            # It's better to just iterate bits of m.
            mm = m
            while mm:
                lsb = mm & -mm
                v = lsb.bit_length() - 1
                valid_masks_containing[v].append(m)
                mm ^= lsb
                
    # To speed up iteration, we can sort valid_masks_containing[v] by popcount descending.
    # Larger parts first might lead to solutions faster in DFS.
    for v in range(n):
        valid_masks_containing[v].sort(key=lambda x: bin(x).count('1'), reverse=True)

    full_mask = (1 << n) - 1
    
    # Memoization table
    # memo[rem][k] = True/False
    # We can use a dictionary or a list of lists.
    # rem is up to 2^14. k is up to n.
    memo = {}
    
    def can_cover(rem, k):
        if rem == 0:
            return k >= 0
        if k == 0:
            return rem == 0
            
        key = (rem, k)
        if key in memo:
            return memo[key]
            
        lsb = rem & -rem
        v = lsb.bit_length() - 1
        
        for m in valid_masks_containing[v]:
            if (m & rem) == m:
                if can_cover(rem ^ m, k - 1):
                    memo[key] = True
                    return True
                    
        memo[key] = False
        return False
        
    # Find minimum k
    for k in range(1, n + 1):
        if can_cover(full_mask, k):
            # Reconstruct path
            # We need to store the choice.
            # Let's do a second pass to get the path.
            path = []
            rem = full_mask
            kk = k
            while rem != 0:
                lsb = rem & -rem
                v = lsb.bit_length() - 1
                found = False
                for m in valid_masks_containing[v]:
                    if (m & rem) == m:
                        if can_cover(rem ^ m, kk - 1):
                            path.append(m)
                            rem ^= m
                            kk -= 1
                            found = True
                            break
                if not found:
                    break # Should not happen
            parts = []
            for m in path:
                verts = [i for i in range(n) if (m & (1 << i)) != 0]
                parts.append(verts)
            return parts
            
    # Fallback
    return [[i] for i in range(n)]

# EVOLVE-BLOCK-END
