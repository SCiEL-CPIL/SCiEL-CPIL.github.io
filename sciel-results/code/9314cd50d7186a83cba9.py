import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # 1. Build adjacency masks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # 2. Precompute is_clique for G and complement G
    # is_clique_G[mask]
    is_clique_G = [False] * (1 << n)
    is_clique_barG = [False] * (1 << n)
    
    # Helper to check if mask is clique given adj
    def check_clique(mask, adj_list):
        if mask == 0: return True
        # Pick a vertex
        v = (mask & -mask).bit_length() - 1
        # All other vertices in mask must be neighbors of v
        others = mask ^ (1 << v)
        if others == 0: return True
        return (adj_list[v] & others) == others
        
    # This recursive check is O(size). Doing it for 16384 masks is fine.
    # But we can do it faster.
    # A mask is a clique iff for every v in mask, adj[v] contains mask\{v}.
    # Actually, it's enough to check if the intersection of adj[v] for all v in mask contains mask\{v}?
    # No.
    # Let's just use the property: mask is clique iff for all v in mask, (adj[v] & (mask ^ (1<<v))) == (mask ^ (1<<v)).
    
    for mask in range(1 << n):
        if mask == 0:
            is_clique_G[mask] = True
            is_clique_barG[mask] = True
            continue
            
        # Check G
        is_c = True
        m = mask
        while m:
            v_bit = m & -m
            v = v_bit.bit_length() - 1
            others = mask ^ v_bit
            if (adj[v] & others) != others:
                is_c = False
                break
            m ^= v_bit
        is_clique_G[mask] = is_c
        
        # Check complement G
        # adj_bar[v] = full_mask ^ (1<<v) ^ adj[v]
        is_c_bar = True
        m = mask
        while m:
            v_bit = m & -m
            v = v_bit.bit_length() - 1
            others = mask ^ v_bit
            adj_bar_v = full_mask ^ (1 << v) ^ adj[v]
            if (adj_bar_v & others) != others:
                is_c_bar = False
                break
            m ^= v_bit
        is_clique_barG[mask] = is_c_bar
        
    is_valid = [is_clique_G[i] or is_clique_barG[i] for i in range(1 << n)]
    
    # 3. DFS
    # Memoization? State is (rem, k).
    # rem has 2^14 states, k has 14 states. 229k states.
    # But we want the solution, not just boolean.
    # We can memoize "is solvable" to prune.
    solvable_memo = {}
    
    def dfs(rem, k):
        if rem == 0:
            return []
        if k == 0:
            return None
        if k == 1:
            if is_valid[rem]:
                return [rem]
            else:
                return None
                
        # Check memo
        key = (rem, k)
        if key in solvable_memo:
            if not solvable_memo[key]:
                return None
            # If solvable, we still need to find the path?
            # Memoizing the actual solution is hard because there are many.
            # Just memoize "is_solvable" for pruning.
            # If not solvable, return None.
            # If solvable, we proceed.
            
        # If we know it's not solvable, prune
        # We need to populate memo.
        
        # Find first vertex
        v_bit = rem & -rem
        v = v_bit.bit_length() - 1
        rest = rem ^ v_bit
        
        # Generate candidates
        candidates = []
        # Iterate submasks of rest
        s = rest
        while True:
            cand = s | v_bit
            if is_valid[cand]:
                candidates.append(cand)
            if s == 0: break
            s = (s - 1) & rest
            
        # Sort by size descending to find solution faster
        candidates.sort(key=lambda x: -bin(x).count('1'))
        
        for cand in candidates:
            new_rem = rem ^ cand
            res = dfs(new_rem, k - 1)
            if res is not None:
                solvable_memo[key] = True
                return [cand] + res
                
        solvable_memo[key] = False
        return None

    # Search for minimal k
    for k in range(1, n + 1):
        sol = dfs(full_mask, k)
        if sol is not None:
            # Convert masks to lists of vertices
            partition = []
            for mask in sol:
                part = []
                for i in range(n):
                    if mask & (1 << i):
                        part.append(i)
                partition.append(part)
            return partition
            
    # Fallback (should not happen)
    return [[i] for i in range(n)]
