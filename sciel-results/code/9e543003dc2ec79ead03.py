import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
        start_time = time.time()
        deadline = start_time + 50.0 # 50s safety margin for 60s limit
        
        n = P['n']
        edges = P['edges']
        
        # Build adjacency bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        full_mask = (1 << n) - 1
        
        # Precompute is_clique and is_ind for all subsets
        is_clique = [False] * (1 << n)
        is_ind = [False] * (1 << n)
        is_valid = [False] * (1 << n)
        
        is_clique[0] = True
        is_ind[0] = True
        is_valid[0] = True
        
        for mask in range(1, 1 << n):
            # Get lowest set bit
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            rest = mask ^ lsb
            
            # Clique check:
            # mask is clique if rest is clique and v is connected to all in rest
            if is_clique[rest] and (rest & ~adj[v]) == 0:
                is_clique[mask] = True
            else:
                is_clique[mask] = False
            
            # Independent set check:
            # mask is ind-set if rest is ind-set and v is NOT connected to any in rest
            if is_ind[rest] and (rest & adj[v]) == 0:
                is_ind[mask] = True
            else:
                is_ind[mask] = False
                
            is_valid[mask] = is_clique[mask] or is_ind[mask]
            
        # Memoization for can_cover
        memo = {}
        choice = {}
        
        def can_cover(rem, k):
            if rem == 0:
                return True
            if k == 0:
                return False
            if k == 1:
                if is_valid[rem]:
                    choice[(rem, 1)] = rem
                    return True
                else:
                    return False
                
            key = (rem, k)
            if key in memo:
                return memo[key]
                
            lsb = rem & -rem
            v = lsb.bit_length() - 1
            rest = rem ^ lsb
            
            # Iterate subsets of rest
            s = rest
            found = False
            while True:
                S = s | lsb
                if is_valid[S]:
                    if can_cover(rem ^ S, k - 1):
                        memo[key] = True
                        choice[key] = S
                        found = True
                        break
                if s == 0:
                    break
                s = (s - 1) & rest
                
            if not found:
                memo[key] = False
                
            return found
            
        # Find minimum k
        min_k = 1
        while min_k <= n:
            if time.time() > deadline:
                # Fallback: return a valid partition (e.g., all singletons, which are cliques/ind-sets)
                # Singletons are both cliques and independent sets.
                # This is a valid partition, just not necessarily minimal.
                return [[i] for i in range(n)]
            if can_cover(full_mask, min_k):
                break
            min_k += 1
            
        if min_k > n:
            return [[i] for i in range(n)]
            
        # Reconstruct partition
        parts = []
        rem = full_mask
        k = min_k
        while rem:
            key = (rem, k)
            S = choice[key]
            # Convert S to list of vertices
            part_vertices = []
            for i in range(n):
                if S & (1 << i):
                    part_vertices.append(i)
            parts.append(part_vertices)
            rem ^= S
            k -= 1
            
        # Convert to list of lists of ints
        # The requirement says "partition of vertices 0..13 into nonempty lists"
        # "sequences must be list or tuple"
        # "integers must be Python int"
        
        return parts
