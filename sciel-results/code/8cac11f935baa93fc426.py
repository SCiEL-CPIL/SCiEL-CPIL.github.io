"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from collections import defaultdict, deque
    
    # Precompute modular inverses
    inv = [0] + [pow(x, -1, 101) for x in range(1, 11)]
    
    # Generate all masks
    num_masks = 1 << 10
    mask_residue = [0] * num_masks
    mask_popcount = [0] * num_masks
    mask_terms = [[] for _ in range(num_masks)]
    
    for mask in range(1, num_masks):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        residue = sum(inv[x] for x in terms) % 101
        popcount = len(terms)
        mask_residue[mask] = residue
        mask_popcount[mask] = popcount
        mask_terms[mask] = terms

    def max_matching(L):
        # Build adjacency list for left nodes (residues)
        # adj[r] = list of masks m such that mask_residue[m] == r and mask_popcount[m] <= L
        adj = defaultdict(list)
        for m in range(1, num_masks):
            if mask_popcount[m] <= L:
                r = mask_residue[m]
                adj[r].append(m)
        
        # If any residue has no candidates, return False
        for r in range(101):
            if not adj[r]:
                return False, {}
                
        # Hopcroft-Karp
        pair_u = [-1] * 101 # for residues
        pair_v = [-1] * num_masks # for masks
        dist = [0] * 101
        
        def bfs():
            queue = deque()
            for u in range(101):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found

        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(101):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
                        
        if matching == 101:
            result = [None] * 101
            for r in range(101):
                m = pair_u[r]
                result[r] = mask_terms[m]
            return True, result
        else:
            return False, {}

    # Binary search for minimum L
    lo, hi = 1, 10
    best_L = 10
    best_result = None
    
    while lo <= hi:
        mid = (lo + hi) // 2
        success, res = max_matching(mid)
        if success:
            best_L = mid
            best_result = res
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_result is None:
        # Fallback to greedy if matching fails (should not happen)
        min_pop = [float('inf')] * 101
        best_terms = [None] * 101
        for m in range(1, num_masks):
            r = mask_residue[m]
            if mask_popcount[m] < min_pop[r]:
                min_pop[r] = mask_popcount[m]
                best_terms[r] = mask_terms[m]
        return best_terms
        
    return best_result
# EVOLVE-BLOCK-END
