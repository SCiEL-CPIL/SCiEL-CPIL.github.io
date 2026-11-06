"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all non-empty subsets: store (size, residue, mask)
    # 1023 subsets total
    all_subsets = []
    for mask in range(1, 1 << 10):
        size = bin(mask).count('1')
        residue = 0
        m = mask
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                residue = (residue + inv[x - 1]) % 101
        all_subsets.append((size, residue, mask))
    
    # Group subsets by residue for quick lookup
    subsets_by_residue = [[] for _ in range(101)]
    for i, (size, residue, mask) in enumerate(all_subsets):
        subsets_by_residue[residue].append(i)
    
    def can_match(K):
        """Check if there's a perfect matching using only subsets of size <= K.
        Uses Hopcroft-Karp algorithm."""
        # Left side: 101 residues (0..100)
        # Right side: 1023 subsets (0..1022)
        # Edge exists if subset[i] has residue == left_node and size <= K
        
        # Build adjacency: for each residue, which subsets (of size <= K) map to it
        adj = [[] for _ in range(101)]
        for r in range(101):
            for idx in subsets_by_residue[r]:
                if all_subsets[idx][0] <= K:
                    adj[r].append(idx)
        
        # Hopcroft-Karp
        n_left = 101
        n_right = 1023
        
        pair_u = [-1] * n_left   # pair_u[u] = v matched to u
        pair_v = [-1] * n_right  # pair_v[v] = u matched to v
        dist = [0] * n_left
        
        def bfs():
            from collections import deque
            queue = deque()
            for u in range(n_left):
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
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        
        return matching == n_left
    
    # Binary search on K (max allowed subset size)
    # K ranges from 1 to 10
    lo, hi = 1, 10
    best_K = 10
    
    while lo <= hi:
        if time.time() > deadline - 5:
            break
        mid = (lo + hi) // 2
        if can_match(mid):
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Now find the actual matching with best_K
    # Re-run Hopcroft-Karp to get the assignment
    K = best_K
    adj = [[] for _ in range(101)]
    for r in range(101):
        for idx in subsets_by_residue[r]:
            if all_subsets[idx][0] <= K:
                adj[r].append(idx)
    
    n_left = 101
    n_right = 1023
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left
    
    def bfs2():
        from collections import deque
        queue = deque()
        for u in range(n_left):
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
    
    def dfs2(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs2(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False
    
    matching = 0
    while bfs2():
        for u in range(n_left):
            if pair_u[u] == -1:
                if dfs2(u):
                    matching += 1
    
    # Build result: for each residue r, get the subset assigned
    result = []
    for r in range(101):
        idx = pair_u[r]
        if idx == -1:
            # Fallback: use any subset for this residue
            if subsets_by_residue[r]:
                idx = subsets_by_residue[r][0]
            else:
                idx = 0
        mask = all_subsets[idx][2]
        terms = [x for x in range(1, 11) if mask & (1 << (x - 1))]
        result.append(terms)
    
    return result
# EVOLVE-BLOCK-END
