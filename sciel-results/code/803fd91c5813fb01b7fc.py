"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all non-empty subsets: (size, residue, mask)
    all_subsets = []
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        size = len(terms)
        residue = sum(inv[x - 1] for x in terms) % 101
        all_subsets.append((size, residue, mask))
    
    # Group subsets by residue
    residue_to_masks = [[] for _ in range(101)]
    for size, residue, mask in all_subsets:
        residue_to_masks[residue].append((size, mask))
    
    # Sort each residue's list by size (smaller first)
    for r in range(101):
        residue_to_masks[r].sort()
    
    # Binary search on maximum allowed size K
    # We need to find min K such that we can assign one subset to each residue
    # where each subset has size <= K, and no two residues get the same mask.
    # This is a bipartite matching problem: left = residues (101 nodes),
    # right = masks (1023 nodes), edge if mask's residue matches and size <= K.
    
    def can_match(K):
        """Check if perfect matching exists with all subset sizes <= K."""
        # Build adjacency: for each residue r, which masks are available
        adj = [[] for _ in range(101)]
        for r in range(101):
            for size, mask in residue_to_masks[r]:
                if size <= K:
                    adj[r].append(mask)
                else:
                    break  # sorted by size
        
        # Hopcroft-Karp algorithm
        # Left side: residues 0..100
        # Right side: masks 1..1023 (we use mask value as ID)
        # We'll use a dictionary for matchR: mask -> residue
        matchL = [-1] * 101  # matchL[r] = mask matched to residue r
        matchR = {}  # matchR[mask] = residue
        
        from collections import deque
        
        def bfs():
            # Build distance labels for unmatched left nodes
            dist = [-1] * 101
            queue = deque()
            for r in range(101):
                if matchL[r] == -1:
                    dist[r] = 0
                    queue.append(r)
            found = False
            while queue:
                r = queue.popleft()
                for mask in adj[r]:
                    m = matchR.get(mask, -1)
                    if m == -1:
                        found = True
                    elif dist[m] == -1:
                        dist[m] = dist[r] + 1
                        queue.append(m)
            return found, dist
        
        def dfs(r, dist):
            for mask in adj[r]:
                m = matchR.get(mask, -1)
                if m == -1 or (dist[m] == dist[r] + 1 and dfs(m, dist)):
                    matchL[r] = mask
                    matchR[mask] = r
                    return True
            dist[r] = -1
            return False
        
        # Run Hopcroft-Karp
        matching_size = 0
        while True:
            found, dist = bfs()
            if not found:
                break
            for r in range(101):
                if matchL[r] == -1:
                    if dfs(r, dist):
                        matching_size += 1
        return matching_size == 101
    
    # Binary search for minimum K
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
    
    # Now construct the actual assignment using best_K
    # Use a greedy + backtracking approach or just re-run matching
    adj = [[] for _ in range(101)]
    for r in range(101):
        for size, mask in residue_to_masks[r]:
            if size <= best_K:
                adj[r].append(mask)
            else:
                break
    
    from collections import deque
    
    matchL = [-1] * 101
    matchR = {}
    
    def bfs():
        dist = [-1] * 101
        queue = deque()
        for r in range(101):
            if matchL[r] == -1:
                dist[r] = 0
                queue.append(r)
        found = False
        while queue:
            r = queue.popleft()
            for mask in adj[r]:
                m = matchR.get(mask, -1)
                if m == -1:
                    found = True
                elif dist[m] == -1:
                    dist[m] = dist[r] + 1
                    queue.append(m)
        return found, dist
    
    def dfs(r, dist):
        for mask in adj[r]:
            m = matchR.get(mask, -1)
            if m == -1 or (dist[m] == dist[r] + 1 and dfs(m, dist)):
                matchL[r] = mask
                matchR[mask] = r
                return True
        dist[r] = -1
        return False
    
    matching_size = 0
    while True:
        if time.time() > deadline - 2:
            break
        found, dist = bfs()
        if not found:
            break
        for r in range(101):
            if matchL[r] == -1:
                if dfs(r, dist):
                    matching_size += 1
    
    # Build result
    result = [None] * 101
    for r in range(101):
        mask = matchL[r]
        if mask is not None and mask != -1:
            terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
            result[r] = terms
        else:
            # Fallback: use any subset for this residue
            if residue_to_masks[r]:
                size, mask = residue_to_masks[r][0]
                terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
                result[r] = terms
            else:
                result[r] = [1]  # shouldn't happen
    
    # Verify all residues are covered
    for r in range(101):
        if result[r] is None or len(result[r]) == 0:
            result[r] = [1]
    
    return result
# EVOLVE-BLOCK-END
