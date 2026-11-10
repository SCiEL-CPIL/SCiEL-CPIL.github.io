import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    p = 101
    denominators = list(range(1, 11))
    invs = [pow(x, -1, p) for x in denominators]
    
    # Precompute all non-empty subsets with their residue and length
    # 2^10 - 1 = 1023 subsets
    subsets_by_residue = {r: [] for r in range(p)}
    
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for i in range(10):
            if mask & (1 << i):
                terms.append(denominators[i])
                s += invs[i]
        s %= p
        length = len(terms)
        subsets_by_residue[s].append((length, tuple(terms)))
    
    # Check if all residues have at least one subset
    for r in range(p):
        if not subsets_by_residue[r]:
            # Fallback: use any valid assignment
            assignment = {}
            for r2 in range(p):
                if subsets_by_residue[r2]:
                    min_len = min(item[0] for item in subsets_by_residue[r2])
                    best = None
                    for item in subsets_by_residue[r2]:
                        if item[0] == min_len:
                            if best is None or item[1] < best:
                                best = item[1]
                    assignment[r2] = list(best)
                else:
                    assignment[r2] = [1]
            return [assignment[r] for r in range(p)]
    
    # Find minimum k such that all residues can be covered using subsets of size <= k
    # Use bipartite matching: left = residues (101), right = subsets (1023)
    # Edge exists if subset's residue matches and size <= k
    
    def can_cover(k):
        """Check if all 101 residues can be matched to distinct subsets of size <= k"""
        # Build adjacency: for each residue, list of subset indices with size <= k
        # Subset index: 0 to 1022 (mask 1 to 1023)
        
        # Pre-build adjacency for this k
        adj = [[] for _ in range(p)]
        for r in range(p):
            for (length, terms) in subsets_by_residue[r]:
                if length <= k:
                    # Find the mask for this subset
                    mask = 0
                    for d in terms:
                        mask |= (1 << (d - 1))
                    adj[r].append(mask)
        
        # Hopcroft-Karp algorithm
        # Left side: residues 0..100
        # Right side: masks 1..1023
        
        pair_u = [-1] * p  # pair_u[u] = matched right node for left node u
        pair_v = {}  # pair_v[v] = matched left node for right node v
        dist = [0] * p
        
        def bfs():
            queue = deque()
            found = False
            for u in range(p):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    matched_u = pair_v.get(v, -1)
                    if matched_u == -1:
                        found = True
                    elif dist[matched_u] == -1:
                        dist[matched_u] = dist[u] + 1
                        queue.append(matched_u)
            return found
        
        def dfs(u):
            for v in adj[u]:
                matched_u = pair_v.get(v, -1)
                if matched_u == -1 or (dist[matched_u] == dist[u] + 1 and dfs(matched_u)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False
        
        matching = 0
        while bfs():
            for u in range(p):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
                        if matching == p:
                            break
            if matching == p:
                break
        
        return matching == p
    
    # Binary search or linear search for minimum k
    # Max possible k is 10 (all denominators)
    best_k = 10
    for k in range(1, 11):
        if time.time() > deadline - 5.0:
            break
        if can_cover(k):
            best_k = k
            break
    
    # Now build the actual assignment using the found k
    # We need to find a valid matching for k = best_k
    adj = [[] for _ in range(p)]
    for r in range(p):
        for (length, terms) in subsets_by_residue[r]:
            if length <= best_k:
                mask = 0
                for d in terms:
                    mask |= (1 << (d - 1))
                adj[r].append(mask)
    
    pair_u = [-1] * p
    pair_v = {}
    dist = [0] * p
    
    def bfs_final():
        queue = deque()
        found = False
        for u in range(p):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                matched_u = pair_v.get(v, -1)
                if matched_u == -1:
                    found = True
                elif dist[matched_u] == -1:
                    dist[matched_u] = dist[u] + 1
                    queue.append(matched_u)
        return found
    
    def dfs_final(u):
        for v in adj[u]:
            matched_u = pair_v.get(v, -1)
            if matched_u == -1 or (dist[matched_u] == dist[u] + 1 and dfs_final(matched_u)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False
    
    matching = 0
    while bfs_final():
        for u in range(p):
            if pair_u[u] == -1:
                if dfs_final(u):
                    matching += 1
                    if matching == p:
                        break
        if matching == p:
            break
    
    # Build result
    assignment = {}
    for r in range(p):
        if pair_u[r] != -1:
            mask = pair_u[r]
            terms = []
            for i in range(10):
                if mask & (1 << i):
                    terms.append(denominators[i])
            assignment[r] = terms
        else:
            # Fallback
            min_len = min(item[0] for item in subsets_by_residue[r])
            best = None
            for item in subsets_by_residue[r]:
                if item[0] == min_len:
                    if best is None or item[1] < best:
                        best = item[1]
            assignment[r] = list(best)
    
    return [assignment[r] for r in range(p)]
