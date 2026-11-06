# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    deadline = time.time() + 80
    P = 101
    N = 10
    # Precompute inverses
    inv = [pow(x, -1, P) for x in range(1, N + 1)]
    
    # Precompute all non-empty subsets
    # subset_info[i] = (residue, mask, size)
    subset_info = []
    for mask in range(1, 1 << N):
        size = 0
        res = 0
        for i in range(N):
            if mask & (1 << i):
                size += 1
                res = (res + inv[i]) % P
        subset_info.append((res, mask, size))
        
    num_subsets = len(subset_info)
    
    def solve_matching_HK(L):
        # Adjacency list for residues
        adj = [[] for _ in range(P)]
        for j in range(num_subsets):
            res, mask, size = subset_info[j]
            if size <= L:
                adj[res].append(j)
        
        # Hopcroft-Karp
        # u: 0..P-1 (residues), v: 0..num_subsets-1 (subsets)
        pair_u = [-1] * P
        pair_v = [-1] * num_subsets
        dist = [0] * P
        
        def bfs():
            q = deque()
            for u in range(P):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
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
            for u in range(P):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
            if matching == P:
                break
                
        if matching < P:
            return None
        
        assign = [None] * P
        for u in range(P):
            assign[u] = pair_u[u]
        return assign

    # Try L from 1 to N
    for L in range(1, N + 1):
        if time.time() > deadline:
            break
        result = solve_matching_HK(L)
        if result is not None:
            final_lists = []
            for r in range(P):
                j = result[r]
                mask = subset_info[j][1]
                terms = [x + 1 for x in range(N) if mask & (1 << x)]
                final_lists.append(terms)
            return final_lists
    
    # Fallback
    used = set()
    final_lists = []
    for r in range(P):
        found = False
        for j in range(num_subsets):
            res, mask, size = subset_info[j]
            if res == r and j not in used:
                used.add(j)
                terms = [x + 1 for x in range(N) if mask & (1 << x)]
                final_lists.append(terms)
                found = True
                break
        if not found:
            final_lists.append([1])
    return final_lists

# EVOLVE-BLOCK-END
