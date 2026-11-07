"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    for i in range(n):
        adj_list[i].sort(key=lambda x: -degrees[x])
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
    
    def find_path(src, dst, forbidden, max_len=5):
        """Find a simple path from src to dst avoiding forbidden vertices (except src, dst)."""
        if src == dst:
            return None
        if src in forbidden or dst in forbidden:
            return None
        
        # BFS
        from collections import deque
        q = deque()
        q.append((src, [src]))
        visited = {src}
        
        while q:
            curr, path = q.popleft()
            if len(path) > max_len:
                continue
            if curr == dst:
                return path
            for nb in adj_list[curr]:
                if nb not in visited and (nb not in forbidden or nb == dst):
                    visited.add(nb)
                    q.append((nb, path + [nb]))
        return None
    
    def check_subdivision(branches):
        """Check if branches form a valid clique subdivision. Returns paths or None."""
        k = len(branches)
        branch_set = set(branches)
        
        # Generate all pairs, order by difficulty
        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                pairs.append((i, j))
        
        # Order: pairs without direct edge first, then by fewer common neighbors
        def pair_key(pair):
            i, j = pair
            b_i, b_j = branches[i], branches[j]
            has_direct = 1 if b_j in adj[b_i] else 0
            common = len((adj[b_i] & adj[b_j]) - branch_set)
            return (has_direct, -common)
        
        pairs.sort(key=pair_key)
        
        used_interiors = set()
        path_map = {}
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(pairs):
                return True
            
            i, j = pairs[idx]
            b_i = branches[i]
            b_j = branches[j]
            
            # Try direct edge first
            if b_j in adj[b_i]:
                path_map[(i, j)] = [b_i, b_j]
                if backtrack(idx + 1):
                    return True
                del path_map[(i, j)]
            
            # Try longer paths
            forbidden = used_interiors.copy()
            forbidden.update(branch_set - {b_i, b_j})
            
            path = find_path(b_i, b_j, forbidden, max_len=4)
            if path is not None:
                interior = set(path[1:-1])
                if interior.isdisjoint(branch_set) and interior.isdisjoint(used_interiors):
                    path_map[(i, j)] = path
                    used_interiors.update(interior)
                    if backtrack(idx + 1):
                        return True
                    used_interiors.difference_update(interior)
                    del path_map[(i, j)]
            
            return False
        
        if backtrack(0):
            result_paths = []
            for i in range(k):
                for j in range(i + 1, k):
                    result_paths.append(path_map[(i, j)])
            return result_paths
        return None
    
    # Try k from 8 down to 2
    for k in range(8, 1, -1):
        if time.time() > deadline:
            break
        
        pool_size = min(16, n)
        candidates = sorted_vertices[:pool_size]
        
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                break
            
            branches = list(combo)
            paths = check_subdivision(branches)
            if paths is not None:
                return {'branches': [int(b) for b in branches], 'paths': [list(p) for p in paths]}
    
    # Fallback: max clique
    best_clique = []
    def bron_kerbosch(R, P_set, X):
        nonlocal best_clique
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if P_set:
            pivot = max(P_set, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
    
    bron_kerbosch(set(), set(range(n)), set())
    
    if len(best_clique) >= 2:
        branches = sorted(best_clique)
        paths = []
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                paths.append([branches[i], branches[j]])
        return {'branches': [int(b) for b in branches], 'paths': [list(p) for p in paths]}
    
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

# EVOLVE-BLOCK-END
