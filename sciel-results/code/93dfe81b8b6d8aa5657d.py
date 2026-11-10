"""Rewritten program: fast greedy clique + disjoint path verification."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

_DEADLINE = time.time() + 50.0

def _build_graph():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _build_square(adj):
    n = len(adj)
    sq = [set() for _ in range(n)]
    # BFS for each node to get distance <= 2
    for u in range(n):
        dist = [-1] * n
        dist[u] = 0
        q = deque([u])
        while q:
            curr = q.popleft()
            for v in adj[curr]:
                if dist[v] == -1:
                    dist[v] = dist[curr] + 1
                    q.append(v)
        for v in range(n):
            if v != u and dist[v] <= 2:
                sq[u].add(v)
                sq[v].add(u)
    return sq

def _find_shortest_path(adj, src, dst, blocked):
    """BFS for shortest path avoiding blocked vertices (except src/dst)."""
    if src == dst:
        return [src]
    if src in blocked or dst in blocked:
        return None
        
    parent = [-1] * len(adj)
    parent[src] = -2
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            path = []
            cur = dst
            while cur != -2:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for v in adj[u]:
            if parent[v] == -1 and (v == dst or v not in blocked):
                parent[v] = u
                q.append(v)
    return None

def _try_find_disjoint_paths(adj, branches):
    """
    Attempts to find internally disjoint paths for all pairs in branches.
    Uses a heuristic order: pairs with shortest distance first.
    """
    if len(branches) < 2:
        return None
        
    k = len(branches)
    pairs = list(combinations(range(k), 2))
    
    # Calculate heuristic weight: shortest path length in original graph
    def pair_weight(p):
        i, j = p
        if branches[i] == branches[j]: return 0
        parent = [-1] * P['n']
        parent[branches[i]] = -2
        q = deque([branches[i]])
        while q:
            u = q.popleft()
            if u == branches[j]:
                length = 0
                cur = u
                while cur != -2:
                    length += 1
                    cur = parent[cur]
                return length
            for v in adj[u]:
                if parent[v] == -1:
                    parent[v] = u
                    q.append(v)
        return P['n']
        
    pairs.sort(key=pair_weight)
    
    blocked = set(branches)
    paths_found = {}
    
    for i, j in pairs:
        b_i = branches[i]
        b_j = branches[j]
        path = _find_shortest_path(adj, b_i, b_j, blocked)
        if path is None:
            return None
        for v in path[1:-1]:
            blocked.add(v)
        paths_found[(i, j)] = path
        
    return paths_found

def _greedy_clique_and_verify(adj, sq):
    """
    Greedily build a clique in sq, then verify paths.
    If verification fails, remove vertices and retry.
    """
    n = P['n']
    degs = [len(sq[i]) for i in range(n)]
    
    best_clique = []
    best_paths = None
    
    candidates = sorted(range(n), key=lambda x: degs[x], reverse=True)
    
    for start_v in candidates[:5]:
        if time.time() > _DEADLINE:
            break
            
        clique = [start_v]
        changed = True
        while changed and time.time() < _DEADLINE:
            changed = False
            candidates_to_add = []
            for v in range(n):
                if v in clique:
                    continue
                if all(u in sq[v] for u in clique):
                    candidates_to_add.append(v)
            
            if not candidates_to_add:
                break
                
            best_v = max(candidates_to_add, key=lambda v: len(sq[v]))
            clique.append(best_v)
            changed = True
            
        current_clique = clique[:]
        
        while current_clique and time.time() < _DEADLINE:
            paths = _try_find_disjoint_paths(adj, current_clique)
            if paths is not None:
                if len(current_clique) > len(best_clique):
                    best_clique = current_clique[:]
                    best_paths = paths
                break
            
            if len(current_clique) <= 2:
                break
            current_clique.pop()
            
    return best_clique, best_paths

def construct():
    adj = _build_graph()
    sq = _build_square(adj)
    
    best_clique, best_paths = _greedy_clique_and_verify(adj, sq)
    
    if not best_clique or best_paths is None:
        if P['edges']:
            u, v = P['edges'][0]
            return {
                'branches': [int(u), int(v)],
                'paths': [[int(u), int(v)]]
            }
        else:
            return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    branches = sorted(best_clique)
    k = len(branches)
    
    path_map = {}
    for (i, j), path in best_paths.items():
        u = best_clique[i]
        v = best_clique[j]
        key = (min(u, v), max(u, v))
        path_map[key] = path
            
    out_paths = []
    for i in range(k):
        for j in range(i + 1, k):
            u = branches[i]
            v = branches[j]
            key = (min(u, v), max(u, v))
            if key in path_map:
                out_paths.append([int(x) for x in path_map[key]])
            else:
                out_paths.append([int(u), int(v)])
                    
    return {
        'branches': [int(b) for b in branches],
        'paths': out_paths
    }

# EVOLVE-BLOCK-END
