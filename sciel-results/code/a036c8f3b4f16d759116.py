"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline, 10s margin
    
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find all maximal cliques using Bron-Kerbosch
    def bron_kerbosch(R, P_set, X, cliques):
        if time.time() > deadline:
            return
        if not P_set and not X:
            cliques.append(list(R))
            return
        # Pivot selection
        if P_set or X:
            pivot = max(P_set | X, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v], cliques)
            P_set.remove(v)
            X.add(v)
    
    cliques = []
    bron_kerbosch(set(), set(range(n)), set(), cliques)
    
    # Sort cliques by size descending
    cliques.sort(key=len, reverse=True)
    
    best_branches = None
    best_paths = None
    
    # Check each maximal clique
    for clique in cliques:
        if time.time() > deadline:
            break
        k = len(clique)
        if k < 2:
            continue
        
        # A clique is always a valid clique subdivision (paths of length 1)
        branches = sorted(clique)
        paths = []
        for i in range(k):
            for j in range(i+1, k):
                paths.append([branches[i], branches[j]])
        
        if best_branches is None or k > len(best_branches):
            best_branches = branches
            best_paths = paths
    
    # Try to extend beyond cliques
    # For each maximal clique, try adding vertices
    for clique in cliques[:10]:  # Only try top 10 cliques
        if time.time() > deadline:
            break
        base = set(clique)
        
        # Try adding each non-clique vertex
        for v in range(n):
            if time.time() > deadline:
                break
            if v in base:
                continue
            candidate = base | {v}
            k = len(candidate)
            
            # Check if candidate is a valid clique subdivision
            # Need k(k-1)/2 internally vertex-disjoint paths
            valid, paths = check_clique_subdivision(candidate, adj, n)
            if valid:
                branches = sorted(candidate)
                if best_branches is None or k > len(best_branches):
                    best_branches = branches
                    best_paths = paths
    
    # Fallback: always have at least 2 branches
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]
    
    return {'branches': list(best_branches), 'paths': [list(p) for p in best_paths]}

def check_clique_subdivision(branches, adj, n):
    """Check if branches form a valid clique subdivision. Returns (valid, paths)."""
    k = len(branches)
    if k < 2:
        return False, []
    
    branch_set = set(branches)
    pairs = list(combinations(sorted(branches), 2))
    
    # For each pair, find a path avoiding other branches
    # All paths must be internally vertex-disjoint
    paths = {}
    used_interior = set()
    
    # Sort pairs by difficulty (fewer options first)
    def pair_key(pair):
        u, v = pair
        # Count available intermediate vertices
        avail = 0
        for w in range(n):
            if w not in branch_set and w not in used_interior:
                if w in adj[u] and w in adj[v]:
                    avail += 1
        return avail
    
    pairs_sorted = sorted(pairs, key=pair_key)
    
    for u, v in pairs_sorted:
        # Find a path from u to v avoiding branch_set - {u,v} and used_interior
        path = find_disjoint_path(u, v, branch_set - {u, v}, used_interior, adj, n)
        if path is None:
            return False, None
        paths[(u, v)] = path
        # Mark interior vertices as used
        for w in path[1:-1]:
            used_interior.add(w)
    
    # Build result
    result_paths = []
    for u, v in pairs:
        result_paths.append(paths[(u, v)])
    
    return True, result_paths

def find_disjoint_path(u, v, forbidden, used, adj, n):
    """Find a simple path from u to v avoiding forbidden and used vertices."""
    from collections import deque
    
    # BFS
    queue = deque()
    queue.append((u, [u]))
    visited = {u}
    
    while queue:
        node, path = queue.popleft()
        if node == v:
            return path
        for neighbor in adj[node]:
            if neighbor in visited:
                continue
            if neighbor in forbidden:
                continue
            if neighbor in used:
                continue
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    
    return None
# EVOLVE-BLOCK-END
