"""Find a Hamilton cycle with minimal maximum edge-arrival index on 24 vertices."""
import time
import random
from itertools import combinations

# Build the edge order and index map
def _build_edge_data():
    n = 24
    rng = random.Random(614)
    all_edges = list(combinations(range(n), 2))
    rng.shuffle(all_edges)
    edge_index = {}
    for i, e in enumerate(all_edges):
        edge_index[e] = i + 1
    return n, all_edges, edge_index

N, EDGE_LIST, EDGE_IDX = _build_edge_data()

def _score_cycle(cycle):
    n = len(cycle)
    mx = 0
    for i in range(n):
        a, b = cycle[i], cycle[(i + 1) % n]
        e = (a, b) if a < b else (b, a)
        idx = EDGE_IDX[e]
        if idx > mx:
            mx = idx
    return mx

def _find_hamilton_cycle(adj, n, time_limit):
    """
    Find a Hamilton cycle in an undirected graph given as adjacency bitmasks.
    Uses DFS with pruning. Returns a list of vertices or None.
    """
    start_time = time.time()
    
    # Precompute neighbor lists sorted by degree
    deg = [bin(adj[v]).count('1') for v in range(n)]
    # For each vertex, list of neighbors sorted by (degree, vertex)
    neigh = []
    for v in range(n):
        nbrs = []
        for u in range(n):
            if u != v and (adj[v] & (1 << u)):
                nbrs.append(u)
        nbrs.sort(key=lambda u: (deg[u], u))
        neigh.append(nbrs)
    
    best = [None]
    
    def dfs(v, visited, path, depth):
        if time.time() - start_time > time_limit:
            return False
        if depth == n:
            if adj[v] & 1:  # edge to vertex 0
                best[0] = path[:]
                return True
            return False
        
        for u in neigh[v]:
            if u == 0:
                continue
            if visited & (1 << u):
                continue
            # Pruning: check if the remaining unvisited vertices are connected
            # to each other and to u (necessary condition for completing the cycle)
            # This is expensive, so only do it occasionally
            if depth > 10 and depth % 3 == 0:
                # Check connectivity of unvisited vertices plus u
                unvisited = ((1 << n) - 1) & ~visited & ~(1 << u)
                # BFS from u on the subgraph induced by unvisited ∪ {u}
                mask = unvisited | (1 << u)
                seen = 1 << u
                stack = [u]
                while stack:
                    x = stack.pop()
                    nbr_mask = adj[x] & mask & ~seen
                    while nbr_mask:
                        lsb = nbr_mask & (-nbr_mask)
                        y = lsb.bit_length() - 1
                        seen |= lsb
                        stack.append(y)
                        nbr_mask ^= lsb
                if seen != mask:
                    continue
            
            visited |= (1 << u)
            path.append(u)
            if dfs(u, visited, path, depth + 1):
                return True
            path.pop()
            visited ^= (1 << u)
        
        return False
    
    visited = 1  # vertex 0
    path = [0]
    dfs(0, visited, path, 1)
    
    return best[0]

def construct():
    start = time.time()
    deadline = start + 80
    
    # Baseline
    base_cycle = list(range(N))
    base_score = _score_cycle(base_cycle)
    best_cycle = base_cycle
    best_score = base_score
    
    # Binary search for minimum K
    lo, hi = 24, len(EDGE_LIST)
    
    while lo < hi and time.time() < deadline - 10:
        mid = (lo + hi) // 2
        adj = [0] * N
        for i in range(mid):
            a, b = EDGE_LIST[i]
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
        
        # Quick connectivity check
        seen = 1
        stack = [0]
        while stack:
            v = stack.pop()
            nm = adj[v] & ~seen
            while nm:
                lsb = nm & (-nm)
                u = lsb.bit_length() - 1
                seen |= lsb
                stack.append(u)
                nm ^= lsb
        if seen != (1 << N) - 1:
            lo = mid + 1
            continue
        
        cycle = _find_hamilton_cycle(adj, N, deadline - 5)
        if cycle is not None:
            hi = mid
            sc = _score_cycle(cycle)
            if sc < best_score:
                best_score = sc
                best_cycle = cycle
        else:
            lo = mid + 1
    
    # Local search: 2-opt
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            n = len(best_cycle)
            for i in range(n):
                for j in range(i + 1, n):
                    new_cycle = best_cycle[:i+1] + best_cycle[i+1:j+1][::-1] + best_cycle[j+1:]
                    sc = _score_cycle(new_cycle)
                    if sc < best_score:
                        best_cycle = new_cycle
                        best_score = sc
                        improved = True
                        break
                if improved:
                    break
    
    return best_cycle
