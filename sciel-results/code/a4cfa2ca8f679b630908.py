import time
import random
from itertools import combinations

N = 20
DEADLINE_FRACTION = 0.90

def construct():
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION
    
    # adj[i] is an integer bitmask of neighbors of i
    def _popcount(x):
        return bin(x).count('1')
    
    def _is_valid_add(adj, u, v):
        # Check if adding edge (u, v) keeps graph K3,3-free
        # 1. Triples (u, x, y) where v is in N(x) and N(y).
        #    x, y must be in N(v).
        #    If |N(u) & N(x) & N(y)| >= 2, then adding v to N(u) makes it >= 3.
        nv = adj[v]
        # Get list of neighbors of v
        neigh_v = [i for i in range(N) if (nv >> i) & 1]
        for x, y in combinations(neigh_v, 2):
            common = adj[u] & adj[x] & adj[y]
            if _popcount(common) >= 2:
                return False
        
        # 2. Triples (v, x, y) where u is in N(x) and N(y).
        #    x, y must be in N(u).
        nu = adj[u]
        neigh_u = [i for i in range(N) if (nu >> i) & 1]
        for x, y in combinations(neigh_u, 2):
            common = adj[v] & adj[x] & adj[y]
            if _popcount(common) >= 2:
                return False
        
        return True

    def _add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def _remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def _edge_count(adj):
        return sum(_popcount(a) for a in adj) // 2

    def _local_search_add(adj):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any missing edge
            # Randomize order
            pairs = [(u, v) for u in range(N) for v in range(u+1, N) if not ((adj[u] >> v) & 1)]
            random.shuffle(pairs)
            for u, v in pairs:
                if _is_valid_add(adj, u, v):
                    _add_edge(adj, u, v)
                    improved = True
                    if time.time() > deadline:
                        break
        return adj

    def _local_search_swap(adj):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Get existing and missing edges
            existing = [(u, v) for u in range(N) for v in range(u+1, N) if (adj[u] >> v) & 1]
            missing = [(u, v) for u in range(N) for v in range(u+1, N) if not ((adj[u] >> v) & 1)]
            
            if not missing:
                break
                
            random.shuffle(existing)
            random.shuffle(missing)
            
            for u1, v1 in existing:
                if time.time() > deadline:
                    break
                for u2, v2 in missing:
                    # Skip if same edge
                    if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                        continue
                    
                    # Try swap
                    _remove_edge(adj, u1, v1)
                    
                    if _is_valid_add(adj, u2, v2):
                        _add_edge(adj, u2, v2)
                        improved = True
                        # Rebuild lists
                        existing = [(u, v) for u in range(N) for v in range(u+1, N) if (adj[u] >> v) & 1]
                        missing = [(u, v) for u in range(N) for v in range(u+1, N) if not ((adj[u] >> v) & 1)]
                        random.shuffle(existing)
                        random.shuffle(missing)
                        break
                    else:
                        # Revert
                        _add_edge(adj, u1, v1)
                if improved:
                    break
        return adj

    best_adj = None
    best_count = 0
    
    # Multiple restarts
    restarts = 0
    while time.time() < deadline and restarts < 200:
        restarts += 1
        
        # Initialize with random graph
        adj = [0] * N
        # Add random edges greedily
        all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if _is_valid_add(adj, u, v):
                _add_edge(adj, u, v)
            if time.time() > deadline:
                break
        
        # Refine
        adj = _local_search_add(adj)
        adj = _local_search_swap(adj)
        
        count = _edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            
        # If we hit a high density, we might want to keep searching more?
        # The deadline guard handles this.

    if best_adj is None:
        return []

    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
