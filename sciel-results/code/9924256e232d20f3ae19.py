import time
import random
from itertools import combinations

N = 20
DEADLINE_FRACTION = 0.95

def construct():
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION

    # Precompute popcount for 0..2^20? No, 20 bits is small.
    # Use a fast popcount for 20-bit integers.
    # bin(x).count('1') is okay, but we can optimize with a lookup for 8-bit chunks.
    # Since N=20, we can just use bin. It's C-optimized.
    
    def _popcount(x):
        return bin(x).count('1')

    def _is_valid_add(adj, u, v):
        # Check if adding (u, v) creates a K3,3
        # K3,3 exists if there are x, y in N(u) such that N(x) & N(y) & N(v) has >= 2 bits
        # OR x, y in N(v) such that N(x) & N(y) & N(u) has >= 2 bits
        
        nu = adj[u]
        nv = adj[v]
        
        # Get neighbors of u
        neigh_u = [i for i in range(N) if (nu >> i) & 1]
        for x, y in combinations(neigh_u, 2):
            common = adj[x] & adj[y] & nv
            if _popcount(common) >= 2:
                return False

        # Get neighbors of v
        neigh_v = [i for i in range(N) if (nv >> i) & 1]
        for x, y in combinations(neigh_v, 2):
            common = adj[x] & adj[y] & nu
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

    def _greedy_fill(adj):
        # Try to add all missing edges in random order
        improved = True
        while improved:
            improved = False
            missing = [(u, v) for u in range(N) for v in range(u+1, N) if not ((adj[u] >> v) & 1)]
            random.shuffle(missing)
            for u, v in missing:
                if _is_valid_add(adj, u, v):
                    _add_edge(adj, u, v)
                    improved = True
                    if time.time() > deadline:
                        return adj
        return adj

    def _local_search_swap(adj, max_iters=1000):
        # Remove an edge, try to add a different one
        best_adj = adj[:]
        best_count = _edge_count(adj)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random existing edge to remove
            existing = [(u, v) for u in range(N) for v in range(u+1, N) if (adj[u] >> v) & 1]
            if not existing:
                break
            u1, v1 = random.choice(existing)
            _remove_edge(adj, u1, v1)
            
            # Try to add a random missing edge
            missing = [(u, v) for u in range(N) for v in range(u+1, N) if not ((adj[u] >> v) & 1)]
            if not missing:
                _add_edge(adj, u1, v1)
                break
            u2, v2 = random.choice(missing)
            
            if _is_valid_add(adj, u2, v2):
                _add_edge(adj, u2, v2)
                count = _edge_count(adj)
                if count > best_count:
                    best_count = count
                    best_adj = adj[:]
            else:
                _add_edge(adj, u1, v1)
        
        return best_adj

    def _kick_and_fill(adj, kick_size=5):
        # Remove kick_size random edges, then greedy fill
        adj_kicked = adj[:]
        existing = [(u, v) for u in range(N) for v in range(u+1, N) if (adj_kicked[u] >> v) & 1]
        random.shuffle(existing)
        for i in range(min(kick_size, len(existing))):
            u, v = existing[i]
            _remove_edge(adj_kicked, u, v)
        
        adj_kicked = _greedy_fill(adj_kicked)
        return adj_kicked

    best_adj = None
    best_count = 0

    # Initial greedy construction
    adj = [0] * N
    adj = _greedy_fill(adj)
    count = _edge_count(adj)
    if count > best_count:
        best_count = count
        best_adj = adj[:]

    # Local search on the initial solution
    adj_ls = _local_search_swap(best_adj, max_iters=500)
    count = _edge_count(adj_ls)
    if count > best_count:
        best_count = count
        best_adj = adj_ls[:]

    # Kick and fill loops
    while time.time() < deadline:
        # Kick the best solution
        adj_k = _kick_and_fill(best_adj, kick_size=random.randint(3, 8))
        
        # Local search on the kicked solution
        adj_ls = _local_search_swap(adj_k, max_iters=200)
        
        count = _edge_count(adj_ls)
        if count > best_count:
            best_count = count
            best_adj = adj_ls[:]
        else:
            # Occasionally restart from scratch or a random graph
            if random.random() < 0.05:
                adj_new = [0] * N
                # Random sparse init
                for u in range(N):
                    for v in range(u+1, N):
                        if random.random() < 0.2:
                            if _is_valid_add(adj_new, u, v):
                                _add_edge(adj_new, u, v)
                adj_new = _greedy_fill(adj_new)
                adj_new = _local_search_swap(adj_new, max_iters=100)
                count_new = _edge_count(adj_new)
                if count_new > best_count:
                    best_count = count_new
                    best_adj = adj_new[:]

    if best_adj is None:
        return []

    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
