"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from functools import lru_cache

_START_TIME = None

def _get_time_left(deadline):
    if _START_TIME is None:
        return deadline
    return deadline - (time.time() - _START_TIME)

def _is_k4_free(adj, n):
    """Check if graph with adjacency matrix adj on n vertices is K4-free."""
    for a in range(n):
        for b in range(a+1, n):
            if not adj[a][b]:
                continue
            for c in range(b+1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c+1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return False
    return True

def _max_triangle_free_induced(adj, n, deadline, start_time):
    """Find largest triangle-free induced subgraph size via branch and bound."""
    best = 0
    deadline_time = start_time + deadline
    
    # Precompute: for each vertex v, which other vertices are adjacent to v
    # A set S is triangle-free iff for every triple (a,b,c) in S, not all three edges exist
    
    # Use bitmask representation for speed
    # adj_mask[v] is a bitmask of vertices adjacent to v
    
    adj_mask = [0] * n
    for v in range(n):
        mask = 0
        for u in range(n):
            if adj[v][u]:
                mask |= (1 << u)
        adj_mask[v] = mask
    
    # For a set S (bitmask), it's triangle-free iff for every v in S,
    # the neighbors of v within S form an independent set (no edges among them)
    # i.e., for every v in S: adj_mask[v] & S & adj_mask[v] has no edges
    # Better: for every v in S, (adj_mask[v] & S) must be an independent set
    
    # Precompute: for each vertex v, the set of vertices that form a triangle with v
    # tri_with_v = {u : adj[v][u] and adj_mask[v] & adj_mask[u] != 0}
    # Actually, u and w form a triangle with v if adj[v][u], adj[v][w], adj[u][w]
    
    # Let's use a different approach: 
    # A set S is triangle-free iff for every v in S, the subgraph induced by 
    # (adj_mask[v] & S) is edge-free.
    
    # Precompute edge existence as a set of frozensets for quick lookup
    edge_set = set()
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v]:
                edge_set.add((u, v))
    
    def is_triangle_free_mask(mask):
        """Check if the set of vertices in mask induces a triangle-free graph."""
        # Get list of vertices in mask
        verts = []
        for v in range(n):
            if mask & (1 << v):
                verts.append(v)
        
        # Check all triples
        k = len(verts)
        for i in range(k):
            for j in range(i+1, k):
                if (verts[i], verts[j]) not in edge_set:
                    continue
                for l in range(j+1, k):
                    if (verts[i], verts[l]) in edge_set and (verts[j], verts[l]) in edge_set:
                        return False
        return True
    
    # Branch and bound with bitmask
    all_mask = (1 << n) - 1
    
    def dfs(available, current_mask, current_size):
        nonlocal best
        if time.time() > deadline_time:
            return
        if current_size > best:
            best = current_size
        if available == 0:
            return
        
        # Upper bound
        if current_size + bin(available).count('1') <= best:
            return
        
        # Try each vertex in available
        avail = available
        while avail:
            # Get lowest set bit
            lsb = avail & (-avail)
            v = lsb.bit_length() - 1
            avail ^= lsb
            
            # Check if v can be added to current
            can_add = True
            cur = current_mask
            while cur:
                lsb2 = cur & (-cur)
                u1 = lsb2.bit_length() - 1
                cur ^= lsb2
                if not (adj_mask[v] & (1 << u1)):
                    continue
                # Check all other vertices in current
                cur2 = current_mask & ~(1 << u1)
                while cur2:
                    lsb3 = cur2 & (-cur2)
                    u2 = lsb3.bit_length() - 1
                    cur2 ^= lsb3
                    if (adj_mask[v] & (1 << u2)) and (adj_mask[u1] & (1 << u2)):
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                new_current_mask = current_mask | (1 << v)
                new_available = available & ~(1 << v)
                dfs(new_available, new_current_mask, current_size + 1)
    
    dfs(all_mask, 0, 0)
    return best

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = 110  # seconds, leave margin for 120s limit
    n = 18
    
    best_edges = None
    best_score = 999
    
    def get_adjacency(edges, n):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj
    
    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges
    
    # Construction 1: K6,6,6 (complete tripartite) - max triangle-free is 12
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    adj = get_adjacency(edges, n)
    if _is_k4_free(adj, n):
        mtf = _max_triangle_free_induced(adj, n, 5, _START_TIME)
        if mtf < best_score:
            best_score = mtf
            best_edges = edges[:]
    
    # Construction 2: Add matchings within parts to break bipartite structure
    edges2 = edges[:]
    for a, b in [(0,3), (1,4), (2,5)]:
        edges2.append([a, b])
    for a, b in [(6,9), (7,10), (8,11)]:
        edges2.append([a, b])
    for a, b in [(12,15), (13,16), (14,17)]:
        edges2.append([a, b])
    
    adj2 = get_adjacency(edges2, n)
    if _is_k4_free(adj2, n):
        mtf2 = _max_triangle_free_induced(adj2, n, 5, _START_TIME)
        if mtf2 < best_score:
            best_score = mtf2
            best_edges = edges2[:]
    
    # Construction 3: Random K4-free graphs with local search
    def is_k4_free_fast(adj, n):
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]:
                    continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]:
                        continue
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return False
        return True
    
    def random_k4_free_graph(n):
        adj = [[False]*n for _ in range(n)]
        perm = list(range(n))
        random.shuffle(perm)
        sizes = [6, 6, 6]
        parts = []
        idx = 0
        for s in sizes:
            parts.append(perm[idx:idx+s])
            idx += s
        
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a][b] = True
                        adj[b][a] = True
        
        for part in parts:
            for i in range(len(part)):
                for j in range(i+1, len(part)):
                    if random.random() < 0.3:
                        adj[part[i]][part[j]] = True
                        adj[part[j]][part[i]] = True
        
        while not is_k4_free_fast(adj, n):
            found = False
            for a in range(n):
                for b in range(a+1, n):
                    if not adj[a][b]:
                        continue
                    for c in range(b+1, n):
                        if not adj[a][c] or not adj[b][c]:
                            continue
                        for d in range(c+1, n):
                            if adj[a][d] and adj[b][d] and adj[c][d]:
                                edges_in_k4 = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                                e = random.choice(edges_in_k4)
                                adj[e[0]][e[1]] = False
                                adj[e[1]][e[0]] = False
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        
        return adj
    
    def local_search(adj, n, max_iter=50):
        """Try to improve by adding/removing edges while maintaining K4-free."""
        adj = [row[:] for row in adj]
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Try adding a random edge
            u, v = random.sample(range(n), 2)
            if not adj[u][v]:
                adj[u][v] = True
                adj[v][u] = True
                if not is_k4_free_fast(adj, n):
                    adj[u][v] = False
                    adj[v][u] = False
                else:
                    # Check if this improves the score
                    mtf_new = _max_triangle_free_induced(adj, n, 2, _START_TIME)
                    if mtf_new >= best_score:
                        adj[u][v] = False
                        adj[v][u] = False
            # Try removing a random edge
            u, v = random.sample(range(n), 2)
            if adj[u][v]:
                adj[u][v] = False
                adj[v][u] = False
                if is_k4_free_fast(adj, n):
                    mtf_new = _max_triangle_free_induced(adj, n, 2, _START_TIME)
                    if mtf_new >= best_score:
                        adj[u][v] = True
                        adj[v][u] = True
        return adj
    
    # Run random search with local search
    time_budget = 60
    start_search = time.time()
    
    for iteration in range(10000):
        if time.time() - start_search > time_budget:
            break
        
        adj_rand = random_k4_free_graph(n)
        
        if not is_k4_free_fast(adj_rand, n):
            continue
        
        mtf_rand = _max_triangle_free_induced(adj_rand, n, 3, _START_TIME)
        if mtf_rand < best_score:
            best_score = mtf_rand
            best_edges = adj_to_edges(adj_rand, n)
            if best_score <= 10:
                break
        
        # Local search from this point
        if time.time() - start_search < time_budget - 10:
            adj_ls = local_search(adj_rand, n, max_iter=20)
            if is_k4_free_fast(adj_ls, n):
                mtf_ls = _max_triangle_free_induced(adj_ls, n, 3, _START_TIME)
                if mtf_ls < best_score:
                    best_score = mtf_ls
                    best_edges = adj_to_edges(adj_ls, n)
                    if best_score <= 10:
                        break
    
    if best_edges is None:
        best_edges = edges
    
    return best_edges
# EVOLVE-BLOCK-END
