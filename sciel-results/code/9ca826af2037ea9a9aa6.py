import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin
    
    n = 40
    all_mask = (1 << n) - 1
    
    def build_k5_base():
        """Build 8 disjoint K5s."""
        adj = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    adj[base + a] |= 1 << (base + b)
                    adj[base + b] |= 1 << (base + a)
        return adj
    
    def has_c6_with_edge(adj, u, v):
        """Check if adding edge (u,v) creates a C6 using bitmasks.
        A C6 through (u,v) means path u-a-b-c-d-v of length 5 with distinct vertices."""
        # S1: neighbors of u excluding v
        S1 = adj[u] & ~(1 << v)
        if S1 == 0:
            return False
        
        # S2: neighbors of vertices in S1, excluding u and v
        S2 = 0
        for x in range(n):
            if S1 & (1 << x):
                S2 |= adj[x]
        S2 &= ~(1 << u) & ~(1 << v)
        S2 &= S1  # No, S2 should exclude S1? No, S2 can include S1 vertices? 
        # Actually S2 = neighbors of S1, we just need to exclude u and v
        # But we also need S2 vertices to be distinct from S1 for the path to be simple
        # Wait, no - in the path u-a-b-c-d-v, a is in S1, b is neighbor of a (so in S2)
        # b must be distinct from u, v, a. So S2 should exclude u, v, and S1.
        S2 &= ~S1
        if S2 == 0:
            return False
        
        # S3: neighbors of vertices in S2, excluding u, v, S1, S2
        S3 = 0
        for x in range(n):
            if S2 & (1 << x):
                S3 |= adj[x]
        S3 &= ~(1 << u) & ~(1 << v) & ~S1 & ~S2
        if S3 == 0:
            return False
        
        # S4: neighbors of vertices in S3, excluding u, v, S1, S2, S3
        S4 = 0
        for x in range(n):
            if S3 & (1 << x):
                S4 |= adj[x]
        S4 &= ~(1 << u) & ~(1 << v) & ~S1 & ~S2 & ~S3
        if S4 == 0:
            return False
        
        # Check if S4 intersects neighbors of v
        # But we need d in S4 such that v is neighbor of d
        # v's neighbors are adj[v], so check if S4 & adj[v] != 0
        # But we need to exclude u from adj[v] consideration? No, u is already excluded from S4
        return (S4 & adj[v]) != 0
    
    def greedy_add(adj, non_edges, rng, time_limit):
        """Add edges greedily from a shuffled list of non-edges."""
        for u, v in non_edges:
            if time.time() > time_limit:
                break
            if (adj[u] >> v) & 1:
                continue
            if has_c6_with_edge(adj, u, v):
                continue
            adj[u] |= 1 << v
            adj[v] |= 1 << u
    
    def count_edges(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2
    
    def verify_c6_free(adj):
        """Verify no C6 exists in the graph."""
        for v0 in range(n):
            for v1 in range(n):
                if v1 <= v0:
                    continue
                if not (adj[v0] >> v1) & 1:
                    continue
                for v2 in range(n):
                    if v2 in (v0, v1):
                        continue
                    if not (adj[v1] >> v2) & 1:
                        continue
                    for v3 in range(n):
                        if v3 in (v0, v1, v2):
                            continue
                        if not (adj[v2] >> v3) & 1:
                            continue
                        for v4 in range(n):
                            if v4 in (v0, v1, v2, v3):
                                continue
                            if not (adj[v3] >> v4) & 1:
                                continue
                            for v5 in range(n):
                                if v5 in (v0, v1, v2, v3, v4):
                                    continue
                                if not (adj[v4] >> v5) & 1:
                                    continue
                                if (adj[v5] >> v0) & 1:
                                    return False
        return True
    
    # Start with K5 base
    base_adj = build_k5_base()
    best_adj = base_adj[:]
    best_count = count_edges(best_adj)
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Multiple random restarts
    for seed in range(500):
        if time.time() > deadline - 10:
            break
        
        adj = base_adj[:]
        rng = random.Random(seed)
        
        # Get non-edges
        non_edges = [(i, j) for i, j in all_pairs if not (adj[i] >> j) & 1]
        rng.shuffle(non_edges)
        
        # Greedy add
        greedy_add(adj, non_edges, rng, deadline - 5)
        
        # Local search: try edge swaps
        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Try to add more edges
                non_edges2 = [(i, j) for i, j in all_pairs if not (adj[i] >> j) & 1]
                rng.shuffle(non_edges2)
                for u, v in non_edges2:
                    if time.time() > deadline - 5:
                        break
                    if (adj[u] >> v) & 1:
                        continue
                    if not has_c6_with_edge(adj, u, v):
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                        improved = True
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
    
    # Final verification
    if verify_c6_free(best_adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (best_adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    # Fallback: return base
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
