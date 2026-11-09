import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    n = 20
    
    def time_left():
        return deadline - time.time()
    
    # Bitmask adjacency
    # adj[i] is a bitmask of neighbors of i
    
    def edge_count(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    def find_k33_violation(adj):
        """Find a triple with >=3 common neighbors. Return (a,b,c, common_mask) or None."""
        for a in range(n):
            for b in range(a+1, n):
                ab = adj[a] & adj[b]
                for c in range(b+1, n):
                    common = ab & adj[c]
                    if bin(common).count('1') >= 3:
                        return (a, b, c, common)
        return None
    
    def is_k33_free(adj):
        return find_k33_violation(adj) is None
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def would_create_k33(adj, u, v):
        """Check if adding edge (u,v) creates a K3,3."""
        # The new K3,3 must use edge (u,v).
        # One side has u, other has v.
        # Side A = {u, x, y}, Side B = {v, p, q}
        # x,y in N(v) (original), p,q in N(u) (original)
        # Need x-p, x-q, y-p, y-q all in adj (original)
        
        nv = [w for w in range(n) if w != u and w != v and (adj[v] >> w) & 1]
        nu = [w for w in range(n) if w != u and w != v and (adj[u] >> w) & 1]
        
        if len(nv) < 2 or len(nu) < 2:
            return False
        
        # For each pair (x,y) in nv, check if they share >=2 neighbors in nu
        for i in range(len(nv)):
            for j in range(i+1, len(nv)):
                x, y = nv[i], nv[j]
                common_in_nu = adj[x] & adj[y] & adj[u]  # neighbors of both x and y that are also neighbors of u
                # Actually we need common neighbors of x and y that are in nu
                common_xy = adj[x] & adj[y]
                common_in_nu = common_xy & adj[u]  # but adj[u] includes v, and we excluded v
                # Filter: we want neighbors in nu
                count = 0
                for w in nu:
                    if (common_xy >> w) & 1:
                        count += 1
                        if count >= 2:
                            return True
        return False
    
    # Phase 1: Initialize with random graph
    best_adj = None
    best_count = 0
    
    random.seed(42)
    
    for attempt in range(50):
        if time_left() < 20:
            break
        
        # Random initialization with varying p
        p = 0.6 + 0.3 * (attempt / 50)
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair: remove edges to fix K3,3
        max_repair = 200
        for _ in range(max_repair):
            if time_left() < 15:
                break
            viol = find_k33_violation(adj)
            if viol is None:
                break
            a, b, c, common = viol
            # Find an edge to remove. Remove an edge incident to one of a,b,c that's in the common neighborhood.
            # Actually, remove an edge that's part of the K3,3. The K3,3 has vertices {a,b,c} and 3 from common.
            # Pick 3 from common, then remove one of the 9 cross-edges.
            common_list = [w for w in range(n) if (common >> w) & 1]
            # Remove edge (a, common_list[0])
            remove_edge(adj, a, common_list[0])
        
        # Improve: add edges
        improved = True
        while improved:
            if time_left() < 10:
                break
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj[i] >> j) & 1:
                        if not would_create_k33(adj, i, j):
                            add_edge(adj, i, j)
                            improved = True
        
        # Local search: edge swaps
        for _ in range(1000):
            if time_left() < 5:
                break
            # Try to add a random non-edge
            i, j = random.sample(range(n), 2)
            if not (adj[i] >> j) & 1:
                if not would_create_k33(adj, i, j):
                    add_edge(adj, i, j)
        
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_adj = adj[:]
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
