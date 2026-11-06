"""K3,3-free dense graph on 20 vertices via bitmask greedy + local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct():
    n = 20
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # adj[u] is a bitmask of neighbors of u
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
        
        bit_u = 1 << u
        bit_v = 1 << v
        
        # Case 1: u, v on opposite sides
        # A = {u, x, y}, B = {v, p, q}
        # x, y in N(v) \ {u}
        # p, q in N(u) \ {v}
        # p, q in N(x) \cap N(y)
        # So | N(u) \cap N(x) \cap N(y) | >= 2
        
        nv = adj[v] & ~bit_u
        # Get list of neighbors for iteration
        nv_list = []
        mask = nv
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            nv_list.append(idx)
            mask ^= lsb
            
        for i in range(len(nv_list)):
            x = nv_list[i]
            for j in range(i+1, len(nv_list)):
                y = nv_list[j]
                # Check if there are at least 2 common neighbors of x and y in N(u)
                common = adj[x] & adj[y] & adj[u]
                if common.bit_count() >= 2:
                    return False
                    
        # Case 2: u, v on same side
        # A = {u, v, z}, B = {p, q, r}
        # p, q, r in N(u) \cap N(v) \cap N(z)
        # So | N(u) \cap N(v) \cap N(z) | >= 3
        
        common_uv = adj[u] & adj[v]
        for z in range(n):
            if z == u or z == v:
                continue
            common = common_uv & adj[z]
            if common.bit_count() >= 3:
                return False
                
        return True

    def rebuild_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask ^= lsb
        return edges

    def greedy_construct(order=None):
        adj = [0] * n
        if order is None:
            order = [(u, v) for u in range(n) for v in range(u+1, n)]
        
        for u, v in order:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return adj

    def local_search(adj):
        nonlocal deadline
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add edges
            for u in range(n):
                if time.time() >= deadline - 5:
                    break
                for v in range(u+1, n):
                    if not ((adj[u] >> v) & 1) and can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
                        break
                if improved:
                    break
            
            # If no edge could be added, try removing an edge and see if we can add more
            if not improved and time.time() < deadline - 10:
                edges = rebuild_edges(adj)
                if not edges:
                    break
                
                # Try up to 10 random removals
                for _ in range(10):
                    if time.time() >= deadline - 10:
                        break
                    u, v = random.choice(edges)
                    adj_copy = adj[:]
                    
                    # Remove edge
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    # Try to add edges again
                    imp = True
                    while imp and time.time() < deadline - 10:
                        imp = False
                        for u2 in range(n):
                            if time.time() >= deadline - 10:
                                break
                            for v2 in range(u2+1, n):
                                if not ((adj[u2] >> v2) & 1) and can_add_edge(adj, u2, v2):
                                    adj[u2] |= (1 << v2)
                                    adj[v2] |= (1 << u2)
                                    imp = True
                                    break
                            if imp:
                                break
                    
                    new_edges = rebuild_edges(adj)
                    old_edges = rebuild_edges(adj_copy)
                    
                    if len(new_edges) > len(old_edges):
                        improved = True
                    else:
                        adj = adj_copy
                        break
        
        return adj

    # Run greedy construction with multiple random orderings
    best_adj = None
    best_count = 0
    
    # Deterministic greedy first
    adj = greedy_construct()
    adj = local_search(adj)
    count = sum(bin(x).count("1") for x in adj) // 2
    if count > best_count:
        best_count = count
        best_adj = adj[:]
        
    # Random restarts
    random.seed(42)
    for restart in range(100):
        if time.time() >= deadline - 10:
            break
            
        # Random edge ordering
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        adj = greedy_construct(all_edges)
        adj = local_search(adj)
        
        count = sum(bin(x).count("1") for x in adj) // 2
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            
    return rebuild_edges(best_adj)
# EVOLVE-BLOCK-END
