import random
import time
import math

def construct():
    n = 20
    # adj[u] is a bitmask of neighbors of u
    adj = [0] * n
    
    def popcount(x):
        return bin(x).count('1')
    
    def is_valid_addition(u, v):
        # Check if adding edge (u, v) violates the constraint
        # Constraint: For all triples {a,b,c}, |N(a) & N(b) & N(c)| <= 2
        # Adding (u,v) updates N(u) and N(v).
        # Only triples containing u or v are affected.
        
        # Triples {u, w, z} where w, z != u, v
        # New common neighbors for {u, w, z} = (N(u) | (1<<v)) & N(w) & N(z)
        # Old common neighbors = N(u) & N(w) & N(z)
        # The new common neighbors are Old UNION ( {v} & N(w) & N(z) )
        # So the count increases by 1 if v is in N(w) and N(z).
        # We need to ensure that if v is in N(w) & N(z), then the old count was < 2.
        # i.e., if v in N(w) & N(z), then popcount(N(u) & N(w) & N(z)) must be <= 1.
        
        # Similarly for triples {v, w, z}
        # If u in N(w) & N(z), then popcount(N(v) & N(w) & N(z)) must be <= 1.
        
        # Iterate over all pairs w, z
        for w in range(n):
            if w == u or w == v:
                continue
            for z in range(w + 1, n):
                if z == u or z == v:
                    continue
                
                # Check triple {u, w, z}
                if (adj[w] >> v) & 1 and (adj[z] >> v) & 1:
                    # v is a neighbor of both w and z
                    # Check if u already has 2 common neighbors with w and z
                    common = adj[u] & adj[w] & adj[z]
                    if popcount(common) >= 2:
                        return False
                
                # Check triple {v, w, z}
                if (adj[w] >> u) & 1 and (adj[z] >> u) & 1:
                    # u is a neighbor of both w and z
                    common = adj[v] & adj[w] & adj[z]
                    if popcount(common) >= 2:
                        return False
        
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return 1

    def edge_count():
        count = 0
        for u in range(n):
            count += popcount(adj[u])
        return count // 2

    best_edges = []
    best_count = 0
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Multiple random restarts
    restarts = 0
    while time.time() < deadline:
        random.shuffle(all_edges)
        for u, v in all_edges:
            if is_valid_addition(u, v):
                add_edge(u, v)
        
        current_count = edge_count()
        if current_count > best_count:
            best_count = current_count
            # Save current adjacency
            best_adj = adj[:]
        
        # Reset adj for next restart
        adj = [0] * n
        restarts += 1
        # If we have enough time, do more restarts
        # Each restart is O(E * N^2) ~ 200 * 400 = 80k ops. Very fast.
        # We can do many restarts.
        
    # Reconstruct edge list from best_adj
    if best_count > 0:
        for u in range(n):
            for v in range(u + 1, n):
                if (best_adj[u] >> v) & 1:
                    best_edges.append([u, v])
    
    # Fallback to seed if something went wrong or no edges found (unlikely)
    if not best_edges:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_edges
