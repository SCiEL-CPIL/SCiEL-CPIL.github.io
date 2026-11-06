import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin before 150s limit
    
    n = 40
    
    def init_adj():
        """Create initial graph: 8 disjoint K5s"""
        adj = np.zeros((n, n), dtype=np.int8)
        for k in range(8):
            base = k * 5
            for i in range(5):
                for j in range(i + 1, 5):
                    u = base + i
                    v = base + j
                    adj[u, v] = 1
                    adj[v, u] = 1
        return adj
    
    def count_edges(adj):
        return int(np.sum(adj) // 2)
    
    def edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return edges
    
    def would_create_c6(adj, u, v):
        """Check if adding edge (u,v) creates a C6 using bitset-based approach.
        A C6 through (u,v) means there's a path of length 5 from u to v not using edge (u,v).
        Equivalently, there exist neighbors a of u and b of v such that there's a path of length 3 from a to b.
        """
        # Get neighbors of u (excluding v since we're checking before adding)
        neigh_u = np.where(adj[u] == 1)[0]
        neigh_v = np.where(adj[v] == 1)[0]
        
        # For each pair (a in neigh_u, b in neigh_v), check if dist(a,b) <= 3
        # But we need exactly a path of length 3 that doesn't use u or v
        # Actually, a C6 is u-a-x-y-b-v where a~u, b~v, and a-x-y-b is a path of length 3
        # So we need: exists a in N(u), b in N(v) such that there's a path a-x-y-b of length 3
        # This means dist(a,b) = 3 in the graph without the edge (u,v)
        
        # More efficiently: check if there's a common neighbor structure
        # Path of length 3 from a to b: a-x-y-b where x~a, y~b, x~y
        # So for each a in N(u) and b in N(v), check if N(a) ∩ N(b) has a vertex y such that y has a neighbor x that is also adjacent to a... 
        # Wait, let me think more carefully.
        
        # A 6-cycle u-a-x-y-b-v-u means:
        # u~a, a~x, x~y, y~b, b~v, v~u
        # So we need: exists a in N(u), b in N(v), x in N(a), y in N(b) such that x~y
        # And x != u, x != v, y != u, y != v, a != b, a != y, b != x, x != y (for simple cycle)
        
        # Efficient check: For each a in N(u) and b in N(v):
        #   Check if there exists x in N(a) \ {u} and y in N(b) \ {v} such that x~y
        #   This is equivalent to: (N(a) \ {u,v}) intersects (N(N(b) \ {u,v})) in some way...
        
        # Simpler: precompute for each pair (a,b) whether there's a length-3 path
        # But let's do it directly with bitsets
        
        # Convert neighbor sets to bitsets for fast intersection
        # Use Python integers as bitsets for 40 vertices
        
        def to_bitset(arr):
            bs = 0
            for x in arr:
                bs |= (1 << x)
            return bs
        
        neigh_u_bits = to_bitset(neigh_u)
        neigh_v_bits = to_bitset(neigh_v)
        
        # Precompute neighbor bitsets for all vertices
        neigh_bits = []
        for i in range(n):
            neigh_bits.append(to_bitset(np.where(adj[i] == 1)[0]))
        
        # For each a in N(u), b in N(v):
        # Check if there's a length-3 path from a to b
        # Length-3 path: a-x-y-b means x in N(a), y in N(b), x~y
        # So: exists x in N(a) such that x ~ some y in N(b)
        # Equivalently: N(a) intersects N(N(b))
        
        # N(N(b)) = union of N(y) for y in N(b)
        # Let's compute this on the fly
        
        for a in neigh_u:
            if a == v:
                continue
            for b in neigh_v:
                if b == u or b == a:
                    continue
                # Check if there's a length-3 path from a to b
                # x in N(a), y in N(b), x~y
                # So: exists x in N(a) such that x is adjacent to some y in N(b)
                # i.e., N(a) intersects (union of N(y) for y in N(b))
                
                # Compute union of N(y) for y in N(b)
                union_Nb = 0
                for y in np.where(adj[b] == 1)[0]:
                    if y != a and y != u and y != v:
                        union_Nb |= neigh_bits[y]
                
                # Check if N(a) intersects union_Nb
                if (neigh_bits[a] & union_Nb) != 0:
                    return True
        
        return False
    
    def greedy_fill(adj, time_limit):
        """Greedily add edges that don't create C6"""
        while True:
            if time.time() > time_limit:
                break
            
            # Find all safe edges
            safe = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v] == 0:
                        if not would_create_c6(adj, u, v):
                            safe.append((u, v))
            
            if not safe:
                break
            
            # Pick a random safe edge
            u, v = random.choice(safe)
            adj[u, v] = 1
            adj[v, u] = 1
        
        return adj
    
    def local_search(adj, time_limit):
        """Try to improve by swapping edges or adding/removing to increase edge count"""
        best_adj = adj.copy()
        best_count = count_edges(adj)
        
        while time.time() < time_limit:
            improved = False
            
            # Try adding any safe edge
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v] == 0:
                        if not would_create_c6(adj, u, v):
                            adj[u, v] = 1
                            adj[v, u] = 1
                            improved = True
            
            # If no improvement, try 2-opt style: remove one edge, add two
            if not improved:
                # Try removing a random edge and see if we can add more
                current_edges = edges_from_adj(adj)
                if current_edges:
                    # Remove a random edge
                    rem_idx = random.randint(0, len(current_edges) - 1)
                    u, v = current_edges[rem_idx]
                    adj[u, v] = 0
                    adj[v, u] = 0
                    
                    # Try to add edges
                    added = 0
                    for u2 in range(n):
                        for v2 in range(u2 + 1, n):
                            if adj[u2, v2] == 0:
                                if not would_create_c6(adj, u2, v2):
                                    adj[u2, v2] = 1
                                    adj[v2, u2] = 1
                                    added += 1
                    
                    new_count = count_edges(adj)
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = adj.copy()
                        improved = True
                    else:
                        # Revert
                        adj = best_adj.copy()
                        if added <= 1:  # Net didn't help, revert the removal
                            pass
                        # Actually we already copied best_adj, so adj is back to best
                        # But we need to be careful - let's just continue
            
            if not improved:
                # Try random restart within local search
                # Perturb: remove a few random edges
                for _ in range(3):
                    u, v = random.randint(0, n-1), random.randint(0, n-1)
                    if u != v and adj[u, v]:
                        adj[u, v] = 0
                        adj[v, u] = 0
                
                # Refill
                adj = greedy_fill(adj, min(time.time() + 5, time_limit))
                new_count = count_edges(adj)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = adj.copy()
        
        return best_adj, best_count
    
    best_edges = []
    best_count = 0
    
    # Phase 1: Multiple greedy restarts
    for restart in range(200):
        if time.time() > deadline - 30:  # Leave 30s for local search
            break
        
        adj = init_adj()
        adj = greedy_fill(adj, deadline - 30)
        count = count_edges(adj)
        
        if count > best_count:
            best_count = count
            best_edges = edges_from_adj(adj)
    
    # Phase 2: Local search on best solution
    if best_count > 0 and time.time() < deadline - 5:
        adj = init_adj()
        # Reconstruct best_adj from best_edges
        for edge in best_edges:
            adj[edge[0], edge[1]] = 1
            adj[edge[1], edge[0]] = 1
        
        best_adj, best_count = local_search(adj, deadline - 5)
        best_edges = edges_from_adj(best_adj)
    
    # Final: make sure we return valid edges
    return best_edges
