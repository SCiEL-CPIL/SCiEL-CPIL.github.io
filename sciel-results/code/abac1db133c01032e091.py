import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145

    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)  # 31

    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q

    # Build base graph (90 edges)
    base_adj = [0] * n
    base_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                base_adj[i] |= (1 << j)
                base_adj[j] |= (1 << i)
                base_edges.append((i, j))
    
    # Function to count C4s using bitsets
    def count_c4(adj):
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                common = (adj[i] & adj[j]).bit_count()
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2

    # Evaluate cost of adding edge (u, v) to graph defined by adj
    # Returns number of new C4s
    def cost_add_edge(adj, u, v):
        # New C4s are formed by paths u-x-y-v where x != y, x != v, y != u
        # x must be neighbor of u, y must be neighbor of v
        # x != y
        # And x-y must be an edge? No, C4 is u-x-y-v-u.
        # So x-y must be an edge.
        # So we need common neighbors of x and y? No.
        # The cycle is u-x-y-v.
        # Edges: (u,x), (x,y), (y,v), (v,u).
        # (v,u) is the new edge.
        # So we need x ~ u, y ~ v, x ~ y, x != y.
        # Also x != v, y != u.
        
        cost = 0
        neighbors_u = adj[u]
        neighbors_v = adj[v]
        
        # Iterate over x in N(u)
        # For each x, iterate over y in N(v)
        # Check if x-y is an edge
        # This is O(deg(u) * deg(v))
        
        # Optimization:
        # For each x in N(u):
        #   candidates = N(v) & N(x)  # y must be neighbor of v and x
        #   Remove x from candidates? No, x != y is required.
        #   Remove u from candidates? y != u.
        #   Remove v from candidates? y != v (y is in N(v), so y != v unless self-loop, which we don't have).
        #   So count bits in (N(v) & N(x)) excluding x and u.
        
        xu = neighbors_u & ~neighbors_v # x in N(u) not in N(v)? No, x can be in N(v).
        # Actually, if x is in N(v), then x-v is an edge.
        # The cycle is u-x-v-u? That's a triangle, not C4.
        # So x must NOT be a neighbor of v?
        # Wait, C4 is u-x-y-v-u.
        # If x=v, then u-v-y-v-u? No, x is a vertex.
        # If x is neighbor of v, then edge (x,v) exists.
        # Then we have u-x-v. And we add (u,v).
        # This forms a triangle u-x-v-u.
        # Triangles are not C4s.
        # So for a C4, x must not be adjacent to v?
        # No, x is a vertex. The edges are (u,x), (x,y), (y,v), (v,u).
        # If x is adjacent to v, then (x,v) is an edge.
        # But (x,v) is not part of the C4 unless x=y? No.
        # The C4 uses edges (u,x), (x,y), (y,v), (v,u).
        # It does NOT use (x,v).
        # So it is allowed that x is adjacent to v.
        # However, if x is adjacent to v, and we have edge (x,y) and (y,v),
        # then x-y-v is a path.
        # And u-x is an edge.
        # And u-v is an edge.
        # So u-x-y-v-u is a cycle.
        # Is it a simple cycle?
        # Vertices: u, x, y, v.
        # If x=v, then u-v-y-v-u. Not simple.
        # So x != v.
        # If y=u, then u-x-u-v-u. Not simple.
        # So y != u.
        # If x=y, then u-x-x-v-u. Not simple.
        # So x != y.
        
        # So we just need x in N(u), y in N(v), x != y, x != v, y != u, and (x,y) is an edge.
        
        # Let's refine the bitset logic:
        # For each x in N(u):
        #   if x == v: continue
        #   # y must be in N(v) and N(x)
        #   # y != u, y != x
        #   candidates = N(v) & N(x)
        #   # Clear bit for u
        #   if (candidates & (1 << u)): candidates &= ~(1 << u)
        #   # Clear bit for x
        #   if (candidates & (1 << x)): candidates &= ~(1 << x)
        #   cost += candidates.bit_count()
        
        # This counts each C4 twice?
        # Cycle u-x-y-v-u.
        # When we pick x, we find y.
        # When we pick y? No, we iterate x in N(u).
        # The cycle is directed u->x->y->v->u.
        # Each C4 corresponds to 2 such directed cycles (u,x,y,v) and (u,v,y,x)?
        # No, (u,v,y,x) would be x in N(v)? No, we iterate x in N(u).
        # So for a fixed C4 {u,x,y,v}, the pairs (x,y) are (x1,y1) and (y1,x1)?
        # If x1 in N(u) and y1 in N(v), and x1-y1 is edge.
        # Then x1 is a neighbor of u.
        # Is y1 a neighbor of u? Not necessarily.
        # So each C4 is counted once for each x in {x1, y1} that is in N(u).
        # In a C4 u-x-y-v-u, the neighbors of u are x and v.
        # So only x is in N(u) (v is the other endpoint).
        # Wait, v is in N(u) because we add (u,v).
        # But we exclude x=v.
        # So only x is considered.
        # What about the other direction?
        # The C4 is undirected.
        # The vertices are u, x, y, v.
        # u is connected to x and v.
        # v is connected to y and u.
        # x is connected to u and y.
        # y is connected to v and x.
        # In our loop, we fix u and v (the new edge).
        # We look for x in N(u) and y in N(v).
        # x is connected to u. y is connected to v.
        # We require x-y edge.
        # This finds the path u-x-y-v.
        # This path is unique for the C4 given the endpoints u,v.
        # So each C4 is counted exactly once.
        
        total_cost = 0
        for x_bit in range(n):
            if not (neighbors_u & (1 << x_bit)): continue
            x = x_bit
            if x == v: continue
            # y must be in N(v) and N(x)
            candidates = neighbors_v & adj[x]
            # Exclude u and x
            mask = ~((1 << u) | (1 << x))
            candidates &= mask
            total_cost += candidates.bit_count()
            
        return total_cost

    # Initial best edge selection
    best_edge = None
    best_cost = float('inf')
    adj = base_adj[:]
    
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1: continue
            c = cost_add_edge(adj, i, j)
            if c < best_cost:
                best_cost = c
                best_edge = (i, j)
                
    if best_edge:
        u, v = best_edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    current_count = count_c4(adj)
    
    # Local Search
    # Swap (u1, v1) with (u2, v2)
    # Remove (u1, v1), Add (u2, v2)
    
    def delta_swap(adj, u1, v1, u2, v2):
        # 1. Remove (u1, v1)
        # 2. Add (u2, v2)
        # We need to compute change in C4 count.
        
        # Let's just recompute count_c4 for the new graph.
        # It's fast enough?
        # count_c4 is O(n^2 * bit_count).
        # n=31. 31*31 = 961 iterations.
        # Each iteration does a few bit ops.
        # Should be ~1 microsecond.
        # 34000 swaps * 1us = 34ms.
        # So we can just recompute.
        
        new_adj = adj[:]
        new_adj[u1] &= ~(1 << v1)
        new_adj[v1] &= ~(1 << u1)
        new_adj[u2] |= (1 << v2)
        new_adj[v2] |= (1 << u2)
        
        new_count = count_c4(new_adj)
        return new_count - current_count

    # To make it faster, we can precompute the list of edges and non-edges.
    
    improved = True
    iteration = 0
    max_iterations = 500
    
    while improved and iteration < max_iterations and time.time() < deadline - 10:
        improved = False
        iteration += 1
        
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
                    
        best_swap = None
        best_delta = 0
        
        for e1 in edges:
            u1, v1 = e1
            for e2 in non_edges:
                u2, v2 = e2
                d = delta_swap(adj, u1, v1, u2, v2)
                if d < best_delta:
                    best_delta = d
                    best_swap = (e1, e2)
                    
        if best_swap:
            e1, e2 = best_swap
            u1, v1 = e1
            u2, v2 = e2
            
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            
            current_count += best_delta
            improved = True
        else:
            break

    # Convert back to list of edges
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                final_edges.append([i, j])
                
    return final_edges
