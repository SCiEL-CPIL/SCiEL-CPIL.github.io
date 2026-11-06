import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    n = 40
    
    def time_ok():
        return time.time() < deadline
    
    # Bitset-based adjacency: adj[u] is a bitmask of neighbors of u
    # adj_mask[u] = bitmask where bit v is set if u-v is an edge
    
    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def has_c6(adj):
        """Check if graph has any C6 using bitset operations."""
        # A C6 is a cycle v0-v1-v2-v3-v4-v5-v0
        # For each pair (u,w) at distance 2, check if there's a path of length 3 between them
        # that doesn't go through the intermediate vertex
        # Actually, let's use a more direct approach:
        # For each edge (u,v), check if there's a path of length 5 from u to v not using edge (u,v)
        # This is expensive. Let's use the standard approach:
        # A C6 exists iff there exist vertices a,b,c,d,e,f such that
        # a-b-c-d-e-f-a are all edges and all distinct.
        
        # Efficient check: for each pair of vertices (i,j) with i<j,
        # count the number of common neighbors at distance 2.
        # Actually, the standard efficient method:
        # For each ordered pair (u,v) that is an edge,
        # compute the set of vertices reachable from u in exactly 2 steps (not through v),
        # and the set reachable from v in exactly 2 steps (not through u),
        # and check if they intersect at a vertex w != u,v.
        # Then for each such w, check if there's a path of length 2 from w back... 
        
        # Let me use a cleaner approach:
        # C6 exists iff there exist 6 distinct vertices forming a cycle.
        # For each vertex u, compute its neighbors N(u).
        # For each pair of neighbors (v,w) of u, the edge v-w would create a triangle (C3), not C6.
        # For C6: u-v-x-y-z-w-u where all are distinct.
        
        # More practical: iterate over all pairs of vertices (a,b) that are NOT adjacent.
        # If they have 2 or more common neighbors, and those common neighbors have...
        # This is getting complex. Let me use a direct but optimized check.
        
        # For each vertex u (0 to n-1):
        #   For each neighbor v of u:
        #     For each neighbor w of v (w != u):
        #       For each neighbor x of w (x != v, x != u):
        #         For each neighbor y of x (y != w, y != u, y != v):
        #           For each neighbor z of y (z != x, z != u, z != v, z != w):
        #             If z == u, we found a C6 (u-v-w-x-y-z-u)
        # But this is O(n * d^5) which could be slow.
        
        # Better: use the fact that in a C6, opposite vertices are at distance 3.
        # For each pair of vertices (a,c) at distance exactly 3,
        # check if there are two distinct paths of length 3 between them.
        
        # Let me just do a direct search with pruning.
        # For each starting vertex u:
        #   neighbors of u: N1
        #   for each v in N1:
        #     neighbors of v (excluding u): N2
        #     for each w in N2:
        #       neighbors of w (excluding u,v): N3
        #       for each x in N3:
        #         neighbors of x (excluding u,v,w): N4
        #         for each y in N4:
        #           neighbors of y (excluding u,v,w,x): N5
        #           if u in N5: found C6
        #           if z in N5 for z in {u,v,w,x}: found C6 (but need to check distinctness)
        
        # This is still expensive. Let me use a bitset approach.
        # For each pair (u,v), compute the set of common "distance-2" neighbors.
        
        # Actually, the most efficient known method for small n:
        # For each vertex u, let N(u) be its neighbor set.
        # For each ordered pair (v,w) with v in N(u), w in N(u), v != w:
        #   This gives a path u-v-w (length 2 from u to w through v)
        # For C6: we need u-v1-v2-v3-v4-v5-u
        # So v5 in N(u), v1 in N(u), v1 != v5
        # v2 in N(v1) \ {u}, v4 in N(v5) \ {u}
        # v3 in N(v2) \ {u,v1}, v3 in N(v4) \ {u,v5}
        # And v2 != v4, v3 not in {u,v1,v2,v4,v5}
        
        # Let me just implement a straightforward check and hope it's fast enough for n=40.
        
        # For n=40, the number of potential C6s is at most C(40,6) * 5! / 2 which is huge.
        # But we can prune heavily.
        
        # Let me use this approach:
        # For each pair of non-adjacent vertices (a, c):
        #   Let S = common neighbors of a and c (these would be b and d in a-b-c-d-e-f-a... no)
        #   Actually in C6 a-b-c-d-e-f-a, a and c are at distance 2 (through b) and distance 4 (through f,e,d).
        #   a and c are NOT adjacent in a C6.
        #   Common neighbors of a and c: b is a common neighbor (a-b-c). Any others?
        #   If a and c have 2+ common neighbors, say b1 and b2, then a-b1-c-b2-a is a C4, not C6.
        
        # Let me just do the direct enumeration with bitsets.
        
        # For each vertex u:
        #   For each v in N(u):
        #     For each w in N(v) \ {u}:
        #       For each x in N(w) \ {u, v}:
        #         # Now we have path u-v-w-x of length 3
        #         # Need y in N(x) \ {u,v,w} and z in N(y) \ {u,v,w,x} with z in N(u)
        #         # And y != u, v, w, x; z != u, v, w, x, y; z in N(u)
        #         # Also need the 6 vertices to be distinct: u,v,w,x,y,z all distinct
        #         # Since we already excluded u,v,w from N(x), x is distinct from u,v,w.
        #         # y is from N(x) excluding u,v,w,x, so y distinct from u,v,w,x.
        #         # z from N(y) excluding u,v,w,x,y, and z in N(u).
        #         # Need z != y (ensured by exclusion), z distinct from u,v,w,x (ensured).
        #         # So just check if there exists y in N(x)\{u,v,w,x} such that
        #         #   N(y) \ {u,v,w,x,y} intersects N(u)
        #         #   AND the resulting z != y (already ensured)
        #         #   AND y != z (ensured by z not in {y})
        #         #   AND all 6 distinct: u,v,w,x,y,z - we've ensured y and z are distinct from u,v,w,x and each other.
        #         # Wait, we also need y != z which is ensured, and we need z in N(u).
        #         # Also need that the edge z-u exists (z in N(u)) and edge y-z exists (z in N(y)).
        #         # And we need y in N(x) (ensured) and x in N(w) (ensured) etc.
        #         
        #         # But we also need to make sure we're not double-counting or creating shorter cycles.
        #         # If any of u,v,w,x,y,z are not all distinct, we don't have a C6.
        #         # We've ensured:
        #         #   v in N(u), v != u (implicit)
        #         #   w in N(v), w != u, w != v (v is in N(v)? no, simple graph)
        #         #   x in N(w), x != u, x != v, x != w
        #         #   y in N(x), y != u, y != v, y != w, y != x
        #         #   z in N(y), z != u, z != v, z != w, z != x, z != y, z in N(u)
        #         # So all 6 are distinct. 
        #         
        #         # But wait: we also need to ensure that the "other" edges don't create a shorter cycle.
        #         # Actually no - if all 6 vertices are distinct and the 6 edges exist, it's a C6.
        #         # It might also contain chords, but the cycle itself is still a C6.
        #         # Actually, a C6 in a graph means a 6-cycle, which is a cycle of length 6.
        #         # If there are chords, it's still a C6 (the cycle exists).
        #         # Wait, no. A "six-cycle" or C6 typically means an induced 6-cycle? 
        #         # No, in extremal graph theory, ex(n, C6) means no cycle of length 6, 
        #         # which means no 6 vertices forming a cycle (possibly with chords).
        #         # Actually, I need to be careful. ex(n, C_k) usually means no cycle of length k,
        #         # where a cycle of length k is k distinct vertices v1,...,vk with edges vi-v(i+1) and vk-v1.
        #         # Chords are allowed in the graph but the cycle itself is just those k edges.
        #         # So yes, if all 6 vertices are distinct and the 6 cycle edges exist, it's a C6.
        
        # Let me implement this.
        
        for u in range(n):
            nu = adj[u]
            if nu == 0:
                continue
            # For each v in N(u)
            v_bits = nu
            while v_bits:
                v_bit = v_bits & (-v_bits)
                v = v_bit.bit_length() - 1
                v_bits ^= v_bit
                nv = adj[v] & ~v_bit & ~(1 << u)  # neighbors of v, excluding u and v itself
                if nv == 0:
                    continue
                w_bits = nv
                while w_bits:
                    w_bit = w_bits & (-w_bits)
                    w = w_bit.bit_length() - 1
                    w_bits ^= w_bit
                    nw = adj[w] & ~v_bit & ~(1 << u) & ~w_bit  # neighbors of w, excluding u, v, w
                    if nw == 0:
                        continue
                    x_bits = nw
                    while x_bits:
                        x_bit = x_bits & (-x_bits)
                        x = x_bit.bit_length() - 1
                        x_bits ^= x_bit
                        nx = adj[x] & ~v_bit & ~(1 << u) & ~w_bit & ~x_bit  # neighbors of x, excluding u,v,w,x
                        if nx == 0:
                            continue
                        # For each y in nx, check if N(y) intersects N(u) (excluding u,v,w,x,y)
                        y_bits = nx
                        while y_bits:
                            y_bit = y_bits & (-y_bits)
                            y = y_bit.bit_length() - 1
                            y_bits ^= y_bit
                            ny = adj[y] & ~v_bit & ~(1 << u) & ~w_bit & ~x_bit & ~y_bit  # neighbors of y, excluding u,v,w,x,y
                            # Check if ny intersects nu (neighbors of u)
                            if ny & nu:
                                return True
        return False
    
    # Strategy: Start with a dense graph, remove edges to eliminate C6s, then try to add back.
    # But removing edges is hard to do optimally. 
    # Better: Build up greedily, always maintaining C6-freeness.
    
    # Greedy construction:
    # Order all possible edges, try to add each one if it doesn't create a C6.
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Use a random order with some structure
    random.seed(42)
    random.shuffle(all_edges)
    
    adj = [0] * n
    edges = []
    
    def would_create_c6(u, v, adj):
        """Check if adding edge (u,v) to the current graph creates a C6."""
        # A C6 containing edge (u,v) must be of the form u-v-x-y-z-w-u
        # where x in N(v)\{u}, y in N(x)\{u,v}, z in N(y)\{u,v,x}, w in N(z)\{u,v,x,y}, w in N(u)
        # Or u-w-z-y-x-v-u (same thing reversed)
        # So we need: exists x in N(v)\{u}, y in N(x)\{u,v}, z in N(y)\{u,v,x}, w in N(z)\{u,v,x,y} with w in N(u)
        # And all 6 vertices distinct.
        
        # Let me be more careful. The C6 is u-v-a-b-c-d-u where a,b,c,d are distinct and not u,v.
        # a in N(v) \ {u}
        # b in N(a) \ {u,v}
        # c in N(b) \ {u,v,a}
        # d in N(c) \ {u,v,a,b}
        # d in N(u)
        # And d != a, b, c (ensured by exclusions)
        # And a,b,c,d all distinct from each other and from u,v.
        
        a_bits = adj[v] & ~(1 << u)  # N(v) \ {u}
        while a_bits:
            a_bit = a_bits & (-a_bits)
            a = a_bit.bit_length() - 1
            a_bits ^= a_bit
            b_bits = adj[a] & ~(1 << u) & ~(1 << v) & ~a_bit  # N(a) \ {u,v,a}
            while b_bits:
                b_bit = b_bits & (-b_bits)
                b = b_bit.bit_length() - 1
                b_bits ^= b_bit
                c_bits = adj[b] & ~(1 << u) & ~(1 << v) & ~a_bit & ~b_bit  # N(b) \ {u,v,a,b}
                while c_bits:
                    c_bit = c_bits & (-c_bits)
                    c = c_bit.bit_length() - 1
                    c_bits ^= c_bit
                    d_bits = adj[c] & ~(1 << u) & ~(1 << v) & ~a_bit & ~b_bit & ~c_bit  # N(c) \ {u,v,a,b,c}
                    # d must be in N(u)
                    d_bits &= adj[u]
                    if d_bits:
                        return True
        return False
    
    for u, v in all_edges:
        if not time_ok():
            break
        if not would_create_c6(u, v, adj):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
    
    # Now try to add more edges via local search
    # Try all non-edges and see if any can be added
    improved = True
    while improved and time_ok():
        improved = False
        for u in range(n):
            for v in range(u+1, n):
                if time_ok() == False:
                    break
                if adj[u] & (1 << v):
                    continue  # already an edge
                if not would_create_c6(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
                    improved = True
    
    # Try multiple random orderings and keep the best
    best_edges = edges
    best_count = len(edges)
    
    for trial in range(20):
        if not time_ok():
            break
        random.seed(1000 + trial)
        random.shuffle(all_edges)
        adj = [0] * n
        edges = []
        for u, v in all_edges:
            if not time_ok():
                break
            if not would_create_c6(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        
        # Local search
        improved = True
        while improved and time_ok():
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if not time_ok():
                        break
                    if adj[u] & (1 << v):
                        continue
                    if not would_create_c6(u, v, adj):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.append([u, v])
                        improved = True
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    return best_edges
