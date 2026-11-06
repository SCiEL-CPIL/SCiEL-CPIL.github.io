import time
import numpy as np
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    n = 31
    target_edges = 91
    q = 5
    
    # 1. Construct the polarity graph of PG(2,5)
    # Points: (1, x, y) for x,y in 0..4 (25 points)
    #          (0, 1, z) for z in 0..4 (5 points)
    #          (0, 0, 1) (1 point)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    adj = np.zeros((n, n), dtype=np.int8)
    edges = []
    
    # Orthogonality: x1*x2 + y1*y2 - z1*z2 = 0 mod 5
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            if (p1[0]*p2[0] + p1[1]*p2[1] - p1[2]*p2[2]) % q == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append((i, j))
                
    current_edges = len(edges)
    
    # The polarity graph of PG(2,5) has 93 edges.
    # We need 91 edges.
    # First, reduce to 90 edges to minimize C4s, then add 1.
    # Or just reduce to 91 directly?
    # Reducing to 90 (C4-free if possible) is better.
    
    def count_c4s(adj):
        # Number of C4s is sum over all pairs of vertices (u,v) of C(common_neighbors(u,v), 2)
        # Or sum over all u,v of (A^2)[u,v] * (A^2)[u,v] / 2 ? No.
        # Common neighbors of u and v is (A^2)[u,v].
        # Each C4 is counted 4 times (once for each pair of opposite vertices? No, once for each pair of diagonals).
        # A C4 has 2 pairs of opposite vertices.
        # So sum over all u<v of C(common_neighbors(u,v), 2) counts each C4 exactly once?
        # Let C4 be u-v-w-x-u.
        # Pairs of opposite vertices: (u,w) and (v,x).
        # common_neighbors(u,w) includes v and x. C(2,2)=1.
        # common_neighbors(v,x) includes u and w. C(2,2)=1.
        # Total sum = 2.
        # So count = sum_{u<v} C(common_neighbors(u,v), 2) / 2.
        
        A2 = np.dot(adj, adj)
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                cn = A2[u, v]
                total += cn * (cn - 1) // 2
        return total // 2

    def get_c4_participation(adj):
        # For each edge, how many C4s does it belong to?
        # An edge (u,v) is in a C4 if there exists w such that u-w-v is a path? No.
        # A C4 containing (u,v) has the form u-v-x-y-u.
        # This requires common neighbors of u and x? No.
        # It requires a path of length 3 from u to v that does not use edge (u,v).
        # Number of such paths = common_neighbors(u, v) is not enough.
        # We need pairs (x, y) such that u-x, x-y, y-v are edges.
        # This is equivalent to the number of length-3 paths between u and v.
        # (A^3)[u, v] counts length-3 walks.
        # Since the graph is simple, length-3 walks are either:
        # 1. u-x-y-v (distinct) -> part of a C4 or P4.
        # 2. u-x-u-v (backtrack) -> x is neighbor of u.
        # 3. u-v-x-v (backtrack) -> x is neighbor of v.
        # 4. u-x-y-x (backtrack) -> not ending at v unless v=x.
        
        # Actually, simpler:
        # A C4 containing edge (u,v) is determined by a pair of vertices (x,y) 
        # such that x is neighbor of u, y is neighbor of v, and x-y is an edge.
        # And x != v, y != u.
        # Let N(u) be neighbors of u.
        # Let N(v) be neighbors of v.
        # We want edges between N(u)\{v} and N(v)\{u}.
        # Let M be the adjacency matrix.
        # Count = sum_{x in N(u)\{v}} sum_{y in N(v)\{u}} M[x,y]
        
        participations = np.zeros((n, n), dtype=np.int32)
        for u in range(n):
            for v in range(n):
                if adj[u, v] == 1:
                    count = 0
                    # Vectorized count of edges between N(u) and N(v)
                    # Mask out u and v themselves? 
                    # adj[u] is 1 for neighbors. adj[u, v] is 1.
                    # We want edges x-y where adj[u,x]=1 and adj[v,y]=1.
                    # This is sum(adj[u, :] * adj[v, :] * adj)
                    # This includes x=u? adj[u,u]=0.
                    # Includes x=v? adj[u,v]=1. If y is neighbor of v, adj[v,y]=1.
                    # Term x=v: adj[u,v]*adj[v,v]*adj[v,y] = 0.
                    # So it correctly counts edges between N(u) and N(v).
                    # Does it count the edge (u,v) itself?
                    # x=u, y=v: adj[u,u]*... = 0.
                    # x=v, y=u: adj[u,v]*adj[v,u]*adj[v,u] = 1*1*1 = 1.
                    # Wait, if x=v and y=u, is (v,u) an edge? Yes.
                    # But (v,u) is the edge (u,v) itself.
                    # A C4 cannot contain the same edge twice.
                    # The path u-v-u-v is not a simple cycle.
                    # So we must subtract the contribution of the edge (u,v) itself if it's counted.
                    # In the sum sum_{x,y} adj[u,x]adj[v,y]adj[x,y]:
                    # If x=v, y=u: adj[u,v]=1, adj[v,u]=1, adj[v,u]=1. Product 1.
                    # This corresponds to the "path" u-v-u. Length 2? No, u-v is edge, v-u is edge.
                    # This is a walk of length 2? No, the formula counts length 3 walks?
                    # Let's re-verify.
                    # We want paths u-x-y-v.
                    # x is neighbor of u. y is neighbor of v. x-y is edge.
                    # If x=v, then u-v is edge. y is neighbor of v. v-y is edge.
                    # Path u-v-y-v. This ends at v.
                    # This is a walk u-v-y-v.
                    # It uses edge (u,v) and (v,y) and (y,v).
                    # This is not a simple cycle.
                    # So we should exclude x=v and y=u.
                    
                    # Let's just iterate to be safe and fast enough for n=31.
                    nu = np.where(adj[u] == 1)[0]
                    nv = np.where(adj[v] == 1)[0]
                    for x in nu:
                        if x == v: continue
                        for y in nv:
                            if y == u: continue
                            if adj[x, y] == 1:
                                count += 1
                    participations[u, v] = count
                    participations[v, u] = count
        return participations

    # Current state: 93 edges.
    # We want to remove 3 edges to get to 90.
    # Then add 1 to get 91.
    
    # Calculate initial C4s
    current_c4 = count_c4s(adj)
    
    # Remove 3 edges with highest participation
    for _ in range(3):
        if time.time() - start_time > DEADLINE - 20:
            break
        parts = get_c4_participation(adj)
        max_part = -1
        max_edge = None
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] == 1 and parts[u, v] > max_part:
                    max_part = parts[u, v]
                    max_edge = (u, v)
        
        if max_edge:
            u, v = max_edge
            adj[u, v] = 0
            adj[v, u] = 0
            edges.remove((u, v))
            current_edges -= 1
            current_c4 = count_c4s(adj)
            
    # Now we have 90 edges (hopefully with 0 or few C4s)
    # Add 1 edge with minimum new C4s
    if current_edges == 90:
        best_pair = None
        best_cost = float('inf')
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v] == 0:
                    # Count C4s created by adding (u,v)
                    # This is the number of length-3 paths between u and v
                    # Which is participations[u,v] in the current graph
                    # We can reuse get_c4_participation but it's for existing edges.
                    # For non-edges, it's the same formula: paths u-x-y-v.
                    count = 0
                    nu = np.where(adj[u] == 1)[0]
                    nv = np.where(adj[v] == 1)[0]
                    for x in nu:
                        for y in nv:
                            if adj[x, y] == 1:
                                count += 1
                    if count < best_cost:
                        best_cost = count
                        best_pair = (u, v)
                        
        if best_pair:
            u, v = best_pair
            adj[u, v] = 1
            adj[v, u] = 1
            edges.append((u, v))
            current_edges += 1
            current_c4 = count_c4s(adj)

    # Local Search
    # Try to swap an edge (u,v) with a non-edge (a,b) if it reduces C4 count.
    # Or just move one edge? No, must keep 91 edges.
    # Swap: remove (u,v), add (a,b).
    
    # To speed up, we can just try to add a non-edge and remove an edge if it helps.
    # But checking all pairs is O(n^4).
    # n=31. 31^4 is ~1M. With 150s, we can do many iterations.
    
    # Precompute neighbors for speed
    # We will do a few rounds of random swaps.
    
    best_c4 = current_c4
    best_adj = adj.copy()
    best_edges = edges.copy()
    
    iterations = 0
    while time.time() - start_time < DEADLINE - 5:
        iterations += 1
        # Pick a random edge to remove
        edge_to_remove = random.choice(edges)
        u1, v1 = edge_to_remove
        
        # Pick a random non-edge to add
        # Find a random non-edge
        while True:
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 != v2 and adj[u2, v2] == 0:
                break
        
        # Calculate delta C4
        # Remove (u1, v1): reduces C4s by participations[u1, v1]
        # Add (u2, v2): increases C4s by participations[u2, v2] in the graph without (u1, v1)?
        # Approximation: use current participations.
        
        # More accurate:
        # 1. Remove (u1, v1)
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        
        # 2. Count C4s in new graph
        new_c4 = count_c4s(adj)
        
        # 3. Add (u2, v2)
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        
        # 4. Count C4s again
        final_c4 = count_c4s(adj)
        
        # Update edges list
        edges.remove(edge_to_remove)
        edges.append((u2, v2))
        
        if final_c4 < best_c4:
            best_c4 = final_c4
            best_adj = adj.copy()
            best_edges = edges.copy()
            
            # If we found a much better solution, we might restart or continue
            if best_c4 == 0:
                break
        else:
            # Revert
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            edges.remove((u2, v2))
            edges.append(edge_to_remove)
            
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            
        # If time is running out, break
        if time.time() - start_time > DEADLINE - 10:
            break
            
    # Return the best found
    # Ensure we return exactly 91 edges
    # best_edges should have 91 edges
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-END
