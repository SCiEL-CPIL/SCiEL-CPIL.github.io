import time
import random

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    q = 5
    n = q * q + q + 1  # 31
    
    # Generate points in P^2(F_q)
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize to ensure unique representation
                # Find first non-zero element
                if x != 0:
                    points.append([x, y, z])
                elif y != 0:
                    points.append([0, y, z])
                elif z != 0:
                    points.append([0, 0, z])
    
    # The above normalization is slightly off for projective space.
    # Better: generate all (x,y,z) != (0,0,0) and map to a canonical form.
    # Canonical form: first non-zero element is 1.
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Find first non-zero
                if x != 0:
                    # Multiply by x^{-1}
                    inv_x = pow(x, q - 2, q)
                    p = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q - 2, q)
                    p = (0, 1, (z * inv_y) % q)
                else:
                    p = (0, 0, 1)
                
                if p not in seen:
                    seen.add(p)
                    points.append(p)
    
    # Map points to indices 0..30
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Build adjacency matrix for polarity graph
    # Polarity: P=(x,y,z) is adjacent to Q=(a,b,c) if ax+by+cz = 0 and P != Q
    adj = [[False] * n for _ in range(n)]
    edges = []
    
    for i, P in enumerate(points):
        for j, Q in enumerate(points):
            if i == j:
                continue
            # Check if P is on polar line of Q
            # Polar line of Q=(a,b,c) is ax+by+cz=0
            # So we check a*Px + b*Py + c*Pz = 0
            a, b, c = Q
            Px, Py, Pz = P
            if (a * Px + b * Py + c * Pz) % q == 0:
                adj[i][j] = True
                if i < j:
                    edges.append([i, j])
    
    # The polarity graph is C4-free.
    # Count edges
    m = len(edges)
    
    # We need 91 edges.
    # If m > 90, we remove m-90 edges.
    # Since it's C4-free, removing edges keeps it C4-free.
    # To minimize C4s when adding the 91st edge, we should keep the graph as "dense" as possible?
    # Or maybe we should remove edges that are "critical" for C4 formation?
    # Actually, we want to add an edge that creates few C4s.
    # The number of C4s created by adding (u,v) is the number of edges between N(u) and N(v).
    # If we remove edges, we reduce the potential C4s for future additions.
    # So we should remove edges that are part of many potential C4s?
    # But we don't know which edge we will add yet.
    
    # Strategy:
    # 1. If m >= 91, we can just pick 90 edges from the polarity graph that form a C4-free subgraph.
    #    Actually, the polarity graph itself is C4-free.
    #    So any subset of its edges is C4-free.
    #    We want to choose 90 edges such that there exists a non-edge (u,v) in the resulting graph
    #    with minimal edges between N(u) and N(v).
    #    
    #    Let's first check m.
    if m >= 91:
        # We have more than 90 edges. We can remove m-90 edges.
        # To minimize the cost of adding the 91st edge, we should remove edges that
        # are most likely to be in N(u) x N(v) for the pair we add.
        # But we don't know the pair yet.
        # 
        # Alternative: Just take the first 90 edges? No, that's arbitrary.
        # Let's try to find a pair (u,v) in the full polarity graph that has minimal "cost"
        # (edges between N(u) and N(v)).
        # Then, if that pair is not an edge in the polarity graph, we can keep all 90 edges
        # that are not (u,v) and add (u,v).
        # Wait, (u,v) is a non-edge in the polarity graph?
        # The polarity graph is not complete.
        # So there are non-edges.
        # We want to add a non-edge (u,v) to a C4-free graph with 90 edges.
        # If the polarity graph has m edges, and we keep 90 of them, we have a C4-free graph G'.
        # We add (u,v) which is not in G'.
        # To minimize C4s, we want (u,v) to have few edges between N_{G'}(u) and N_{G'}(v).
        #
        # Let's calculate the cost of adding each non-edge (u,v) to the FULL polarity graph.
        # cost(u,v) = number of edges (w,x) in polarity graph such that w in N(v) and x in N(u).
        #
        # If we remove edges, the cost can only decrease or stay same.
        # So the minimum cost in the full graph is an upper bound on the minimum cost in any subgraph.
        #
        # Let's find the non-edge (u,v) with minimum cost in the full polarity graph.
        # Let this be (u*, v*).
        # Now, we need to form a graph with 90 edges from the polarity graph.
        # We should keep all edges EXCEPT those that are not needed?
        # Actually, we can just keep any 90 edges.
        # But we must ensure (u*, v*) is not one of the kept edges (it's a non-edge anyway).
        # And we want to minimize the cost of adding (u*, v*) to the 90-edge subgraph.
        # The cost is the number of edges between N(u*) and N(v*) in the subgraph.
        # To minimize this, we should remove edges that are between N(u*) and N(v*).
        #
        # So:
        # 1. Find (u*, v*) non-edge with min cost in full graph.
        # 2. Identify all edges in full graph that are between N(u*) and N(v*).
        # 3. Remove these edges first.
        # 4. If we still have more than 90 edges, remove arbitrary others.
        # 5. Add (u*, v*).
        
        # Step 1: Find min cost non-edge
        min_cost = float('inf')
        best_pair = None
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    continue
                # Calculate cost: edges between N(v) and N(u)
                cost = 0
                for w in range(n):
                    if adj[v][w]:
                        for x in range(n):
                            if adj[u][x] and adj[w][x]:
                                cost += 1
                if cost < min_cost:
                    min_cost = cost
                    best_pair = (u, v)
        
        u_star, v_star = best_pair
        
        # Step 2: Identify edges between N(u_star) and N(v_star)
        edges_to_remove = set()
        for w in range(n):
            if adj[v_star][w]:
                for x in range(n):
                    if adj[u_star][x] and adj[w][x]:
                        e1 = (min(w, x), max(w, x))
                        e2 = (min(u_star, v_star), max(u_star, v_star)) # Not this
                        # The edge is (w, x)
                        edges_to_remove.add(e1)
        
        # Step 3: Remove these edges
        kept_edges = []
        removed_count = 0
        target_edges = 90
        # We have m edges in polarity graph.
        # We want to keep 90.
        # Prioritize keeping edges NOT in edges_to_remove.
        
        # Sort edges: those NOT in edges_to_remove first?
        # Actually, we want to REMOVE edges in edges_to_remove.
        # So we keep edges not in edges_to_remove.
        
        edges_not_to_remove = []
        edges_to_remove_list = []
        for e in edges:
            e_tuple = (min(e[0], e[1]), max(e[0], e[1]))
            if e_tuple in edges_to_remove:
                edges_to_remove_list.append(e)
            else:
                edges_not_to_remove.append(e)
        
        # We need to keep 90 edges.
        # If len(edges_not_to_remove) >= 90, we keep 90 from there.
        # Else, we keep all from there, and some from edges_to_remove_list.
        
        if len(edges_not_to_remove) >= 90:
            final_edges = edges_not_to_remove[:90]
        else:
            final_edges = edges_not_to_remove
            need_more = 90 - len(final_edges)
            final_edges.extend(edges_to_remove_list[:need_more])
        
        # Add the new edge
        final_edges.append([u_star, v_star])
        
        return final_edges
    else:
        # m < 91. This shouldn't happen for q=5 polarity graph (m=93?).
        # If m < 91, we need to add edges.
        # But polarity graph is C4-free, so adding edges creates C4s.
        # We need 91 edges.
        # If m < 91, we add 91-m edges.
        # This case is unlikely.
        return edges + [[i, j] for i in range(n) for j in range(i+1, n) if not adj[i][j]][:91-m]

# Test locally
if __name__ == "__main__":
    edges = construct()
    print(len(edges))
    # Count C4s
    n = 31
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    c4_count = 0
    for u in range(n):
        for v in range(u+1, n):
            if not adj[u][v]:
                continue
            # Count common neighbors
            common = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    common += 1
            # Number of C4s containing (u,v) is not just common.
            # A C4 is u-v-w-x-u.
            # This is counted by pairs of common neighbors? No.
            # If we iterate over all pairs of edges, it's complicated.
            # Simpler: count all 4-cycles.
            pass
    
    # Better C4 count
    c4_count = 0
    for u in range(n):
        for v in range(u+1, n):
            if not adj[u][v]:
                continue
            # For each pair of neighbors w, x of u, check if v is connected to both? No.
            # Standard count:
            # Sum over all pairs of vertices (u, v) of (common_neighbors(u, v) choose 2)?
            # No, that counts C4s where u, v are opposite?
            # If u, v are adjacent, they cannot be opposite in a C4.
            # So C4s are formed by pairs of vertices that are NOT adjacent?
            # No, a C4 is u-v-w-x-u.
            # u and w are opposite. v and x are opposite.
            # So we sum over all pairs (u, w) that are NOT adjacent?
            # No, u and w can be adjacent? No, in a simple graph C4, opposite vertices are not adjacent.
            # So we sum over all pairs (u, w) such that u != w, and count common neighbors.
            # Each C4 is counted twice (once for each pair of opposite vertices).
            # So C4s = 1/2 * Sum_{u != w} (common_neighbors(u, w) choose 2)?
            # No, if u, w have k common neighbors, the number of C4s with u, w opposite is k choose 2.
            # And each C4 has 2 pairs of opposite vertices.
            # So Total C4s = 1/2 * Sum_{u, w} C(common(u, w), 2).
            
            common = 0
            for w in range(n):
                if w != u and w != v:
                    if adj[u][w] and adj[v][w]:
                        common += 1
            # This counts C4s where u, v are adjacent? No.
            # If u, v are adjacent, they are part of the C4 as adjacent vertices.
            # The formula above counts C4s where u, v are OPPOSITE?
            # No, if u, v are adjacent, they are not opposite.
            # So we should sum over ALL pairs (u, v), not just adjacent.
            
    total_c4 = 0
    for u in range(n):
        for v in range(u+1, n):
            common = 0
            for w in range(n):
                if w != u and w != v:
                    if adj[u][w] and adj[v][w]:
                        common += 1
            total_c4 += common * (common - 1) // 2
    total_c4 //= 2
    print("C4s:", total_c4)
