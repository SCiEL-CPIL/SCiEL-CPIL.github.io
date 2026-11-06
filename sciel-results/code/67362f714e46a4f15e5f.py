import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    best_n = 39
    best_edges = []

    # Helper to check if adding edge (u, v) maintains C4-free property
    def can_add_edge(adj, n, u, v):
        # A C4 is formed if there exists w, x such that:
        # (u,w), (w,x), (x,v), (v,u) are edges.
        # Adding (u,v) creates a C4 if there is a path of length 2 from u to v.
        # i.e., if there is a w such that (u,w) and (w,v) are edges.
        # This is equivalent to checking if u and v have a common neighbor.
        # common_neighbors = adj[u] & adj[v]
        # If count(common_neighbors) > 0, then adding (u,v) creates a C4?
        # Wait.
        # If u-w-v is a path, and we add u-v, we get a triangle u-w-v-u.
        # A triangle is a C3, not a C4.
        # The problem forbids C4 (cycle of length 4).
        # A C4 is u-w-x-v-u.
        # Adding (u,v) creates a C4 if there is a path of length 3 from u to v.
        # i.e., if there exist w, x such that (u,w), (w,x), (x,v) are edges.
        # This is equivalent to checking if the number of common neighbors of u and v is... no.
        # Let's re-evaluate.
        # A C4-free graph has no induced cycle of length 4.
        # Adding an edge (u,v) creates a C4 if there is a path of length 3 between u and v in the existing graph.
        # Path: u - w - x - v.
        # This means w is a neighbor of u, x is a neighbor of v, and (w,x) is an edge.
        # So we need to check if there exists w in N(u) and x in N(v) such that (w,x) is an edge.
        # This is equivalent to: (adj[u] & (adj @ adj)[v]) != 0?
        # (adj @ adj)[v][w] is the number of common neighbors of v and w.
        # If w is a neighbor of u, and w has a common neighbor with v (which is x), and (w,x) is an edge...
        # Actually, if w is neighbor of u, and x is neighbor of v, and (w,x) is an edge, then u-w-x-v is a path of length 3.
        # So we check: is there w in N(u) such that N(w) intersects N(v)?
        # i.e., sum over w in N(u) of (N(w) & N(v)) > 0.
        # This can be computed as: adj[u] @ (adj @ adj[:, v])? No.
        # Let M = adj @ adj. M[i,j] is number of common neighbors of i and j.
        # We want to know if there is a path u-w-x-v.
        # This is equivalent to: (adj[u] @ adj)[:, v] > 0?
        # (adj[u] @ adj) is a vector y where y[j] = sum_k adj[u,k] * adj[k,j].
        # y[j] is the number of neighbors of u that are also neighbors of j. i.e., common neighbors of u and j.
        # So y[v] is the number of common neighbors of u and v.
        # If u and v have a common neighbor w, then u-w-v is a path of length 2.
        # Adding u-v creates a triangle.
        # The problem is C4-free. Triangles are allowed.
        # So adding an edge that creates a triangle is fine.
        # Adding an edge that creates a C4 is bad.
        # A C4 is u-w-x-v-u.
        # This requires a path of length 3: u-w-x-v.
        # So we need to check if there is a path of length 3 between u and v.
        # This is equivalent to: (adj[u] @ (adj @ adj))[:, v] > 0?
        # Let's check.
        # Path u-w-x-v.
        # w is neighbor of u.
        # x is neighbor of w and v.
        # So x is a common neighbor of w and v.
        # So we need to check if there exists w in N(u) such that N(w) intersects N(v).
        # This is: sum_{w in N(u)} |N(w) \cap N(v)| > 0.
        # |N(w) \cap N(v)| is the number of common neighbors of w and v.
        # This is M[w, v].
        # So we need: sum_{w} adj[u,w] * M[w,v] > 0.
        # This is (adj[u] @ M)[:, v].
        # So we can precompute M = adj @ adj.
        # Then check if (adj[u] @ M)[:, v] > 0.
        # Note: M is symmetric.
        # Also, we should exclude the case where w=v or w=u?
        # If w=v, then adj[u,v] must be 1? No, we are adding u-v, so adj[u,v] is currently 0.
        # So w cannot be v because adj[u,v]=0.
        # If w=u, then adj[u,u]=0.
        # So the check is valid.

        # Optimization:
        # We can maintain M dynamically?
        # When adding (u,v), M changes.
        # M[i,j] increases by 1 if i is neighbor of u or v?
        # Actually, adding (u,v) increases the degree of u and v.
        # It affects M[i,j] if i or j is u or v, or if i and j are both neighbors of u or v?
        # This is getting complex.
        # For n <= 47, we can just recompute M or do the check directly.
        # Direct check:
        # for w in range(n):
        #   if adj[u,w] == 1:
        #     for x in range(n):
        #       if adj[w,x] == 1 and adj[x,v] == 1:
        #         return False
        # This is O(n^2) per edge addition.
        # With ~1000 edges, this is 1000 * 47^2 ~ 2.2M ops. Fast enough.

        # Even faster:
        # N_u = np.where(adj[u] == 1)[0]
        # N_v = np.where(adj[v] == 1)[0]
        # For each w in N_u, check if N(w) intersects N_v.
        # N(w) = np.where(adj[w] == 1)[0]
        # Intersection = np.intersect1d(N(w), N_v)
        # If len > 0, return False.

        N_u = np.where(adj[u] == 1)[0]
        N_v = np.where(adj[v] == 1)[0]
        for w in N_u:
            # Check if any neighbor of w is in N_v
            # adj[w, N_v] gives the edges between w and N_v
            if np.any(adj[w, N_v] == 1):
                return False
        return True

    def try_construct(n, seed):
        random.seed(seed)
        target_deg = n - 39

        # Polarity Graph of PG(2,7)
        def generate_polarity_graph(q):
            points = []
            for x in range(q):
                for y in range(q):
                    for z in range(q):
                        if x == 0 and y == 0 and z == 0: continue
                        if x != 0:
                            inv_x = pow(x, -1, q)
                            px = 1
                            py = (y * inv_x) % q
                            pz = (z * inv_x) % q
                            points.append((px, py, pz))
                        elif y != 0:
                            inv_y = pow(y, -1, q)
                            px = 0
                            py = 1
                            pz = (z * inv_y) % q
                            points.append((px, py, pz))
                        else:
                            inv_z = pow(z, -1, q)
                            px = 0
                            py = 0
                            pz = 1
                            points.append((px, py, pz))
            
            unique_points = list(set(points))
            unique_points.sort()
            
            n_pts = len(unique_points)
            adj = np.zeros((n_pts, n_pts), dtype=np.int8)
            
            for i in range(n_pts):
                for j in range(i+1, n_pts):
                    u = unique_points[i]
                    v = unique_points[j]
                    dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
                    if dot == 0:
                        adj[i][j] = 1
                        adj[j][i] = 1
                        
            return unique_points, adj

        # Try Polarity Graph Subset for n=46, 45, etc.
        # For n=47, target=8. PG(2,7) is 8-regular on 57 vertices.
        # Induced subgraph of size 47 will have degrees < 8 for some vertices.
        # So n=47 is unlikely to work with just induced subgraph.
        # But we can try to ADD edges to the induced subgraph to boost degrees?
        # No, adding edges might create C4.
        # So for n=47, we might need a different approach or accept n=46.
        # The task says "value is known to be 46 or 47, so 46 vertices would settle it."
        # So finding n=46 is a huge win (score 46/seed).
        # Current best is ~1.128 * seed. If seed is 40, score is ~45.
        # If we find 46, score is 46.
        # Let's focus on finding n=46 reliably.

        try:
            pts, full_adj = generate_polarity_graph(7)
            n_full = len(pts)
            
            if n_full >= n:
                # We want a subset of size n with min degree >= target_deg.
                # For n=46, target=7.
                # Remove 11 vertices from 57.
                # Each removed vertex reduces degree of its neighbors by 1.
                # If we remove a set R of vertices, the degree of v in induced subgraph is:
                # deg_full(v) - |N(v) \cap R|.
                # We need deg_full(v) - |N(v) \cap R| >= target_deg for all v in S.
                # deg_full(v) = 8.
                # So we need |N(v) \cap R| <= 8 - target_deg.
                # For target=7, we need |N(v) \cap R| <= 1.
                # So each vertex in S can have at most 1 neighbor in R.
                # This is a constraint on R.
                # R must be an independent set? No.
                # If u, v in R, and (u,v) is an edge, then u and v are neighbors.
                # But u and v are removed.
                # The constraint is on vertices in S.
                # For v in S, at most 1 neighbor in R.
                
                # Greedy removal:
                # Start with R = empty, S = all.
                # Iteratively pick a vertex to remove.
                # Which one?
                # We want to minimize the impact on S.
                # Removing v adds v to R.
                # This increases |N(u) \cap R| by 1 for all u in N(v) \cap S.
                # We must ensure that for all u in N(v) \cap S, the new count <= 1.
                # So we can only remove v if for all u in N(v) \cap S, current count < 1 (i.e., 0).
                # This means N(v) \cap S must be disjoint from any previous R-neighbor?
                # Basically, we are looking for a set R such that the bipartite graph between R and S has max degree 1 on the S side.
                # This is equivalent to saying R is a "1-independent" set?
                
                # Let's try a heuristic:
                # Randomly shuffle vertices.
                # Iterate through them.
                # If we can remove it (i.e., all its neighbors in S have count 0), remove it.
                # Stop when |R| = n_full - n.
                
                for attempt in range(200):
                    if time.time() > deadline:
                        return None
                    
                    order = list(range(n_full))
                    random.shuffle(order)
                    
                    R = set()
                    counts = np.zeros(n_full, dtype=np.int8) # counts[i] = |N(i) \cap R|
                    
                    removed = 0
                    target_removed = n_full - n
                    
                    for v in order:
                        if removed == target_removed:
                            break
                        # Check if we can remove v
                        # v is in S initially.
                        # We need to check neighbors in S.
                        # Neighbors of v are N(v).
                        # Those in S are N(v) \ R.
                        # For all u in N(v) \ R, we need counts[u] == 0.
                        # If any counts[u] > 0, we cannot remove v.
                        
                        can_remove = True
                        for u in range(n_full):
                            if full_adj[v][u] == 1 and u not in R:
                                if counts[u] > 0:
                                    can_remove = False
                                    break
                        
                        if can_remove:
                            R.add(v)
                            removed += 1
                            # Update counts for neighbors of v in S
                            for u in range(n_full):
                                if full_adj[v][u] == 1 and u not in R:
                                    counts[u] += 1
                    
                    if removed == target_removed:
                        # Build induced subgraph on S = all \ R
                        S = [i for i in range(n_full) if i not in R]
                        # Map old to new
                        old_to_new = {old: new for new, old in enumerate(S)}
                        
                        adj_sub = np.zeros((n, n), dtype=np.int8)
                        for new_i, old_i in enumerate(S):
                            for new_j, old_j in enumerate(S):
                                if new_i < new_j:
                                    if full_adj[old_i][old_j]:
                                        adj_sub[new_i][new_j] = 1
                                        adj_sub[new_j][new_i] = 1
                        
                        degs = adj_sub.sum(axis=1)
                        if np.min(degs) >= target_deg:
                            edges = []
                            for i in range(n):
                                for j in range(i+1, n):
                                    if adj_sub[i][j]:
                                        edges.append([i, j])
                            return {'n': n, 'edges': edges}
            
        except Exception:
            pass

        # Fallback: Random + Repair
        # Start with random graph, remove edges to break C4s.
        # This is what the original did.
        # Let's try a denser start and aggressive repair.
        
        p = 0.3
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Repair loop
        for step in range(2000):
            if time.time() > deadline:
                return None
            
            # Check for C4s
            # Find a C4.
            # Iterate over pairs (u, v) with no edge?
            # If they have 2 common neighbors, it's a C4?
            # No, C4 is u-w-x-v-u.
            # If u and v have 2 common neighbors w, x, and (w,x) is NOT an edge, then u-w-x-v-u is a C4?
            # No, u-w, w-x, x-v, v-u.
            # If (w,x) is not an edge, it's an induced C4.
            # If (w,x) is an edge, it's two triangles sharing an edge? No.
            # The condition for C4-free is: for any u != v, the number of common neighbors is at most 1?
            # No. If u and v have 2 common neighbors w, x.
            # Then u-w-v and u-x-v are triangles.
            # If (w,x) is not an edge, then u-w-x-v-u is a cycle of length 4.
            # So if there exists u, v with 2 common neighbors w, x such that (w,x) is not an edge, we have a C4.
            # If (w,x) is an edge, then w and x are adjacent.
            # Then u-w-x-v-u is not a simple cycle?
            # u-w, w-x, x-v, v-u.
            # This is a simple cycle of length 4.
            # So ANY pair u, v with 2 common neighbors creates a C4?
            # Wait.
            # If u, v have common neighbors w, x.
            # Edges: (u,w), (v,w), (u,x), (v,x).
            # Cycle: u-w-v-x-u.
            # This is a C4.
            # So the condition for C4-free is: for all u != v, |N(u) \cap N(v)| <= 1.
            # This is a very strong condition.
            # It means the graph is a "linear space" or something similar?
            # Actually, this is the definition of a graph with no C4.
            
            # So we just need to ensure that no pair has 2 common neighbors.
            
            common = adj @ adj
            np.fill_diagonal(common, 0)
            
            # Find pair with common >= 2
            mask = common >= 2
            if not np.any(mask):
                if np.min(adj.sum(axis=1)) >= target_deg:
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i][j]:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                else:
                    break
            
            # Pick a random violating pair
            indices = np.where(mask)
            pos = random.randint(0, len(indices[0])-1)
            u, v = indices[0][pos], indices[1][pos]
            
            # Find common neighbors
            commons = np.where((adj[u] & adj[v]) == 1)[0]
            
            # Remove an edge to break the C4.
            # We can remove (u, w) or (v, w) for some w in commons.
            # To minimize degree loss, remove edge incident to higher degree vertex?
            # Or just random.
            w = random.choice(commons.tolist())
            
            deg_u = adj[u].sum()
            deg_v = adj[v].sum()
            
            if deg_u > deg_v:
                x = u
            else:
                x = v
                
            adj[x][w] = 0
            adj[w][x] = 0
            
            if np.min(adj.sum(axis=1)) < target_deg:
                break

        return None

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        found = False
        for seed in range(1000):
            if time.time() > deadline:
                break
            res = try_construct(n, seed)
            if res is not None:
                if n > best_n:
                    best_n = n
                    best_edges = res['edges']
                found = True
                if n == 47:
                    return {'n': n, 'edges': best_edges}
                break
        
        if found:
            continue
        else:
            pass

    return {'n': best_n, 'edges': best_edges}
