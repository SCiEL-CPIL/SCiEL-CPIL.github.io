"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin

    def build_polarity_graph_q5():
        q = 5
        n = q * q + q + 1
        points = []
        for x in range(q):
            for y in range(q):
                points.append((1, x, y))
        for y in range(q):
            points.append((0, 1, y))
        points.append((0, 0, 1))
        
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                u = points[i]
                v = points[j]
                dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
                if dot == 0:
                    adj[i][j] = True
                    adj[j][i] = True
        np.fill_diagonal(adj, False)
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return n, edges

    best_n, best_edges = build_polarity_graph_q5()

    def check_c4_free(adj):
        n = len(adj)
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    common = np.sum(adj[u] & adj[v])
                    if common > 1:
                        return False
        return True

    def find_independent_set(adj, k, max_iter=100):
        """
        Greedy randomized search for independent set of size k.
        Returns True if found, False otherwise.
        """
        n = len(adj)
        verts = list(range(n))
        for _ in range(max_iter):
            random.shuffle(verts)
            independent = []
            for v in verts:
                if all(not adj[v][u] for u in independent):
                    independent.append(v)
                    if len(independent) >= k:
                        return True
        return False

    def greedy_construct(n_target, seed=None):
        """
        Construct a C4-free graph on n_target vertices using a greedy approach.
        We pick vertices one by one. For each new vertex, we choose a subset of
        existing vertices to connect to, such that no C4 is formed.
        To maximize density (and minimize alpha), we want to connect to as many
        as possible, but we must respect the C4 constraint.
        
        Constraint: If u and v are existing vertices, and we add new vertex w,
        we can connect w to u and w to v ONLY IF u and v do NOT share any other
        common neighbor (other than w, which doesn't exist yet, so just check
        if u and v have a common neighbor in the existing graph).
        Actually, the condition for adding edges (w,u) and (w,v) without creating
        a C4 is that N(u) ∩ N(v) = ∅ in the current graph?
        No. A C4 is u-v-w-x-u.
        If we add w connected to u and v, a C4 is formed if there is a vertex x
        such that x is connected to u and x is connected to v.
        So, N(u) ∩ N(v) must be empty for the pair (u,v) that w connects to?
        Wait. If w connects to u and v, and there exists x such that x~u and x~v,
        then u-v-w-x-u is a C4? No, u-v is not necessarily an edge.
        The cycle is u-x-v-w-u.
        Edges: (u,x), (x,v), (v,w), (w,u).
        Yes. So if we add w connected to S (subset of existing vertices),
        then for any pair u,v in S, N(u) ∩ N(v) must be empty?
        Actually, if u,v in S, and x is a common neighbor of u and v,
        then u-x-v-w-u is a C4.
        So, the set of vertices S that w connects to must be an independent set
        in the "line graph" sense? No.
        The condition is: For all u, v in S, N(u) ∩ N(v) = ∅.
        This means S is a set of vertices in the current graph such that no two
        vertices in S have a common neighbor.
        
        This is a very strong constraint. It implies that the subgraph induced by
        N(u) for u in S is a matching? No, just that the neighborhoods are disjoint.
        
        Let's refine the greedy step:
        1. Start with empty graph.
        2. For each new vertex w:
           a. Compute for each existing vertex u, its neighborhood N(u).
           b. We want to choose a subset S of existing vertices to connect to w.
           c. Constraint: For all u, v in S, N(u) ∩ N(v) = ∅.
           d. To maximize density, we want S to be as large as possible.
           e. This is equivalent to finding a large independent set in a graph H
              where vertices are existing vertices, and edges in H connect u,v
              if N(u) ∩ N(v) != ∅.
        
        This is still expensive. Let's use a simpler heuristic:
        Randomly permute existing vertices.
        Iterate through them.
        Maintain a set of "forbidden" vertices for w.
        Initially forbidden = ∅.
        For each candidate u in permutation:
            If u is not in forbidden:
                Connect w to u.
                For all v such that N(u) ∩ N(v) != ∅:
                    Add v to forbidden.
                    (Also, if v was already connected, we might have a problem?
                     No, we are building S. If we connect to u, we cannot connect
                     to any v that shares a neighbor with u.)
        
        This greedy choice is fast and ensures C4-freeness.
        """
        adj = np.zeros((n_target, n_target), dtype=bool)
        
        for w in range(n_target):
            if w == 0:
                continue
            
            # Get neighborhoods of existing vertices
            # We need to check conflicts.
            # Conflict(u, v) if N(u) ∩ N(v) is not empty.
            
            # Heuristic:
            # 1. Create list of existing vertices 0..w-1
            # 2. Shuffle
            # 3. Iterate
            # 4. Keep track of 'blocked' vertices.
            
            blocked = np.zeros(w, dtype=bool)
            candidates = list(range(w))
            random.shuffle(candidates)
            
            for u in candidates:
                if blocked[u]:
                    continue
                
                # Check if we can connect w to u.
                # We can, if no previously selected vertex v has N(u) ∩ N(v) != ∅.
                # Wait, my 'blocked' logic above was:
                # If I connect to u, I block all v such that N(u) ∩ N(v) != ∅.
                # So I just need to check if u is blocked.
                
                # Connect w to u
                adj[w][u] = True
                adj[u][w] = True
                
                # Now block all v that share a neighbor with u.
                # N(u) are the neighbors of u in the current graph (indices < w).
                # For each x in N(u), for each y in N(x) (y < w, y != u):
                #     y shares neighbor x with u. So y is blocked.
                
                # Efficient way:
                # For each x in N(u):
                #     neighbors_of_x = np.where(adj[x])[0]
                #     neighbors_of_x = neighbors_of_x[neighbors_of_x < w]
                #     for y in neighbors_of_x:
                #         if y != u:
                #             blocked[y] = True
                
                neighbors_u = np.where(adj[u])[0]
                neighbors_u = neighbors_u[neighbors_u < w]
                
                for x in neighbors_u:
                    neighbors_x = np.where(adj[x])[0]
                    neighbors_x = neighbors_x[neighbors_x < w]
                    for y in neighbors_x:
                        if y != u:
                            blocked[y] = True
        
        return adj

    def local_search_improve(adj, n_target):
        """
        Try to add edges to reduce independence number.
        We iterate through non-edges (u,v).
        If adding (u,v) doesn't create a C4, we add it.
        Check: Adding (u,v) creates a C4 iff |N(u) ∩ N(v)| > 0.
        Wait, if |N(u) ∩ N(v)| >= 1, let x be a common neighbor.
        Then u-x-v-...-u? No.
        The C4 would be u-x-v-y-u where y is another common neighbor?
        No.
        If we add edge (u,v), a C4 is formed if there exists a path of length 3
        from u to v in the existing graph.
        Path: u-x-y-v.
        This requires x in N(u), y in N(v), and (x,y) is an edge.
        So we need to check if there is an edge between N(u) and N(v).
        
        Let's check:
        For each x in N(u):
            For each y in N(v):
                If adj[x][y] is True:
                    Adding (u,v) creates C4 u-x-y-v-u.
        
        So the condition to add (u,v) is:
        No edge exists between N(u) and N(v).
        """
        n = n_target
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not adj[u][v]:
                        # Check if adding (u,v) creates a C4
                        # N(u) and N(v)
                        nu = np.where(adj[u])[0]
                        nv = np.where(adj[v])[0]
                        
                        # Check if any x in nu is connected to any y in nv
                        # This is a bit expensive if done naively.
                        # Use matrix multiplication or boolean ops.
                        # Create a mask for nu and nv
                        # Actually, just check if (adj[nu][:, nv]).any()
                        # But nu and nv are arrays of indices.
                        
                        # Faster:
                        # common_edges = np.sum(adj[np.ix_(nu, nv)])
                        # if common_edges > 0: C4 created.
                        
                        if len(nu) == 0 or len(nv) == 0:
                            # Safe to add
                            adj[u][v] = True
                            adj[v][u] = True
                            changed = True
                        else:
                            # Check intersection
                            # We want to know if there is any edge between sets nu and nv
                            # sub_adj = adj[np.ix_(nu, nv)]
                            # if np.any(sub_adj):
                            #     pass
                            # else:
                            #     add edge
                            
                            # Optimization: If |nu| * |nv| is small, check directly.
                            # Otherwise, use matrix product.
                            # Given n <= 43, direct check is fine.
                            
                            conflict = False
                            for x in nu:
                                # Check if x is connected to any y in nv
                                # adj[x, nv]
                                if np.any(adj[x, nv]):
                                    conflict = True
                                    break
                            
                            if not conflict:
                                adj[u][v] = True
                                adj[v][u] = True
                                changed = True
        return adj

    # Try to construct graphs with n=39, 40, etc.
    # Start with 39 as it's the known bound.
    
    for n_target in [39, 40, 41, 42, 43]:
        if time.time() > deadline - 10:
            break
            
        best_adj_for_n = None
        best_alpha = 999
        
        # Try multiple random seeds for the greedy construction
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Set seed for reproducibility within attempt? No, use global random.
            adj = greedy_construct(n_target)
            
            # Verify C4-free (should be, but just in case)
            if not check_c4_free(adj):
                # Fallback: remove C4s
                # This is a safety net
                for _ in range(100):
                    c4_found = False
                    for u in range(n_target):
                        for v in range(u + 1, n_target):
                            if adj[u][v]:
                                common = np.where(adj[u] & adj[v])[0]
                                if len(common) > 1:
                                    adj[u][v] = False
                                    adj[v][u] = False
                                    c4_found = True
                                    break
                        if c4_found:
                            break
                    if not c4_found:
                        break
            
            # Improve by adding edges
            adj = local_search_improve(adj, n_target)
            
            # Check independence number
            # Use greedy search to see if alpha >= 11
            # If we can't find an independent set of 11, it's likely alpha < 11.
            # To be more robust, we can try to find the max independent set size
            # using a simple branch and bound or just trust the greedy for now.
            # Given the time limit, greedy is the way to go.
            
            has_large_ind_set = find_independent_set(adj, 11, max_iter=200)
            
            if not has_large_ind_set:
                # This is a candidate
                # Estimate alpha more accurately?
                # Let's just pick the one with the most edges (densest)
                # Denser graph usually has smaller alpha.
                edge_count = np.sum(adj) // 2
                if best_adj_for_n is None:
                    best_adj_for_n = adj
                else:
                    # Compare edge counts
                    # If new has more edges, it's likely better
                    if edge_count > np.sum(best_adj_for_n) // 2:
                        best_adj_for_n = adj
                
                if time.time() > deadline - 5:
                    break
                # If we found one, we might want to try more to see if we can get denser
                # But if we are running out of time, break.
        
        if best_adj_for_n is not None:
            # Final verification
            if check_c4_free(best_adj_for_n) and not find_independent_set(best_adj_for_n, 11, max_iter=500):
                edges = []
                for u in range(n_target):
                    for v in range(u + 1, n_target):
                        if best_adj_for_n[u][v]:
                            edges.append([u, v])
                
                if n_target > best_n:
                    best_n = n_target
                    best_edges = edges
                break # Found a valid graph for this n, no need to try larger n if we are tight on time?
                # Actually, if we found n=39, we can try 40 if time permits.
                # But the loop structure breaks out of the attempt loop, not the n_target loop.
                # Let's not break the n_target loop immediately, but check time.

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
