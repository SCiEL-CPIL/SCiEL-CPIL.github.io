def construct():
    import time
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    def instance_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def solve_instance(n, p, seed):
        edges = instance_graph(n, p, seed)
        if not edges:
            return []
        
        m = len(edges)
        
        # Build bipartite graph for maximum matching
        # Left side: v_in for each vertex v
        # Right side: v_out for each vertex v
        # For each edge {u,v}: add u_in -> v_out and v_in -> u_out
        # This is a bipartite graph with n nodes on each side
        # Edge from left i to right j exists if there's an original edge {i,j}
        
        # Cost matrix: 0 where edge exists, large value where it doesn't
        # We want maximum matching, so minimize cost with 0 for valid edges
        INF = 10**6
        cost = np.full((n, n), INF, dtype=float)
        
        for u, v in edges:
            cost[u][v] = 0  # u_in -> v_out
            cost[v][u] = 0  # v_in -> u_out
        
        # Find maximum matching using linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # The matching size is the number of assignments where cost < INF
        matching_size = 0
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost[r][c] < INF:
                matching_size += 1
                matches.append((r, c))
        
        # Number of paths = m - matching_size
        # Now reconstruct paths
        
        # The matching tells us: for each matched (r, c), we have a connection from r_in to c_out
        # This means: an edge ending at r is connected to an edge starting at c
        # But we need to figure out which specific edges are connected
        
        # Better approach: Build directed graph based on matching
        # For each original edge {u,v}, we need to decide its direction
        # The matching gives us: if (u, v) is in matching, it means u_in matched to v_out
        # This implies that at vertex u, some edge enters, and at vertex v, some edge leaves
        # But we need to be more careful
        
        # Alternative reconstruction:
        # The matching pairs incoming half-edges with outgoing half-edges
        # For each vertex v, the matched pairs tell us which edges are "connected" through v
        # We can build a directed graph where:
        #   - Each original edge {u,v} gets a direction
        #   - The direction is determined by the matching
        
        # Actually, let's think differently. The matching in the bipartite graph gives us:
        #   - A set of directed edges in the original graph
        #   - Specifically, if (u, v) is in the matching, it contributes to the flow
        #   
        # Let me use a different reconstruction method:
        # Build a directed multigraph where each original edge is directed
        # such that the in/out degrees match the matching
        
        # For each vertex v:
        #   - in_degree(v) = number of matches where v is the column (v_out is matched)
        #   - out_degree(v) = number of matches where v is the row (v_in is matched)
        #   - unmatched_in(v) = deg(v) - in_degree(v) = number of edges ending at v that are path starts
        #   - unmatched_out(v) = deg(v) - out_degree(v) = number of edges starting at v that are path ends
        
        # Wait, I think I have the directions confused. Let me reconsider.
        # 
        # In the bipartite graph:
        #   - Left node v_in represents "v as a tail" (edge starts at v)
        #   - Right node v_out represents "v as a head" (edge ends at v)
        #   - Edge u_in -> v_out in bipartite graph means: we can connect an edge starting at u to an edge ending at v
        #   
        # No wait, that's not right either. Let me look at this more carefully.
        #
        # Standard formulation for path cover:
        #   - Create bipartite graph with V_left = V, V_right = V
        #   - For each undirected edge {u,v} in G, add directed edges u->v and v->u in the bipartite graph
        #   - Find maximum matching M
        #   - Number of paths = |E| - |M|
        #
        # The matching M is a set of pairs (u, v) meaning that in the path cover, 
        # there's a path that goes through edge ...-u-v-... (i.e., the edge {u,v} is 
        # oriented as u->v in the path)
        #
        # So each matched pair (u, v) corresponds to orienting edge {u,v} as u->v
        # But wait, each edge {u,v} can only be oriented one way, and the matching 
        # can have at most one of (u,v) or (v,u) for each edge {u,v}
        #
        # Actually, the matching can have both (u,v) and (v,u) only if there are 
        # multiple edges between u and v, but in our simple graph, there's at most 
        # one edge {u,v}. So the matching has at most one of (u,v) or (v,u).
        #
        # Hmm, but the matching size can be up to m (one per edge), and we have 
        # 2m possible directed edges in the bipartite graph (2 per undirected edge).
        # The constraint is that for each undirected edge {u,v}, at most one of 
        # (u,v) or (v,u) can be in the matching.
        #
        # But linear_sum_assignment doesn't enforce this constraint! It just finds 
        # a maximum matching in the bipartite graph, which could potentially match 
        # both (u,v) and (v,u) for the same undirected edge {u,v}.
        #
        # Wait, no. In the bipartite graph, (u,v) and (v,u) are different edges. 
        # The matching can include both. But that would mean we're using the same 
        # undirected edge twice, which is not allowed.
        #
        # I think I'm overcomplicating this. Let me look up the correct formulation.
        #
        # Correct formulation: The minimum path cover in a general graph (not DAG) 
        # is NP-hard in general! But for our purpose, we just need a good approximation.
        #
        # Actually, for the edge-disjoint path cover problem, the formulation I described 
        # is for the case where we're finding a minimum number of paths that cover all 
        # edges, and the paths are vertex-disjoint... no, that's not right either.
        #
        # Let me step back. The problem is: decompose the edge set into the minimum 
        # number of edge-disjoint paths. This is the "minimum path cover" problem for 
        # edges, which is different from the vertex path cover.
        #
        # For the edge version: 
        #   - Each edge must be in exactly one path
        #   - Paths are sequences of edges where consecutive edges share a vertex
        #   - Minimize the number of paths
        #
        # This is equivalent to: find a set of directed edges (orienting each undirected 
        # edge) such that the resulting directed graph is a collection of directed paths, 
        # and minimize the number of paths.
        #
        # The number of paths in a directed graph that is a collection of paths equals:
        #   (number of vertices with outdeg > indeg) = (number of vertices with indeg > outdeg)
        #   = (1/2) * sum_v |outdeg(v) - indeg(v)|
        #
        # So we want to orient the edges to minimize (1/2) * sum_v |outdeg(v) - indeg(v)|.
        #
        # This is the "minimum imbalance orientation" problem, which can be solved 
        # via min-cost flow or matching.
        #
        # The matching formulation: 
        #   - Create a bipartite graph with left = {v_in} and right = {v_out}
        #   - For each undirected edge {u,v}, add edges u_in->v_out and v_in->u_out
        #   - Find maximum matching M
        #   - The matching tells us how to "pair" incoming and outgoing half-edges
        #   - Number of paths = (1/2) * sum_v |deg(v) - 2 * (number of matches at v)|
        #
        # Actually, I think the correct formula is:
        #   - For each vertex v, let in_v = number of matches where v is on the right (v_out matched)
        #   - Let out_v = number of matches where v is on the left (v_in matched)
        #   - The imbalance at v is |deg(v) - in_v - out_v| ... no, that's not right.
        #
        # I'm going in circles. Let me just use the LP approach from the inspiration, 
        # which is known to work, but optimize it for speed.
        
        # Actually, let me try a simpler greedy approach with local search.
        # The current program uses greedy, which gets score 5.46.
        # The LP approach got 1.68.
        # Let me combine: use LP to get initial solution, then improve with local search.
        
        # But the LP might be slow. Let me estimate: for n=30, m~100, 
        # the LP has 2m + n = 230 variables and ~2m + 2n = 260 constraints.
        # This should be fast with HiGHS.
        
        # Let me implement the LP approach properly.
        
        from scipy.optimize import linprog
        
        # Variables: x_i for each directed arc (2m variables), t_v for each vertex (n variables)
        # Total: 2m + n variables
        
        num_vars = 2 * m + n
        
        # Objective: minimize sum t_v
        c = np.zeros(num_vars)
        for v in range(n):
            c[2 * m + v] = 1.0
        
        # Constraints:
        # 1) For each undirected edge i = (u, v): x_{u,v} + x_{v,u} <= 1
        # 2) For each vertex v: t_v >= outdeg(v) - indeg(v)
        # 3) For each vertex v: t_v >= -(outdeg(v) - indeg(v))
        
        A_ub_rows = []
        b_ub_vals = []
        
        # Constraint 1: edge constraints
        for i in range(m):
            row = np.zeros(num_vars)
            row[2 * i] = 1.0
            row[2 * i + 1] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(1.0)
        
        # Precompute coefficients for each vertex
        for v in range(n):
            # outdeg(v) - indeg(v) = sum of x_{v,w} for arcs v->w minus sum of x_{w,v} for arcs w->v
            # For arc 2*i (u_i->v_i): contributes +1 to outdeg(u_i), +1 to indeg(v_i)
            # For arc 2*i+1 (v_i->u_i): contributes +1 to outdeg(v_i), +1 to indeg(u_i)
            
            # Constraint 2: t_v - (outdeg(v) - indeg(v)) >= 0  =>  -t_v + (outdeg(v) - indeg(v)) <= 0
            row = np.zeros(num_vars)
            row[2 * m + v] = -1.0
            for i in range(m):
                u, w = edges[i]
                if u == v:
                    row[2 * i] += 1.0  # arc u->w, u is tail
                if w == v:
                    row[2 * i] -= 1.0  # arc u->w, w is head
                if w == v:
                    row[2 * i + 1] += 1.0  # arc w->u, w is tail
                if u == v:
                    row[2 * i + 1] -= 1.0  # arc w->u, u is head
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
            
            # Constraint 3: t_v + (outdeg(v) - indeg(v)) >= 0  =>  -t_v - (outdeg(v) - indeg(v)) <= 0
            row2 = np.zeros(num_vars)
            row2[2 * m + v] = -1.0
            for i in range(m):
                u, w = edges[i]
                if u == v:
                    row2[2 * i] -= 1.0
                if w == v:
                    row2[2 * i] += 1.0
                if w == v:
                    row2[2 * i + 1] -= 1.0
                if u == v:
                    row2[2 * i + 1] += 1.0
            A_ub_rows.append(row2)
            b_ub_vals.append(0.0)
        
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
        
        bounds = [(0, 1)] * (2 * m) + [(0, None)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if not res.success:
                # Fallback to greedy
                return greedy_path_cover(edges, n)
            
            x = res.x
            
            # Build directed edges
            directed = []
            for i in range(m):
                u, v = edges[i]
                xv = x[2 * i]
                xu = x[2 * i + 1]
                if xv > 0.5:
                    directed.append((u, v))
                elif xu > 0.5:
                    directed.append((v, u))
                else:
                    # Fractional; pick arbitrarily
                    directed.append((u, v))
            
            # Reconstruct paths from directed edges
            return reconstruct_paths(directed, n)
        
        except Exception as e:
            return greedy_path_cover(edges, n)
    
    def greedy_path_cover(edges, n):
        """Fallback greedy approach."""
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
        
        remaining_edges = set(edge_set)
        paths = []
        
        while remaining_edges:
            rem_deg = {i: 0 for i in range(n)}
            for u, v in remaining_edges:
                rem_deg[u] += 1
                rem_deg[v] += 1
            
            start = max(range(n), key=lambda i: rem_deg[i])
            
            path = [start]
            used_vertices = {start}
            current = start
            
            while True:
                candidates = []
                for neighbor in adj[current]:
                    if (min(current, neighbor), max(current, neighbor)) in remaining_edges:
                        if neighbor not in used_vertices:
                            candidates.append(neighbor)
                
                if not candidates:
                    break
                
                candidates.sort(key=lambda x: rem_deg[x], reverse=True)
                next_v = candidates[0]
                
                path.append(next_v)
                used_vertices.add(next_v)
                
                edge_tuple = (min(current, next_v), max(current, next_v))
                remaining_edges.remove(edge_tuple)
                
                rem_deg[current] -= 1
                rem_deg[next_v] -= 1
                
                current = next_v
            
            paths.append(path)
        
        return paths
    
    def reconstruct_paths(directed, n):
        """Reconstruct paths from directed edges."""
        # Build adjacency
        out_adj = {i: [] for i in range(n)}
        in_adj = {i: [] for i in range(n)}
        for u, v in directed:
            out_adj[u].append(v)
            in_adj[v].append(u)
        
        # Find path starts (vertices with outdeg > indeg)
        starts = []
        for v in range(n):
            if len(out_adj[v]) > len(in_adj[v]):
                starts.append(v)
        
        paths = []
        visited_edges = set()
        
        for start in starts:
            if not out_adj[start]:
                continue
            
            path = [start]
            current = start
            while out_adj[current]:
                # Find next unvisited edge
                next_v = None
                for v in out_adj[current]:
                    edge_key = (min(current, v), max(current, v))
                    if edge_key not in visited_edges:
                        next_v = v
                        break
                
                if next_v is None:
                    break
                
                visited_edges.add((min(current, next_v), max(current, next_v)))
                path.append(next_v)
                current = next_v
            
            if len(path) > 1:
                paths.append(path)
        
        # Handle any remaining edges (shouldn't happen if LP is correct)
        remaining = set()
        for u, v in directed:
            edge_key = (min(u, v), max(u, v))
            if edge_key not in visited_edges:
                remaining.add(edge_key)
        
        # Add remaining edges as separate paths
        for u, v in remaining:
            paths.append([u, v])
        
        return paths
    
    # Solve all three instances
    results = []
    for n, p, seed in [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]:
        if time.time() > DEADLINE:
            # Fallback to greedy for remaining
            edges = instance_graph(n, p, seed)
            results.append(greedy_path_cover(edges, n))
        else:
            results.append(solve_instance(n, p, seed))
    
    return results
