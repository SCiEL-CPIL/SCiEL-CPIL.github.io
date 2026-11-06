import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from random import Random

def construct():
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_path_cover(edges, n):
        if not edges:
            return []
        
        # Build adjacency and edge indices
        # Edge list: list of (u, v)
        # Map edge to index
        edge_to_idx = {}
        for i, (u, v) in enumerate(edges):
            edge_to_idx[(u, v)] = i
            edge_to_idx[(v, u)] = i
            
        # Incident edges for each vertex
        incident = {v: [] for v in range(n)}
        for i, (u, v) in enumerate(edges):
            incident[u].append(i)
            incident[v].append(i)
            
        # Variables:
        # For each vertex v, for each pair of incident edges (e1, e2), variable x_{v, e1, e2}
        # Also, we don't explicitly need z variables if we minimize sum of (1 - connected)
        # Constraint for each (v, e): sum_{f != e} x_{v, e, f} + z_{v, e} = 1
        # Minimize sum z_{v, e}
        
        # To simplify, let's just use x variables and derive z.
        # z_{v, e} = 1 - sum_{f != e} x_{v, e, f}
        # Objective: Minimize sum_{v, e} (1 - sum_{f != e} x_{v, e, f})
        #           = Minimize (2*|E| - sum_{v, e} sum_{f != e} x_{v, e, f})
        #           = Maximize sum_{v, e} sum_{f != e} x_{v, e, f}
        # Note: Each pair (e,f) at v contributes to x_{v,e,f} and x_{v,f,e}.
        # So sum_{v, e} sum_{f != e} x_{v, e, f} = 2 * sum_{v} sum_{e < f} x_{v, e, f}
        # So Maximize sum_{v} sum_{e < f} x_{v, e, f} is equivalent to Minimize dangling.
        
        # Let's set up MILP to Maximize connections.
        # c = -1 for all x variables (since milp minimizes, we minimize -connections)
        
        var_indices = []
        var_map = {} # (v, e1, e2) -> index
        for v in range(n):
            inc = incident[v]
            for i in range(len(inc)):
                for j in range(i + 1, len(inc)):
                    e1 = inc[i]
                    e2 = inc[j]
                    idx = len(var_indices)
                    var_indices.append(idx)
                    var_map[(v, e1, e2)] = idx
        
        num_vars = len(var_indices)
        if num_vars == 0:
            # All degrees 0 or 1
            # If degree 1, edge is dangling at both ends? No, dangling at one end?
            # If degree 1, it must be dangling at v.
            # Path is just the edge.
            paths = []
            for u, v in edges:
                paths.append([u, v])
            return paths

        c = np.full(num_vars, -1.0) # Minimize -sum(x) -> Maximize sum(x)
        
        # Constraints:
        # For each vertex v and edge e incident to v:
        # sum_{f in incident[v], f != e} x_{v, e, f} <= 1
        # (Since x are 0/1, and we want to pick at most 1 partner for e)
        # Actually, we want exactly 1 partner OR 0 partners (dangling).
        # The constraint is sum <= 1.
        # If sum == 1, it's connected. If sum == 0, it's dangling.
        
        A_rows = []
        b_lower = []
        b_upper = []
        
        for v in range(n):
            inc = incident[v]
            for e in inc:
                # Find all x variables involving e at v
                coeffs = np.zeros(num_vars)
                for f in inc:
                    if f == e:
                        continue
                    # Check if (v, e, f) or (v, f, e) exists in var_map
                    # We stored (v, e1, e2) with e1 < e2 in terms of list index?
                    # No, we stored based on indices in incident list.
                    # Let's just iterate through var_map keys to be safe or build a lookup.
                    pass
                
                # Efficiently build constraint row
                # For a fixed v and e, the variables are x_{v, e, f} for all f != e.
                # In our var_map, we have keys (v, min(e,f), max(e,f))?
                # No, we used indices i, j from the incident list.
                # Let's re-verify var_map keys.
                # We used e1, e2 as edge indices.
                # Key is (v, e1, e2).
                
                for f in inc:
                    if f == e:
                        continue
                    key1 = (v, e, f)
                    key2 = (v, f, e)
                    if key1 in var_map:
                        coeffs[var_map[key1]] = 1.0
                    elif key2 in var_map:
                        coeffs[var_map[key2]] = 1.0
                
                if np.any(coeffs > 0):
                    A_rows.append(coeffs)
                    b_lower.append(0)
                    b_upper.append(1)
        
        if not A_rows:
            A = np.array([]).reshape(0, num_vars)
        else:
            A = np.array(A_rows)
        
        constraints = LinearConstraint(A, b_lower, b_upper)
        bounds = Bounds(0, 1)
        
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(num_vars))
        
        if not res.success:
            # Fallback: simple greedy or individual edges
            return [[u, v] for u, v in edges]
        
        x = res.x
        
        # Reconstruct connections
        # connected[v][e] = f if x_{v, e, f} == 1, else None
        connected = {v: {e: None for e in incident[v]} for v in range(n)}
        
        for (v, e1, e2), idx in var_map.items():
            if x[idx] > 0.5:
                connected[v][e1] = e2
                connected[v][e2] = e1
        
        # Build paths
        # Start from edges that are dangling at one end.
        # An edge e=(u,v) is dangling at u if connected[u][e] is None.
        # An edge e=(u,v) is dangling at v if connected[v][e] is None.
        
        # We need to traverse the graph.
        # Since it's a set of paths and cycles, we can find endpoints.
        
        endpoints = []
        for i, (u, v) in enumerate(edges):
            # Check if e is dangling at u
            if connected[u][i] is None:
                endpoints.append((u, i, v)) # Start at u, edge i, next is v
            # Check if e is dangling at v
            if connected[v][i] is None:
                endpoints.append((v, i, u)) # Start at v, edge i, next is u
        
        paths = []
        visited_edges = set()
        
        # Traverse from endpoints
        for start_v, start_e, next_v in endpoints:
            if start_e in visited_edges:
                continue
            
            path = [start_v]
            curr_v = start_v
            curr_e = start_e
            
            while curr_e is not None:
                path.append(next_v)
                visited_edges.add(curr_e)
                
                # Move to next edge
                # At next_v, which edge is connected to curr_e?
                next_e = connected[next_v][curr_e]
                
                if next_e is None:
                    # End of path
                    break
                else:
                    # Determine the other vertex of next_e
                    u2, v2 = edges[next_e]
                    if u2 == next_v:
                        next_v = v2
                    else:
                        next_v = u2
                    curr_e = next_e
                    curr_v = next_v
                    # Note: The loop variable next_v is updated, but we need to be careful
                    # The 'next_v' in the tuple was the destination of the previous edge.
                    # Let's restructure the loop to be safer.
            
            # The above loop logic is slightly flawed with variable reuse. Let's rewrite traversal.
            
        # Better Traversal:
        paths = []
        visited_edges = set()
        
        # Find all start nodes (dangling ends)
        starts = []
        for i, (u, v) in enumerate(edges):
            if connected[u][i] is None:
                starts.append((u, i))
            if connected[v][i] is None:
                starts.append((v, i))
        
        for start_v, start_e in starts:
            if start_e in visited_edges:
                continue
                
            path = [start_v]
            curr_v = start_v
            curr_e = start_e
            
            while True:
                # Add the other end of curr_e
                u, v = edges[curr_e]
                if u == curr_v:
                    next_v = v
                else:
                    next_v = u
                
                path.append(next_v)
                visited_edges.add(curr_e)
                
                # Find next edge from next_v
                next_e = connected[next_v][curr_e]
                
                if next_e is None:
                    break
                else:
                    curr_v = next_v
                    curr_e = next_e
            
            paths.append(path)
            
        # Check for cycles (edges not visited)
        # If there are unvisited edges, they form cycles.
        # We need to break them into paths.
        for i, (u, v) in enumerate(edges):
            if i not in visited_edges:
                # Found a cycle
                path = []
                curr_v = u
                curr_e = i
                start_v = u
                start_e = i
                
                while True:
                    u, v = edges[curr_e]
                    if u == curr_v:
                        next_v = v
                    else:
                        next_v = u
                    
                    path.append(next_v)
                    visited_edges.add(curr_e)
                    
                    next_e = connected[next_v][curr_e]
                    
                    if next_e == start_e:
                        # Closed the cycle
                        # Break it by stopping here?
                        # The path is next_v ... start_v.
                        # But start_v is already in path?
                        # Path is [v1, v2, ..., vk, v1]?
                        # No, path starts empty.
                        # Iteration 1: append v2.
                        # ...
                        # Iteration k: append v1 (which is start_v).
                        # So path is [v2, ..., v1].
                        # We need to remove the last element to make it a path?
                        # Or just keep it? A path is a sequence of vertices.
                        # If we have a cycle v1-v2-v3-v1, the path can be v1-v2-v3.
                        # The edge v3-v1 is not used? No, all edges must be covered.
                        # If we break the cycle, we make v1 and v3 endpoints?
                        # Wait, if we just output [v2, v3, v1], the edges are (v2,v3) and (v3,v1).
                        # Edge (v1,v2) is missing?
                        # In the cycle traversal:
                        # Start at u (v1), edge i (v1,v2).
                        # Loop:
                        # curr_v = v1, curr_e = (v1,v2). next_v = v2. path=[v2].
                        # next_e = connected[v2][(v1,v2)] = (v2,v3).
                        # curr_v = v2, curr_e = (v2,v3). next_v = v3. path=[v2, v3].
                        # next_e = connected[v3][(v2,v3)] = (v3,v1).
                        # curr_v = v3, curr_e = (v3,v1). next_v = v1. path=[v2, v3, v1].
                        # next_e = connected[v1][(v3,v1)] = (v1,v2) (start_e).
                        # Loop ends.
                        # Path is [v2, v3, v1]. Edges used: (v2,v3), (v3,v1).
                        # Edge (v1,v2) is in visited_edges but not in path edges?
                        # No, the path [v2, v3, v1] uses edges (v2,v3) and (v3,v1).
                        # The edge (v1,v2) was the first edge, but we didn't add v1 to the path initially.
                        # We should start path with [start_v].
                        
                        # Let's fix the cycle path construction.
                        break
                    
                    curr_v = next_v
                    curr_e = next_e
                
                # Construct path from cycle
                # The cycle is start_v - ... - start_v.
                # We can represent it as a path by removing one edge?
                # No, we must use all edges.
                # A cycle cannot be a single path without repeating a vertex.
                # But the problem requires *simple* paths.
                # So a cycle must be broken into a path by "cutting" it.
                # Cutting a cycle at an edge (u,v) makes u and v endpoints.
                # The path is u - ... - v.
                # In our traversal, we ended at start_v.
                # The path constructed is [next_v_1, ..., start_v].
                # The edge (start_v, next_v_1) is the first edge.
                # So the path is [start_v, next_v_1, ..., start_v]? No, that repeats start_v.
                # Simple path cannot repeat vertices.
                # So we must drop the last vertex?
                # If we drop the last vertex, we drop the last edge.
                # But we must cover all edges.
                # This implies a single cycle cannot be a single simple path.
                # It must be decomposed into... wait.
                # A cycle of length k has k edges.
                # A simple path has k-1 edges for k vertices.
                # So a cycle of k edges cannot be covered by a single simple path.
                # It must be covered by... 1 path?
                # No, Gallai's conjecture allows paths.
                # If the graph is a cycle, the minimum number of paths is 1?
                # No, a cycle $C_n$ can be covered by 1 path? No.
                # A path is a sequence of distinct vertices.
                # A cycle visits all vertices.
                # To cover all edges of a cycle with simple paths, you need 1 path?
                # No, you can't traverse the whole cycle without repeating the start vertex.
                # So you must break it.
                # If you break it, you have a path that covers $k-1$ edges.
                # The remaining 1 edge is a path of length 1.
                # So 2 paths?
                # Or, if you allow the path to be "open", you just can't close it.
                # So a cycle requires 1 path if we allow the path to be a cycle?
                # But "simple path" usually means no repeated vertices.
                # So a cycle is NOT a simple path.
                # Therefore, a cycle must be decomposed into at least 1 path?
                # Actually, if you have a cycle, you can pick any edge and remove it, leaving a path.
                # But you can't remove edges.
                # So you must cover the cycle edges with multiple paths?
                # No, you can cover a cycle with 1 path if you "break" the cycle in the decomposition.
                # Wait, if I have edges (1,2), (2,3), (3,1).
                # Path 1: [1, 2, 3]. Edges (1,2), (2,3).
                # Path 2: [1, 3]. Edge (3,1).
                # Total 2 paths.
                # So a cycle of length k requires 1 path if we can use a path that visits k vertices and uses k-1 edges?
                # No, we must use ALL edges.
                # So a cycle of k edges requires at least 1 path?
                # No, it requires 1 path if the path can be a cycle? No.
                # It requires 1 path if we interpret "path" loosely?
                # No, "simple path" is strict.
                # So a cycle of k edges requires 1 path of length k-1 and 1 path of length 1?
                # That's 2 paths.
                # Or, can we do it with 1 path?
                # No.
                # So each cycle adds 1 to the path count compared to the "dangling" count?
                # Dangling count for a cycle is 0.
                # Actual paths: 2?
                # Wait, if we have a triangle, min paths is 2?
                # Let's check Gallai for a cycle.
                # $\Delta = 2$. $\lceil 2/2 \rceil = 1$?
                # But we need 2 paths.
                # So the bound is not tight for cycles?
                # Or maybe I can form 1 path?
                # No.
                
                # So, for each cycle, we need to output 2 paths?
                # Or 1 path if we break it?
                # If we break it, we have 1 long path and 1 short edge.
                # So +1 path per cycle.
                
                # Let's just extract the cycle vertices and create two paths.
                # Cycle vertices: start_v, v1, ..., vk-1, start_v.
                # Path 1: [start_v, v1, ..., vk-1]
                # Path 2: [start_v, vk-1] ? No, edge (vk-1, start_v) is used in Path 1?
                # Path 1 uses edges (start_v, v1), ..., (vk-2, vk-1).
                # Path 2 uses edge (vk-1, start_v).
                # So Path 2 is [vk-1, start_v].
                
                # We need to reconstruct the cycle order.
                # My traversal gave path = [v2, ..., v1].
                # start_v = v1.
                # So cycle is v1 - v2 - ... - vk - v1.
                # Path 1: [v1, v2, ..., vk]
                # Path 2: [vk, v1]
                
                # Let's re-run traversal to get the full cycle list.
                cycle_vertices = [start_v]
                curr_v = start_v
                curr_e = start_e
                while True:
                    u, v = edges[curr_e]
                    if u == curr_v:
                        next_v = v
                    else:
                        next_v = u
                    cycle_vertices.append(next_v)
                    visited_edges.add(curr_e)
                    next_e = connected[next_v][curr_e]
                    if next_e == start_e:
                        break
                    curr_v = next_v
                    curr_e = next_e
                
                # cycle_vertices is [v1, v2, ..., vk, v1]
                # Remove last v1
                cycle_vertices.pop()
                # Path 1: cycle_vertices
                # Path 2: [cycle_vertices[-1], cycle_vertices[0]]
                paths.append(cycle_vertices)
                paths.append([cycle_vertices[-1], cycle_vertices[0]])

        return paths

    res1 = solve_path_cover(instance(20, 0.3, 1), 20)
    res2 = solve_path_cover(instance(25, 0.3, 2), 25)
    res3 = solve_path_cover(instance(30, 0.25, 3), 30)
    
    return [res1, res2, res3]

# EVOLVE-BLOCK-START
def construct():
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_path_cover(edges, n):
        if not edges:
            return []
        
        edge_to_idx = {}
        for i, (u, v) in enumerate(edges):
            edge_to_idx[(u, v)] = i
            edge_to_idx[(v, u)] = i
            
        incident = {v: [] for v in range(n)}
        for i, (u, v) in enumerate(edges):
            incident[u].append(i)
            incident[v].append(i)
            
        var_indices = []
        var_map = {}
        for v in range(n):
            inc = incident[v]
            for i in range(len(inc)):
                for j in range(i + 1, len(inc)):
                    e1 = inc[i]
                    e2 = inc[j]
                    idx = len(var_indices)
                    var_indices.append(idx)
                    var_map[(v, e1, e2)] = idx
        
        num_vars = len(var_indices)
        if num_vars == 0:
            return [[u, v] for u, v in edges]

        c = np.full(num_vars, -1.0)
        
        A_rows = []
        b_lower = []
        b_upper = []
        
        for v in range(n):
            inc = incident[v]
            for e in inc:
                coeffs = np.zeros(num_vars)
                for f in inc:
                    if f == e:
                        continue
                    key1 = (v, e, f)
                    key2 = (v, f, e)
                    if key1 in var_map:
                        coeffs[var_map[key1]] = 1.0
                    elif key2 in var_map:
                        coeffs[var_map[key2]] = 1.0
                
                if np.any(coeffs > 0):
                    A_rows.append(coeffs)
                    b_lower.append(0)
                    b_upper.append(1)
        
        if not A_rows:
            A = np.array([]).reshape(0, num_vars)
        else:
            A = np.array(A_rows)
        
        constraints = LinearConstraint(A, b_lower, b_upper)
        bounds = Bounds(0, 1)
        
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(num_vars))
        
        if not res.success:
            return [[u, v] for u, v in edges]
        
        x = res.x
        
        connected = {v: {e: None for e in incident[v]} for v in range(n)}
        
        for (v, e1, e2), idx in var_map.items():
            if x[idx] > 0.5:
                connected[v][e1] = e2
                connected[v][e2] = e1
        
        paths = []
        visited_edges = set()
        
        starts = []
        for i, (u, v) in enumerate(edges):
            if connected[u][i] is None:
                starts.append((u, i))
            if connected[v][i] is None:
                starts.append((v, i))
        
        for start_v, start_e in starts:
            if start_e in visited_edges:
                continue
                
            path = [start_v]
            curr_v = start_v
            curr_e = start_e
            
            while True:
                u, v = edges[curr_e]
                if u == curr_v:
                    next_v = v
                else:
                    next_v = u
                
                path.append(next_v)
                visited_edges.add(curr_e)
                
                next_e = connected[next_v][curr_e]
                
                if next_e is None:
                    break
                else:
                    curr_v = next_v
                    curr_e = next_e
            
            paths.append(path)
            
        for i, (u, v) in enumerate(edges):
            if i not in visited_edges:
                cycle_vertices = [u]
                curr_v = u
                curr_e = i
                while True:
                    u2, v2 = edges[curr_e]
                    if u2 == curr_v:
                        next_v = v2
                    else:
                        next_v = u2
                    cycle_vertices.append(next_v)
                    visited_edges.add(curr_e)
                    next_e = connected[next_v][curr_e]
                    if next_e == i:
                        break
                    curr_v = next_v
                    curr_e = next_e
                
                cycle_vertices.pop()
                paths.append(cycle_vertices)
                paths.append([cycle_vertices[-1], cycle_vertices[0]])

        return paths

    res1 = solve_path_cover(instance(20, 0.3, 1), 20)
    res2 = solve_path_cover(instance(25, 0.3, 2), 25)
    res3 = solve_path_cover(instance(30, 0.25, 3), 30)
    
    return [res1, res2, res3]
# EVOLVE-BLOCK-END
