import random
import time
from collections import defaultdict
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_path_decomposition(n, edges):
        if not edges:
            return []
        
        # Map undirected edges to indices
        edge_map = {}
        for i, e in enumerate(edges):
            edge_map[e] = i
        
        # Build directed arcs
        # For each undirected edge {u, v}, we have two arcs (u, v) and (v, u)
        # Variable x[i] corresponds to arc i
        # Let's index arcs:
        # For edge index k (u, v), arc 2k is u->v, arc 2k+1 is v->u
        m = len(edges)
        num_arcs = 2 * m
        
        # Objective function
        # We want to minimize sum(p_v)
        # Variables: x_arcs (2m), p_v (n), n_v (n)
        # Total variables: 2m + 2n
        # c vector: 0 for x, 1 for p, 0 for n
        c = np.zeros(2 * m + 2 * n)
        c[2 * m : 2 * m + n] = 1.0
        
        # Constraints
        # 1. x_{uv} + x_{vu} = 1 for each edge
        # 2. b_v = sum(x_out) - sum(x_in)
        # 3. b_v - p_v + n_v = 0
        
        # A_eq matrix
        # Rows: m (edge constraints) + n (balance def) + n (p/n def)
        # Actually, we can combine balance and p/n.
        # b_v - p_v + n_v = 0  =>  sum(x_out) - sum(x_in) - p_v + n_v = 0
        
        num_constraints = m + n
        A_eq = np.zeros((num_constraints, 2 * m + 2 * n))
        b_eq = np.zeros(num_constraints)
        
        # Edge constraints
        for i, (u, v) in enumerate(edges):
            # Arc 2i: u->v, Arc 2i+1: v->u
            A_eq[i, 2 * i] = 1.0
            A_eq[i, 2 * i + 1] = 1.0
            b_eq[i] = 1.0
        
        # Balance constraints
        # For each vertex v, calculate contribution to b_v
        # b_v = sum_{u} x_{uv} - sum_{u} x_{vu}
        # We need to map arcs to vertices.
        # For edge i (u, v):
        # Arc 2i (u->v): contributes +1 to b_u, -1 to b_v
        # Arc 2i+1 (v->u): contributes +1 to b_v, -1 to b_u
        
        # Let's build the balance part of A_eq
        # Row index m + v corresponds to balance equation for vertex v
        for i, (u, v) in enumerate(edges):
            # Arc 2i: u->v
            A_eq[m + u, 2 * i] += 1.0
            A_eq[m + v, 2 * i] -= 1.0
            # Arc 2i+1: v->u
            A_eq[m + v, 2 * i + 1] += 1.0
            A_eq[m + u, 2 * i + 1] -= 1.0
            
        # Add -p_v and +n_v to the balance equation
        # Equation: sum(x_out) - sum(x_in) - p_v + n_v = 0
        for v in range(n):
            A_eq[m + v, 2 * m + v] = -1.0  # -p_v
            A_eq[m + v, 2 * m + n + v] = 1.0  # +n_v
        
        # Bounds
        # x_arcs are binary [0, 1]
        # p_v, n_v are non-negative [0, inf]
        bounds = []
        for _ in range(2 * m):
            bounds.append((0, 1))
        for _ in range(2 * n):
            bounds.append((0, None))
            
        # Solve
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not res.success:
            # Fallback to greedy if LP fails
            return greedy_decompose(n, edges)
            
        # Extract orientation
        # x_arcs are in res.x[0 : 2*m]
        x_arcs = res.x[:2 * m]
        
        # Build directed graph
        # out_edges[v] = list of (neighbor, edge_index)
        # in_edges[v] = list of (neighbor, edge_index)
        out_edges = defaultdict(list)
        in_edges = defaultdict(list)
        
        for i, (u, v) in enumerate(edges):
            # If x[2i] > 0.5, then u->v is selected
            if x_arcs[2 * i] > 0.5:
                out_edges[u].append(v)
                in_edges[v].append(u)
            else:
                # v->u is selected
                out_edges[v].append(u)
                in_edges[u].append(v)
                
        # Decompose into paths and cycles
        # We track visited edges to ensure we don't reuse them (though orientation is 1-1)
        # Actually, since each undirected edge is oriented exactly once,
        # and we just follow the directions, we don't need to track visited edges
        # if we remove them from the graph as we go?
        # No, the directed graph is fixed.
        # We just need to find the paths.
        
        # A vertex is a start if out_deg > in_deg
        # A vertex is an end if in_deg > out_deg
        # A vertex is internal if out_deg == in_deg > 0
        # A vertex is isolated if out_deg == in_deg == 0
        
        paths = []
        visited_edges = set() # To be safe, though not strictly necessary if we follow unique outgoing edges?
        # Wait, a vertex can have multiple outgoing edges if it's a start of multiple paths.
        # So we must pick one outgoing edge, follow it, and mark it used.
        # But the "flow" is fixed. If a vertex has out_deg=2, it means 2 paths start there.
        # We can just iterate through all vertices.
        
        # Let's maintain a list of available outgoing edges for each vertex.
        available_out = {v: list(neighs) for v, neighs in out_edges.items()}
        
        # Find starts
        starts = []
        for v in range(n):
            out_d = len(out_edges[v])
            in_d = len(in_edges[v])
            if out_d > in_d:
                # This vertex starts (out_d - in_d) paths
                starts.extend([v] * (out_d - in_d))
                
        # If no starts, we have only cycles.
        # We need to find cycles.
        
        # To handle cycles, we can just pick any unvisited edge and follow it until we loop.
        # But we already have the starts.
        # Let's process starts first.
        
        # We need a way to track which directed edges have been used in a path.
        # Since each directed edge is unique, we can just remove it from available_out.
        
        # Helper to find a path starting from 'start'
        def find_path(start):
            path = [start]
            current = start
            while True:
                if not available_out[current]:
                    # This shouldn't happen if we started from a valid start and followed flow
                    # unless we hit a dead end which is an end vertex
                    break
                # Pick an outgoing edge
                # Since the flow is fixed, any outgoing edge is valid for the flow decomposition?
                # Yes, the flow is a sum of paths.
                # We can just pick the first one.
                next_node = available_out[current].pop(0)
                path.append(next_node)
                current = next_node
                
                # Check if we've completed a path (reached an end)
                # An end vertex has in_deg > out_deg.
                # In the residual graph of available edges, if current has no outgoing edges,
                # it might be an end.
                # However, if it's an internal node, it should have outgoing edges.
                # If we loop back to a visited node in this path, it's a cycle.
                if current in path:
                    # Cycle found
                    # Extract cycle
                    cycle_start_idx = path.index(current)
                    cycle = path[cycle_start_idx:]
                    # The path part before the cycle is a path
                    # The cycle part is a cycle
                    # But wait, if we entered a cycle, the edge leading into it was part of the path.
                    # This logic is tricky.
                    
                    # Simpler approach for decomposition:
                    # Just follow edges.
                    # If we hit a vertex already in the current path, we have a cycle.
                    # We break the cycle.
                    return path, cycle
                if not available_out[current]:
                    # Reached an end
                    return path, None
            return path, None

        # This simple follow-up might fail if a vertex has multiple outgoing edges
        # and we pick one that leads to a cycle, while another leads to an end.
        # But the flow conservation ensures that the "excess" out-degree is balanced by "excess" in-degree.
        # So following any outgoing edge from a start should eventually lead to an end or a cycle.
        # If it leads to a cycle, we extract the cycle.
        
        # Let's refine the decomposition algorithm.
        # 1. Remove all cycles first?
        #    Find a cycle, remove it, convert to 2 paths.
        # 2. Then the remaining graph is a DAG of paths?
        #    No, it's a set of vertex-disjoint paths (if we remove cycles).
        
        # Algorithm:
        # 1. While there are edges:
        #    a. Find a cycle.
        #    b. If found, remove it, add 2 paths to result.
        #    c. If not found, find a path from a start to an end.
        #       Remove it, add 1 path to result.
        
        # How to find a cycle?
        # DFS on the directed graph.
        
        # Let's implement this robustly.
        
        # Re-initialize available_out
        available_out = {v: list(neighs) for v, neighs in out_edges.items()}
        
        # Function to find a cycle
        def find_cycle():
            # Try to find a cycle starting from any vertex with out-degree > 0
            for start in range(n):
                if not available_out[start]:
                    continue
                visited = {start}
                path = [start]
                stack = [(start, 0)]
                
                while stack:
                    node, idx = stack[-1]
                    if idx < len(available_out[node]):
                        next_node = available_out[node][idx]
                        stack[-1] = (node, idx + 1)
                        
                        if next_node == start:
                            # Found cycle
                            return path
                        if next_node not in visited:
                            visited.add(next_node)
                            path.append(next_node)
                            stack.append((next_node, 0))
                        # If next_node is in visited but not start, it's not a simple cycle from start
                        # We just skip it for this path
                    else:
                        stack.pop()
                        if len(stack) > 0:
                            path.pop()
            return None
            
        # Function to find a path from start to end
        def find_path_from(start):
            path = [start]
            current = start
            while available_out[current]:
                next_node = available_out[current].pop(0)
                path.append(next_node)
                current = next_node
            return path
            
        # Main loop
        while any(available_out.values()):
            if time.time() - start_time > time_limit - 5:
                break
                
            cycle = find_cycle()
            if cycle:
                # Remove cycle edges from available_out
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    # Remove v from available_out[u]
                    if v in available_out[u]:
                        available_out[u].remove(v)
                        
                # Convert cycle to 2 paths
                # Cycle: [v0, v1, ..., vk-1]
                # Split at middle
                k = len(cycle)
                mid = k // 2
                path1 = cycle[:mid + 1]
                path2 = cycle[mid:] + [cycle[0]]
                # Ensure simple
                # path1: v0...vmid
                # path2: vmid...vk-1, v0
                paths.append(path1)
                paths.append(path2)
            else:
                # No cycles, find a path
                # Pick a start
                start = None
                for v in range(n):
                    if available_out[v]:
                        # Check if it's a start (out > in in original? or just has out)
                        # In a graph with no cycles, all components are paths.
                        # Any vertex with out-degree > 0 can be a start of a path segment.
                        # But we want the full path from the very start.
                        # The "starts" are those with original out_deg > in_deg.
                        # However, after removing cycles, the degrees change.
                        # Let's just pick any vertex with available_out.
                        # If it's not the true start, we might get a sub-path.
                        # But we want to cover all edges.
                        # If we pick a middle vertex, we get a path ending at the end.
                        # The prefix is still available.
                        # So it's fine.
                        start = v
                        break
                        
                if start is None:
                    break
                    
                p = find_path_from(start)
                paths.append(p)
                
        return paths

    def greedy_decompose(n, edges):
        # Fallback if LP fails
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining = set(edges)
        paths = []
        
        while remaining:
            # Find a path
            # Pick an edge
            u, v = next(iter(remaining))
            remaining.discard((u, v))
            path = [u, v]
            # Extend
            while True:
                last = path[-1]
                next_node = None
                for nb in adj[last]:
                    e = tuple(sorted((last, nb)))
                    if e in remaining:
                        next_node = nb
                        break
                if next_node is None:
                    break
                remaining.discard(tuple(sorted((last, next_node))))
                path.append(next_node)
            paths.append(path)
            
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = solve_path_decomposition(n, edges)
        results.append(paths)
        
    return results
