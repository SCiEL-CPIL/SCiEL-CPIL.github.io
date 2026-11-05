import time
import sys
from collections import deque
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before the 150s kill

    # --- Helper: Check girth using BFS from every vertex (optimized) ---
    def check_girth_fast(adj, n, g):
        half_g = g // 2
        max_depth_allowed = (g - 1) // 2
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if dist[u] <= max_depth_allowed and dist[v] <= max_depth_allowed:
                            if parent[u] == v or parent[v] == u:
                                continue
                            return False
        return True

    # --- Fallback: LUW construction (Baseline) ---
    def construct_luw(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Get baseline
    baseline_n, baseline_edges = construct_luw(k, g)
    best_n = baseline_n
    best_edges = baseline_edges

    # --- Strategy: Deterministic Ramanujan-like Construction via Cayley Graphs ---
    # Instead of random voltage lifting, we use a specific algebraic structure.
    # For k=3, the graph is a Cayley graph of a group G with 3 generators {a, b, c} 
    # such that a, b, c are involutions (a^2 = b^2 = c^2 = e) or similar.
    # Actually, for 3-regular, we can use generators {x, y, z} where x, y, z are involutions.
    # Or we can use a specific known construction for (3, 16) cages.
    
    # A known approach is to use the "Cayley graph of the triangle group" quotient.
    # The (2,3,7) triangle group has generators x, y with x^2 = y^3 = (xy)^7 = e.
    # This is 3-regular? No, x and y are 2 generators.
    # For 3-regular, we need 3 generators or a different setup.
    
    # Let's try a different structural approach: 
    # Construct a graph by gluing together copies of a "cage-like" base structure.
    # Or, use a specific known small (3,16) cage if one is hardcoded? 
    # The task says "smallest known such graph has 936 vertices". 
    # The baseline LUW construction likely produces something around that size or larger.
    # If we can find a specific construction that yields < 936, we win.
    
    # Let's try a "voltage graph" approach but with a smarter base graph.
    # Base graph: The K4 (complete graph on 4 vertices) is 3-regular.
    # Lift K4 with group Z_m.
    # K4 has 6 edges.
    # Lifted graph has 4*m vertices.
    # If m=200, n=800. This is < 936.
    # Let's check if we can find voltages for K4 lift with girth 16.
    
    if k == 3:
        # Base graph: K4
        # Vertices: 0, 1, 2, 3
        # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        base_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        n_base = 4
        n_base_edges = len(base_edges)
        
        def build_lift(base_edges, m, voltages):
            n = n_base * m
            adj = [[] for _ in range(n)]
            
            for idx, (u, v) in enumerate(base_edges):
                vol = voltages[idx] % m
                for i in range(m):
                    u_lift = u * m + i
                    v_lift = v * m + (i + vol) % m
                    adj[u_lift].append(v_lift)
                    adj[v_lift].append(u_lift)
            
            return adj, n

        # We want n = 4*m < best_n.
        # So m < best_n // 4.
        max_m = best_n // 4
        if max_m < 50:
            max_m = 50
            
        search_deadline = deadline - 5.0
        
        # Try to find a valid voltage assignment for K4 lift.
        # We iterate m from small to large? No, we want smallest n.
        # So we iterate m from 1 upwards? But small m might not have enough "room" for girth 16.
        # For girth 16, the ball of radius 7 must be a tree.
        # In the lift, the ball of radius 7 in the base graph (which is a tree of depth 7 in the universal cover)
        # must not close up.
        # The universal cover of K4 is a binary tree? No, K4 has cycles of length 3.
        # The girth of the lift is determined by the shortest non-zero sum of voltages along a closed walk in the base.
        # Specifically, any closed walk in the base with voltage sum 0 mod m creates a cycle in the lift.
        # The length of the cycle in the lift is the length of the walk in the base.
        # So we need: for every closed walk W in K4 of length L < 16, the voltage sum is NOT 0 mod m.
        
        # Closed walks in K4 of length < 16:
        # Length 2: back-and-forth. Voltage v + (-v) = 0. This is always 0.
        # Wait, the lift is a simple graph. The edge (u, i) - (v, i+v) is a single edge.
        # A walk of length 2 in the lift corresponds to u -> v -> u.
        # In the base, this is u -> v -> u.
        # Voltage sum: vol(u,v) + vol(v,u). Since we define vol(v,u) = -vol(u,v) (for undirected edges in lift),
        # the sum is 0.
        # So the lift ALWAYS has cycles of length 2?
        # No, a cycle of length 2 in a simple graph is a double edge.
        # The lift of a simple graph with a voltage assignment is a simple graph IF AND ONLY IF
        # for every pair of vertices u, v in the base, and every i, j, there is at most one edge.
        # If (u,v) is an edge in base, it lifts to (u,i) - (v, i+vol).
        # If there is another edge (u,v) in base (multigraph), it would create a parallel edge.
        # But K4 is simple. So the lift is simple.
        # However, the "cycle of length 2" in the walk sense is just traversing an edge back and forth.
        # That is not a cycle in the graph theory sense (simple cycle).
        # A simple cycle of length L in the lift corresponds to a closed walk of length L in the base
        # that does not immediately backtrack, AND whose voltage sum is 0 mod m.
        
        # So we need to check all "non-backtracking" closed walks in K4 of length L < 16.
        # And ensure their voltage sum is not 0 mod m.
        
        # Let's generate all non-backtracking closed walks in K4 up to length 15.
        # K4 is small, so this is feasible.
        
        def get_non_backtracking_closed_walks(base_adj, max_len):
            # base_adj: dict of neighbors
            walks = []
            # Start from each vertex, do DFS up to max_len, avoiding immediate backtracking
            for start in range(n_base):
                # Path: [start, ...]
                # Last vertex, prev vertex (to avoid backtracking)
                stack = [(start, -1, [start])]
                while stack:
                    curr, prev, path = stack.pop()
                    if len(path) > max_len:
                        continue
                    if len(path) >= 3 and path[-1] == start:
                        # Found a closed walk
                        # Check if it's non-backtracking (already enforced by stack)
                        # Store the walk (sequence of vertices)
                        # We need the sequence of edges to compute voltage sum.
                        walks.append(path)
                    for next_v in base_adj[curr]:
                        if next_v != prev:
                            stack.append((next_v, curr, path + [next_v]))
            return walks

        # Build adjacency for K4
        k4_adj = {i: [] for i in range(4)}
        for u, v in base_edges:
            k4_adj[u].append(v)
            k4_adj[v].append(u)
            
        # Precompute walks
        walks = get_non_backtracking_closed_walks(k4_adj, 15)
        
        # For each walk, we have a sequence of vertices [v0, v1, ..., vk] where v0 = vk.
        # The voltage sum is sum of vol(v_i, v_{i+1}) for i=0..k-1.
        # Note: vol(u,v) is defined for the edge (u,v) in base_edges.
        # We need to map (u,v) to an index in base_edges to get the voltage variable.
        
        edge_to_idx = {}
        for idx, (u, v) in enumerate(base_edges):
            edge_to_idx[(u, v)] = idx
            edge_to_idx[(v, u)] = idx # Undirected
            
        # For a walk [v0, v1, ..., vk], the voltage sum is:
        # sum_{i=0}^{k-1} s_i * vol(edge_i)
        # where s_i is +1 if we traverse in the direction of base_edges[idx], -1 if opposite.
        # Let's define the voltage variable x_idx for edge idx.
        # If we traverse (u,v) and base_edges[idx] is (u,v), contribution is +x_idx.
        # If we traverse (v,u) and base_edges[idx] is (u,v), contribution is -x_idx.
        
        # Let's represent each walk as a linear equation: sum(c_i * x_i) = 0 mod m.
        # We want to choose x_i such that NONE of these sums is 0 mod m.
        
        equations = []
        for path in walks:
            coeffs = [0] * n_base_edges
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i+1]
                idx = edge_to_idx[(u, v)]
                # Determine sign
                bu, bv = base_edges[idx]
                if u == bu and v == bv:
                    coeffs[idx] += 1
                else:
                    coeffs[idx] -= 1
            # Skip trivial equations like x1 - x1 = 0 (which is always 0)
            # Actually, if all coeffs are 0, the sum is always 0, so this walk creates a cycle of length L in the lift for ANY m.
            # This means the lift ALWAYS has a cycle of length L.
            # If L < 16, we fail.
            if all(c == 0 for c in coeffs):
                if len(path) - 1 < 16:
                    # This walk creates a short cycle unconditionally.
                    # So K4 lift cannot have girth 16?
                    # Let's check.
                    # A walk of length 3 in K4: 0-1-2-0.
                    # Coeffs: x(0,1) + x(1,2) + x(2,0).
                    # This is not identically 0.
                    # A walk of length 2: 0-1-0.
                    # Coeffs: x(0,1) - x(0,1) = 0.
                    # Length 2 < 16.
                    # So the K4 lift ALWAYS has cycles of length 2?
                    # No, as discussed, length 2 walks are not simple cycles.
                    # My `get_non_backtracking_closed_walks` function should not return length 2 walks if it avoids backtracking?
                    # Path: [0, 1, 0].
                    # Start 0, prev -1. Next 1. Stack: (1, 0, [0,1]).
                    # From 1, prev 0. Neighbors of 1: 0, 2, 3.
                    # Next 0 is prev, so skip.
                    # Next 2. Stack: (2, 1, [0,1,2]).
                    # Next 3. Stack: (3, 1, [0,1,3]).
                    # So [0,1,0] is NOT generated. Good.
                    # What about length 3? [0,1,2,0].
                    # From 2, prev 1. Neighbors: 0, 1, 3.
                    # Next 0 is not prev (1). So [0,1,2,0] is generated.
                    # Coeffs: x(0,1) + x(1,2) - x(0,2) = 0.
                    # This is a valid equation.
                    
                    pass # This block is for unconditional short cycles. If any exist, K4 lift is invalid.
                    # Since length 2 is excluded by non-backtracking, we are safe from length 2.
                    # Are there any other walks with identically 0 coeffs?
                    # Only if the walk traverses edges in a way that cancels out perfectly, which implies backtracking or repeating the same edge in opposite direction.
                    # Non-backtracking prevents immediate reversal.
                    # So we are likely safe.
                    continue
            
            equations.append((coeffs, len(path) - 1))
            
        # Now, for each m, we need to find x_0..x_5 such that for all (coeffs, L) in equations:
        # sum(c_i * x_i) != 0 mod m.
        
        # This is a constraint satisfaction problem.
        # We can try random assignments for x_i.
        
        for m in range(100, max_m + 1):
            if time.time() > search_deadline:
                break
                
            found = False
            attempts = 20
            for _ in range(attempts):
                if time.time() > search_deadline:
                    break
                    
                x = [random.randint(1, m-1) for _ in range(n_base_edges)]
                valid = True
                for coeffs, L in equations:
                    s = sum(c * xi for c, xi in zip(coeffs, x)) % m
                    if s == 0:
                        valid = False
                        break
                if valid:
                    # Found a valid assignment
                    adj, n = build_lift(base_edges, m, x)
                    if check_girth_fast(adj, n, g):
                        if n < best_n:
                            edges = []
                            for u in range(n):
                                for v in adj[u]:
                                    if u < v:
                                        edges.append((u, v))
                            best_n = n
                            best_edges = edges
                            found = True
                            break
            if found:
                break # Found a valid graph for this m, and m is increasing, so this is the smallest m found?
                # Wait, we are iterating m from 100 upwards.
                # If we find one at m=100, n=400.
                # If we didn't find one at m=50 (which we didn't try), maybe m=50 works?
                # We should iterate m from small to large.
                # But small m is harder to satisfy.
                # Let's iterate m from 50 to max_m.
                
    # If we didn't find a K4 lift, we return the baseline.
    return best_n, best_edges
