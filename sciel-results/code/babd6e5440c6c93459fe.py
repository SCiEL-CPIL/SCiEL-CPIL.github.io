import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = 120.0 # Safety margin
    
    # Fallback: LUW graph
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
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

    # Try to find a lift of K5 with girth g
    if k == 4 and g == 10:
        # Lift of K5 by Z2^6
        # Base graph K5
        base_edges = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
        num_base_vertices = 5
        group_order = 64
        total_vertices = num_base_vertices * group_order
        
        # Volatile voltages
        # Each voltage is an integer 0..63 representing Z2^6
        # We assign a voltage to each edge in base_edges
        # voltages[i] corresponds to base_edges[i]
        
        def check_girth(voltages):
            # BFS from every vertex to find shortest cycle
            # Since graph is 4-regular, we can use non-backtracking walks
            # Or just standard BFS with parent tracking
            # For speed, we can check if any cycle of length < g exists
            # Using a set of visited (vertex, prev_vertex) to avoid backtracking
            
            # Build adjacency list for the lift
            # adj[v] = list of (neighbor, edge_id_in_base?)
            # Actually, we just need neighbors.
            # Vertex v = u * group_order + g_val
            # u = v // group_order
            # g_val = v % group_order
            # Neighbors of (u, g_val):
            # For each edge (u, v_base) in base_edges:
            #   neighbor_u = v_base
            #   neighbor_g = (g_val + voltages[edge_idx]) % group_order
            #   neighbor_vertex = neighbor_u * group_order + neighbor_g
            
            # Precompute base adjacency with voltages
            # base_adj[u] = list of (v_base, voltage)
            base_adj = [[] for _ in range(num_base_vertices)]
            for idx, (u, v) in enumerate(base_edges):
                w = voltages[idx]
                base_adj[u].append((v, w))
                base_adj[v].append((u, w)) # Undirected, same voltage (Z2)
            
            # Check girth
            # We can do BFS from each vertex up to depth g/2 - 1
            # If we meet a vertex at depth <= g/2 - 1 from another path, cycle exists
            # Actually, simpler: 
            # For each start vertex, do BFS.
            # Keep track of (vertex, depth, parent)
            # If we reach a vertex already visited with a different parent, 
            # and depth1 + depth2 + 1 < g, then cycle.
            
            # Optimization: Since we just need to know if girth >= 10,
            # we can check for cycles of length 4, 6, 8.
            # If none, girth >= 10 (for 4-regular, next possible is 10? No, 10 is even. 
            # Cycles are even in bipartite graphs, but this is not necessarily bipartite.
            # K5 is not bipartite. So odd cycles are possible.
            # So we must check all lengths < 10.
            
            # This is expensive if done naively for 320 vertices.
            # 320 vertices * (4^5) walks = 320 * 1024 = 327k steps.
            # This is fast enough for a single check.
            
            # Let's implement a fast girth check.
            # We want to know if there is a cycle of length < g.
            
            # We can use the "non-backtracking" property.
            # A cycle of length L corresponds to a non-backtracking walk of length L that returns to start.
            
            # For each vertex v:
            #   For each neighbor u:
            #     Walk from u, not going back to v, for L-1 steps.
            #     If we reach v, we have a cycle of length L.
            
            # This is still 320 * 4 * 4^(L-2).
            # For L=9, 320 * 4 * 4^7 = 320 * 4 * 16384 = 20M ops.
            # This might be too slow for many iterations.
            
            # Alternative: 
            # Use the fact that we are doing a search.
            # We can just check for cycles of length 4, 6, 8.
            # If none, assume girth >= 10? 
            # No, there could be a 5-cycle or 7-cycle or 9-cycle.
            # But K5 has triangles. Lifts might have odd cycles.
            
            # Let's try a simpler heuristic:
            # Randomly sample walks of length 9 from each vertex.
            # If we find a cycle, return False.
            # If we don't, return True.
            # This is probabilistic.
            
            # Better: 
            # Just do BFS up to depth 4 from each vertex.
            # If any vertex is reached by two different paths of length <= 4, 
            # then there is a cycle of length <= 4+4+1 = 9?
            # If two paths of length d1 and d2 meet at a vertex, cycle length is d1+d2+1?
            # No, if they meet at a common neighbor, cycle is d1+d2.
            
            # Let's just implement a standard BFS for shortest cycle for one start vertex, 
            # and do it for a few random start vertices?
            # No, must be certified.
            
            # Given time constraints, I will implement a check for cycles of length < 10 
            # using a bitset or simple array for visited states (vertex, prev).
            
            # State: (current_vertex, previous_vertex)
            # Start states: (u, v) for all edges (u,v).
            # Step: (u, v) -> (v, w) where w is neighbor of v, w != u.
            # If we reach (v, u), we have a cycle of length equal to the depth.
            
            # We can do this for all start states simultaneously using BFS layers.
            # Layer 1: all directed edges (u,v).
            # Layer 2: all non-backtracking walks of length 2.
            # ...
            # Layer L: all non-backtracking walks of length L.
            # If any walk in Layer L ends at the start vertex of a walk in Layer 0?
            # No, a cycle of length L is a walk of length L that returns to start.
            
            # So, for each start vertex s:
            #   Do BFS in the "line graph" (states are directed edges) 
            #   starting from all neighbors of s.
            #   If we reach a state (s, u) at depth L, we have a cycle of length L+1.
            
            # We want to know if there is ANY cycle of length < 10.
            # So we run BFS for all s simultaneously?
            # No, s is different.
            
            # Let's just do it for all s.
            # 320 BFSs.
            # Each BFS explores up to 4^5 = 1024 states.
            # Total 320k states.
            # This is very fast in Python if optimized?
            # 320 * 1024 = 327,680.
            # Yes, this is fine.
            
            # Implementation:
            # adj[v] = list of neighbors
            # For each s in range(total_vertices):
            #   visited = set() # (vertex, prev)
            #   queue = deque()
            #   for u in adj[s]:
            #       queue.append((u, s))
            #       visited.add((u, s))
            #   depth = 1
            #   while queue and depth < g:
            #       next_queue = []
            #       for v, prev in queue:
            #           for w in adj[v]:
            #               if w == prev: continue
            #               if w == s:
            #                   return False # Found cycle of length depth+1 < g
            #               if (w, v) not in visited:
            #                   visited.add((w, v))
            #                   next_queue.append((w, v))
            #       queue = next_queue
            #       depth += 1
            # return True

            # Build adj for the lift
            adj = [[] for _ in range(total_vertices)]
            for u in range(num_base_vertices):
                for v, w in base_adj[u]:
                    for g_val in range(group_order):
                        src = u * group_order + g_val
                        dst = v * group_order + ((g_val + w) % group_order)
                        adj[src].append(dst)
            
            # Check girth
            from collections import deque
            for s in range(total_vertices):
                visited = set()
                queue = deque()
                for u in adj[s]:
                    queue.append((u, s))
                    visited.add((u, s))
                depth = 1
                while queue and depth < g:
                    next_queue = []
                    for v, prev in queue:
                        for w in adj[v]:
                            if w == prev:
                                continue
                            if w == s:
                                return False
                            if (w, v) not in visited:
                                visited.add((w, v))
                                next_queue.append((w, v))
                    queue = deque(next_queue)
                    depth += 1
            return True

        # Search loop
        best_voltages = None
        for i in range(1000):
            if time.time() > deadline:
                break
            voltages = [random.randint(0, 63) for _ in range(10)]
            if check_girth(voltages):
                best_voltages = voltages
                break
            # Optional: Local search
            # ...
        
        if best_voltages:
            # Construct edges
            edges = []
            for idx, (u, v) in enumerate(base_edges):
                w = best_voltages[idx]
                for g_val in range(group_order):
                    src = u * group_order + g_val
                    dst = v * group_order + ((g_val + w) % group_order)
                    if src < dst:
                        edges.append((src, dst))
            return total_vertices, edges

    # Fallback to LUW
    return luw_construct(k, g)
