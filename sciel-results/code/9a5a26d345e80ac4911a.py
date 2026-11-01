import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Specific handling for the (4,10) cage problem
    if k == 4 and g == 10:
        # Strategy: Search for a voltage graph lift of K5 by a group of order 64
        # to get a 320-vertex 4-regular graph with girth 10.
        # If successful, return n=320. Otherwise, fall back to LUW (n=8192).
        
        # K5 has 5 vertices. Edges: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        # Group: (Z2)^6, represented as integers 0..63
        # Voltage assignment: dict (u,v) -> int
        
        def check_girth_lift(voltages, group_order=64):
            # Build the lifted graph
            # Vertices are (base_vertex, group_element)
            # n_vertices = 5 * 64 = 320
            n_base = 5
            n_total = n_base * group_order
            adj = [[] for _ in range(n_total)]
            
            # Edges in K5
            edges_k5 = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
            
            for u, v in edges_k5:
                volt = voltages.get((u,v), 0)
                for g_elem in range(group_order):
                    u_id = u * group_order + g_elem
                    # Edge to v with voltage volt: (v, g_elem XOR volt)
                    # Since group is (Z2)^6, operation is XOR
                    v_id = v * group_order + (g_elem ^ volt)
                    adj[u_id].append(v_id)
                    # Undirected: also add reverse
                    # Reverse edge voltage is inverse. In (Z2)^6, inverse is self.
                    # So (v, g_elem ^ volt) -> (u, g_elem)
                    adj[v_id].append(u_id)
            
            # Check degree
            for i in range(n_total):
                if len(adj[i]) != 4:
                    return -1 # Invalid
            
            # Check girth using BFS from each vertex up to depth 4 (girth 10 means no cycle < 10)
            # Actually, for girth 10, we need to check for cycles of length 4, 6, 8.
            # BFS up to depth 4 from each vertex. If we find a neighbor at depth 2 or 3 that is also a neighbor at depth 2 or 3 via different path, cycle exists.
            # More efficiently: BFS from each vertex, keep track of parent.
            # If we visit a node already visited in the same BFS tree, and the distance sum < 10, fail.
            
            for start_v in range(n_total):
                if time.time() > deadline:
                    return -1
                visited = [-1] * n_total
                parent = [-1] * n_total
                depth = [0] * n_total
                queue = [start_v]
                visited[start_v] = start_v
                while queue:
                    v = queue.pop(0)
                    for w in adj[v]:
                        if visited[w] == -1:
                            visited[w] = w
                            parent[w] = v
                            depth[w] = depth[v] + 1
                            queue.append(w)
                        else:
                            # Found a cycle
                            # Length is depth[v] + depth[w] + 1
                            # But we must ensure it's a simple cycle and not just a back-edge
                            # If w is parent of v, it's the edge we came from.
                            if w == parent[v]:
                                continue
                            # If w is visited, we have a cycle.
                            # The length of the cycle is depth[v] + depth[w] + 1?
                            # Not exactly, depends on LCA.
                            # For a quick check, if depth[v] + depth[w] + 1 < 10, we have a short cycle.
                            # This is a sufficient condition for failure, but not necessary.
                            # However, for a 4-regular graph, if girth is >= 10, no such short cycles exist.
                            if depth[v] + depth[w] + 1 < 10:
                                return -1
            return 10 # Success
            
        # Random search for voltages
        best_girth = 0
        best_voltages = None
        
        # We can try a few random assignments
        # There are 10 edges, each with 64 voltages. 64^10 is huge.
        # But we can fix some voltages to 0 to reduce symmetry.
        # Set voltages of edges (0,1), (0,2), (0,3), (0,4) to 0.
        # Only randomize (1,2), (1,3), (1,4), (2,3), (2,4), (3,4).
        # 64^6 ~ 6.8e10. Too many for exhaustive.
        # Random sampling: try 1000 random assignments.
        
        for _ in range(1000):
            if time.time() > deadline - 20:
                break
            voltages = {}
            # Fix star edges at 0 to 0
            for v in [1, 2, 3, 4]:
                voltages[(0, v)] = 0
            # Randomize others
            for u, v in [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]:
                voltages[(u, v)] = random.randint(0, 63)
            
            girth = check_girth_lift(voltages)
            if girth >= 10:
                best_girth = girth
                best_voltages = voltages
                break # Found a valid one!
                
        if best_voltages is not None:
            # Construct the edge list
            n_total = 320
            edges = []
            edges_k5 = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
            for u, v in edges_k5:
                volt = best_voltages.get((u,v), 0)
                for g_elem in range(64):
                    u_id = u * 64 + g_elem
                    v_id = v * 64 + (g_elem ^ volt)
                    edges.append((u_id, v_id))
            return 320, edges
            
        # Fallback: LUW construction
    # LUW Construction for general k, g
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
