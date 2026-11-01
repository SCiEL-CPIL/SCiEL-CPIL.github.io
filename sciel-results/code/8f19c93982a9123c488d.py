import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, 10s margin
    
    q = k
    
    def build_luw(d):
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
        index = {start: 0}
        order = [start]
        edges = []
        
        i = 0
        while i < len(order):
            if time.time() > deadline:
                return None
            vertex = order[i]
            i += 1
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    def check_girth(n, edges, target_girth):
        # Check if girth >= target_girth
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # We need to ensure no cycle < target_girth
        # Since graph is bipartite, girth is even.
        # If target_girth is 9, we need no cycles of length 2, 4, 6, 8.
        # So we check for cycles of length <= target_girth - 1.
        # Max depth for BFS to detect cycle of length L is L/2.
        # We want to detect cycles < target_girth.
        # So max cycle length to detect is target_girth - 1.
        # If target_girth is odd, target_girth-1 is even.
        # If target_girth is even, target_girth-1 is odd, but cycles are even, so we check up to target_girth - 2.
        
        max_cycle_len = target_girth - 1 if target_girth % 2 == 1 else target_girth - 2
        if max_cycle_len < 4:
            return True
            
        max_depth = max_cycle_len // 2
        
        # For each vertex, do BFS up to max_depth
        # If we find a back-edge to a node at depth d' such that depth + 1 + d' <= max_cycle_len, we have a short cycle?
        # Actually, standard BFS girth check:
        # For each start node, BFS.
        # If we encounter a visited node (not parent), we have a cycle.
        # The length of the cycle is depth + 1 + dist(visited_node, start)?
        # No, that's not correct if the paths merge.
        # However, if we just want to check if girth < G, we can do:
        # For each node, BFS up to depth G/2.
        # If we visit a node already visited in this BFS tree, and the sum of depths + 1 < G, we have a short cycle.
        # Actually, simpler:
        # Just run BFS. If we find a back edge, the cycle length is at least depth + 1 + dist(neighbor, start) is NOT correct.
        # Correct way:
        # For each start, BFS. Keep track of (node, depth, parent).
        # If neighbor is visited and neighbor != parent:
        #   cycle_len = depth + 1 + visited_depth[neighbor]
        #   This is the length of the cycle formed by the two paths from start.
        #   If cycle_len < target_girth, return False.
        #   This detects cycles that go through start.
        #   Does it detect all cycles?
        #   Any cycle contains a start node. So yes, if we iterate all start nodes, we will find every cycle.
        
        for start_node in range(n):
            visited = {start_node: (None, 0)}
            queue = [(start_node, None, 0)]
            while queue:
                node, parent, depth = queue.pop(0)
                if depth >= max_depth:
                    continue
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor in visited:
                        # Cycle found
                        # Length is depth + 1 + visited[neighbor][1]
                        # This is the length of the cycle formed by paths from start_node
                        cycle_len = depth + 1 + visited[neighbor][1]
                        if cycle_len < target_girth:
                            return False
                    else:
                        visited[neighbor] = (node, depth + 1)
                        queue.append((neighbor, node, depth + 1))
        return True

    # Estimate minimal d
    # Girth of CD(d, q) is approx 2d+2
    # We need girth >= g
    # If g is odd, we need girth >= g+1 (since girth is even)
    # So 2d+2 >= g+1 => 2d >= g-1 => d >= (g-1)/2
    # If g is even, we need girth >= g
    # So 2d+2 >= g => 2d >= g-2 => d >= (g-2)/2
    
    if g % 2 == 1:
        d_min = (g - 1) // 2
    else:
        d_min = (g - 2) // 2
        
    d_min = max(3, d_min)
    
    best_n = float('inf')
    best_edges = None
    
    # Try d from d_min upwards
    # Limit d to avoid explosion
    max_d = d_min + 5 
    
    for d in range(d_min, max_d + 1):
        if time.time() > deadline:
            break
            
        result = build_luw(d)
        if result is None:
            break
            
        n, edges = result
        
        # Check girth
        if check_girth(n, edges, g):
            if n < best_n:
                best_n = n
                best_edges = edges
            # If we found a valid graph with minimal d, we can stop?
            # Larger d gives larger n. So minimal d is best.
            break
            
    if best_edges is None:
        # Fallback to original logic if something went wrong
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        result = build_luw(d)
        if result:
            return result
        return 0, []
        
    return best_n, best_edges
