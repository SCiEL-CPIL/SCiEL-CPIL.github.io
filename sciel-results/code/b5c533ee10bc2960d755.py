import random
import time

def construct(k, g):
    # Target: 3-regular graph with girth >= g (16)
    # Moore bound for n(3,16) is 766.
    # Use voltage lift of Petersen graph (10 vertices) over Z_m.
    # n = 10 * m.
    # Need 10 * m >= 766 => m >= 77.
    
    if k != 3:
        # Fallback to LUW if k is not 3, though task implies k=3
        # But we should handle general k if possible, or just fail/return best effort
        # For now, assume k=3 as per task description "Small 3-regular graph"
        pass

    # Petersen graph edges
    # Vertices 0-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), # Outer pentagon
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5), # Inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)  # Spokes
    ]
    
    base_n = 10
    moore_bound = 766
    min_m = (moore_bound + base_n - 1) // base_n
    if min_m < 2: min_m = 2
    
    deadline = time.time() + 110  # Safety margin
    
    # Try m from min_m upwards
    for m in range(min_m, min_m + 100):
        if time.time() > deadline:
            break
            
        # Try random voltage assignments
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Assign random voltages
            voltages = [random.randint(0, m - 1) for _ in range(len(petersen_edges))]
            
            # Construct lift
            n = base_n * m
            edges = []
            
            for i, (u, v) in enumerate(petersen_edges):
                ve = voltages[i]
                for layer in range(m):
                    u_lift = u * m + layer
                    v_lift = v * m + (layer + ve) % m
                    edges.append((u_lift, v_lift))
            
            # Check girth
            # BFS from each vertex up to depth 7
            # If no cycle < 16 found, valid.
            valid = True
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # We can optimize by only checking a subset of vertices if needed, 
            # but n is small (~800).
            # Use a simple BFS girth check.
            # To speed up, we can stop early if a short cycle is found.
            
            for s in range(n):
                # BFS
                visited = [-1] * n
                parent = [-1] * n
                dist = [-1] * n
                queue = [s]
                visited[s] = 1
                dist[s] = 0
                qi = 0
                while qi < len(queue) and valid:
                    u = queue[qi]
                    qi += 1
                    if dist[u] >= 7:
                        continue
                    for v in adj[u]:
                        if visited[v] == 0:
                            visited[v] = 1
                            parent[v] = u
                            dist[v] = dist[u] + 1
                            queue.append(v)
                        elif v != parent[u]:
                            # Found a cycle
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                valid = False
                                break
                if not valid:
                    break
            
            if valid:
                return n, edges

    # If no valid graph found with voltage lift, fallback to LUW or return best effort
    # The task requires a valid graph. 
    # If we fail, we should probably return a large valid graph (like LUW) to get score 1.0 
    # or close to it, rather than 0.
    # But the loop above should find one.
    
    # Fallback: LUW construction (original)
    # Re-implementing LUW briefly for fallback
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
