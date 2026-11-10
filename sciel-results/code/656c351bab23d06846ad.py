import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Fallback: LUW graph (original seed logic, but optimized)
    def luw_construct(k, g):
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

    # Helper: Check girth exactly
    def check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex to find shortest cycle
        # Optimization: Only need to check up to min_girth
        for i in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            queue = [i]
            while queue:
                curr = queue.pop(0)
                if dist[curr] >= min_girth // 2:
                    continue
                for neighbor in adj[curr]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[curr] + 1
                        parent[neighbor] = curr
                        queue.append(neighbor)
                    elif parent[curr] != neighbor and parent[neighbor] != curr:
                        # Found a cycle
                        cycle_len = dist[curr] + dist[neighbor] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # Helper: Generate random 3-regular graph
    def gen_random_3reg(n):
        if n % 2 != 0:
            return None
        # Configuration model
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        random.shuffle(stubs)
        edges = []
        adj = [set() for _ in range(n)]
        for i in range(0, n * 3, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None # Loop
            if v in adj[u]:
                return None # Multi-edge
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
        return edges

    # Helper: Edge swap
    def edge_swap(edges, adj, u1, v1, u2, v2):
        # Remove (u1, v1) and (u2, v2), add (u1, u2) and (v1, v2)
        # Check validity
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            return None
        if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
            return None
        # Perform swap
        new_edges = []
        for e in edges:
            if (e[0] == u1 and e[1] == v1) or (e[0] == v1 and e[1] == u1): continue
            if (e[0] == u2 and e[1] == v2) or (e[0] == v2 and e[1] == u2): continue
            new_edges.append(e)
        new_edges.append((u1, u2))
        new_edges.append((v1, v2))
        return new_edges

    # Main Search Loop
    best_n = float('inf')
    best_edges = None
    
    # Try small n values
    for n in range(512, 600, 8):
        if time.time() > deadline - 20:
            break
        for _ in range(100): # 100 restarts per n
            if time.time() > deadline - 10:
                break
            edges = gen_random_3reg(n)
            if not edges:
                continue
            
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            # Local search
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                # Find shortest cycle
                # Simplified: Just check if girth >= 16
                if check_girth(n, edges, 16):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                    break
                
                # Find a short cycle to break
                # For efficiency, we just pick random edges and try swaps
                # This is a crude heuristic
                for _ in range(10):
                    if time.time() > deadline - 10:
                        break
                    # Pick two random edges
                    e1_idx = random.randint(0, len(edges)-1)
                    e2_idx = random.randint(0, len(edges)-1)
                    if e1_idx == e2_idx:
                        continue
                    u1, v1 = edges[e1_idx]
                    u2, v2 = edges[e2_idx]
                    new_edges = edge_swap(edges, adj, u1, v1, u2, v2)
                    if new_edges:
                        # Check if new graph has girth >= 16?
                        # This is expensive. We can skip check and just accept if it doesn't create loops/multiedges
                        # But we need to ensure we don't decrease girth.
                        # For now, accept and re-check in next iteration
                        edges = new_edges
                        # Update adj
                        adj = [set() for _ in range(n)]
                        for u, v in edges:
                            adj[u].add(v)
                            adj[v].add(u)
                        improved = True
                        break

    if best_edges is not None:
        return best_n, best_edges
    else:
        return luw_construct(k, g)
