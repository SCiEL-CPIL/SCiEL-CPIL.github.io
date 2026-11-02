import random
import time
import sys
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses a local search heuristic to find a small cage.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # For k=3, g=16, we expect N around 1000-2000
    # Moore bound is 512.
    
    def is_valid_graph(n, edges, g):
        # Check if graph is k-regular and has girth >= g
        # We assume edges are a set of (u, v) with u < v
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check girth using BFS from each vertex
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            found_cycle = False
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
            # Note: BFS from each vertex finds the shortest cycle containing that vertex.
            # If no cycle < g is found from any vertex, girth >= g.
        return True

    def generate_random_3_regular(n):
        # Use configuration model
        if n % 2 != 0:
            return None
        half_edges = n * k // 2
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        edges = set()
        adj = [set() for _ in range(n)]
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None
            if v in adj[u]:
                return None
            edges.add((min(u, v), max(u, v)))
            adj[u].add(v)
            adj[v].add(u)
        return list(edges), adj

    def find_short_cycle(adj, g):
        # Find a cycle of length < g
        n = len(adj)
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            # Reconstruct cycle
                            path1 = []
                            path2 = []
                            cur = u
                            while cur != start:
                                path1.append(cur)
                                cur = parent[cur]
                            path1.append(start)
                            cur = v
                            while cur != start:
                                path2.append(cur)
                                cur = parent[cur]
                            path2.append(start)
                            # Combine
                            cycle = path1[::-1] + path2[1:]
                            return cycle
        return None

    def swap_edges(adj, edges, u1, v1, u2, v2):
        # Swap (u1, v1) and (u2, v2) to (u1, v2) and (u2, v1)
        # Must ensure no multi-edges or loops
        if v2 in adj[u1] or v1 in adj[u2]:
            return False
        if u1 == v2 or u2 == v1:
            return False
        # Remove old
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        edges.discard((min(u1, v1), max(u1, v1)))
        edges.discard((min(u2, v2), max(u2, v2)))
        # Add new
        adj[u1].add(v2)
        adj[v2].add(u1)
        adj[u2].add(v1)
        adj[v1].add(u2)
        edges.add((min(u1, v2), max(u1, v2)))
        edges.add((min(u2, v1), max(u2, v1)))
        return True

    best_n = None
    best_edges = None
    
    # Try increasing N
    for n in range(512, 2000, 32):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random instances
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
                
            result = generate_random_3_regular(n)
            if result is None:
                continue
            
            edges, adj = result
            if not is_valid_graph(n, edges, g):
                # Try to repair
                for step in range(1000):
                    if time.time() > deadline - 5:
                        break
                    cycle = find_short_cycle(adj, g)
                    if cycle is None:
                        # Found a valid graph
                        if best_n is None or n < best_n:
                            best_n = n
                            best_edges = list(edges)
                        break
                    else:
                        # Pick two edges in the cycle to swap
                        # Simple heuristic: pick random edges
                        edge_indices = [(i, (cycle[i], cycle[(i+1)%len(cycle)])) for i in range(len(cycle))]
                        random.shuffle(edge_indices)
                        swapped = False
                        for i in range(min(10, len(edge_indices))):
                            u1, v1 = edge_indices[i][1]
                            u2, v2 = edge_indices[(i+1)%len(edge_indices)][1]
                            if swap_edges(adj, edges, u1, v1, u2, v2):
                                swapped = True
                                break
                        if not swapped:
                            break
            else:
                # Already valid
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = list(edges)
                break

    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: LUW construction
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
