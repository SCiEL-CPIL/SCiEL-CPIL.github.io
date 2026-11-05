import random
import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Fallback: LUW graph with minimal d for girth g
    # Girth of CD(d, q) is 2d+2.
    # We need 2d+2 >= g => d >= (g-2)/2.
    # For g=16, d >= 7.
    q = k
    d_min = (g - 2) // 2
    if d_min < 3: d_min = 3
    
    def build_luw(d):
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

    best_n, best_edges = build_luw(d_min)
    
    # Try to find a smaller graph using local search
    # Start from N = 512 (Moore bound) up to best_n
    # We prioritize N close to the known best (936) if 512 fails quickly
    # But let's try a range.
    
    def generate_random_3_regular(n):
        if n % 2 != 0 or n < 4:
            return None
        # Use random perfect matching on 2(n-1) edges?
        # Standard method:
        # 1. Create 2n half-edges.
        # 2. Randomly pair them.
        # 3. If loops or multi-edges, retry.
        # For n=936, this is fast.
        while True:
            half_edges = list(range(n)) + list(range(n))
            random.shuffle(half_edges)
            edges = []
            seen = set()
            valid = True
            for i in range(0, 2 * n, 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    valid = False
                    break
                edge = (min(u, v), max(u, v))
                if edge in seen:
                    valid = False
                    break
                seen.add(edge)
                edges.append(edge)
            if valid:
                return edges
    
    def check_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex, stop at depth g//2
        # If we find a back-edge to a vertex at depth < current depth - 1, we have a cycle
        # Actually, simpler: BFS from each vertex, track parent.
        # If we reach a visited node that is not the parent, we found a cycle.
        # The length is depth[u] + depth[v] + 1.
        # We want to know if any cycle is < g.
        
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            queue = deque([(start, 0)])
            visited[start] = 0
            while queue:
                u, depth = queue.popleft()
                if depth >= g // 2:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = depth + 1
                        parent[v] = u
                        queue.append((v, depth + 1))
                    elif v != parent[u]:
                        # Found a cycle
                        # Length is depth[u] + depth[v] + 1
                        # This is a lower bound on the girth if it's the first one found?
                        # Not necessarily, but if any is < g, we return False.
                        cycle_len = depth + visited[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def find_shortest_cycle(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        min_len = float('inf')
        min_cycle = None
        for start in range(n):
            visited = [-1] * n
            parent = [-1] * n
            queue = deque([(start, 0)])
            visited[start] = 0
            while queue:
                u, depth = queue.popleft()
                if depth >= min_len // 2:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = depth + 1
                        parent[v] = u
                        queue.append((v, depth + 1))
                    elif v != parent[u]:
                        cycle_len = depth + visited[v] + 1
                        if cycle_len < min_len:
                            min_len = cycle_len
                            # Reconstruct cycle
                            cycle = []
                            u1, u2 = u, v
                            while u1 != u2:
                                cycle.append(u1)
                                u1 = parent[u1]
                                if u1 == -1: break
                            # This reconstruction is tricky with BFS tree
                            # For now, just return length
        return min_len

    # Local search
    def local_search(n, edges, target_girth, time_limit):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check initial girth
        if check_girth(n, edges, target_girth):
            return edges
        
        end_time = time.time() + time_limit
        while time.time() < end_time:
            # Find a short cycle
            # This is expensive. We can just pick a random edge and try to swap it.
            # If the swap reduces the number of short cycles, keep it.
            # Simpler: Pick random edge, swap with random other edge.
            # Check if girth increased.
            
            e1 = random.choice(edges)
            e2 = random.choice(edges)
            if e1 == e2: continue
            
            u1, v1 = e1
            u2, v2 = e2
            
            # Ensure no loops or multi-edges
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
                
            # Try two swaps
            # 1. (u1, u2) and (v1, v2)
            # 2. (u1, v2) and (v1, u2)
            
            for swap in [( (u1, u2), (v1, v2) ), ( (u1, v2), (v1, u2) )]:
                ne1, ne2 = swap
                # Check if ne1 or ne2 exist in graph
                # This is slow.
                pass
        
        return None

    # Given the complexity and time, the LUW graph is a safe bet.
    # We will return the LUW graph with d_min.
    
    return best_n, best_edges
