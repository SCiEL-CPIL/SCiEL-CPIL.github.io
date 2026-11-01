import time
import random
import sys
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Fallback: LUW construction
    def luw_construct():
        q = k
        d_candidates = []
        for d in range(3, 20):
            if d % 2 == 1:
                if d + 5 >= g:
                    d_candidates.append(d)
            else:
                if d + 4 >= g:
                    d_candidates.append(d)
        if not d_candidates:
            d = max(3, g - 5)
            if d % 2 == 0:
                d += 1
        else:
            d = min(d_candidates)
            
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
            rule[c] = rules_cycle[r]
            
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side), tuple(y)
                
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    # Try to find a smaller graph using local search for k=4, g=10
    if k == 4 and g == 10:
        n_target = 320
        search_deadline = start_time + 30.0
        
        def is_valid_graph(n, edges):
            degree = [0] * n
            for u, v in edges:
                degree[u] += 1
                degree[v] += 1
            if any(d != k for d in degree):
                return False
            # Check girth
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            for start_v in range(n):
                visited = [[-1] * n for _ in range((g-1)//2 + 1)]
                visited[0][start_v] = -2
                queue = deque([(start_v, 0)])
                while queue:
                    v, depth = queue.popleft()
                    if depth == (g-1)//2:
                        continue
                    for w in adj[v]:
                        if visited[depth+1][w] == -1:
                            visited[depth+1][w] = v
                            queue.append((w, depth+1))
                        elif w != v and visited[depth+1][w] != -2:
                            # Cycle found of length depth+1
                            if depth + 1 < g:
                                return False
            return True

        def generate_random_4_reg(n):
            # Wormald-Havel-Hakimi style or simple random pairing
            # Simple: Create 2n half-edges, pair them randomly
            half_edges = [i for i in range(n) for _ in range(2)]
            random.shuffle(half_edges)
            edges = []
            for i in range(n):
                u = half_edges[2*i]
                v = half_edges[2*i+1]
                if u != v:
                    edges.append((u, v))
            # Check for multi-edges and loops
            # This simple method might produce multi-edges.
            # For robustness, we might need a better generator.
            # But for local search, we can start with a valid one.
            # Let's use a circulant graph as a starting point?
            # Circulant on Z_320 with steps {1, 2, 3, 4}? Girth is small.
            # Let's just use the random pairing and check validity.
            return edges

        def local_search(n, edges, deadline):
            # Convert to set for fast lookup
            edge_set = set((min(u,v), max(u,v)) for u,v in edges)
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            def get_short_cycles():
                cycles = []
                for start_v in range(n):
                    visited = [[-1] * n for _ in range(5)] # Depth 4 is enough for girth 10
                    visited[0][start_v] = -2
                    queue = deque([(start_v, 0)])
                    while queue:
                        v, depth = queue.popleft()
                        if depth == 4:
                            continue
                        for w in adj[v]:
                            if visited[depth+1][w] == -1:
                                visited[depth+1][w] = v
                                queue.append((w, depth+1))
                            elif w != v and visited[depth+1][w] != -2:
                                # Cycle of length depth+1
                                if depth + 1 < 10:
                                    cycles.append((start_v, w, depth+1))
                return cycles

            best_edges = edges[:]
            best_cost = float('inf')
            
            while time.time() < deadline:
                cycles = get_short_cycles()
                if not cycles:
                    return best_edges
                # Pick a random cycle
                s, w, l = random.choice(cycles)
                # Find an edge to swap
                # This is heuristic and complex. 
                # For simplicity, we might skip this if it's too slow to implement correctly.
                # Given the risk, I will rely on the LUW fallback.
                break
                
            return None

        # Attempt local search
        # Due to complexity and time, I will skip the full local search implementation
        # and rely on the LUW graph, which is valid.
        # However, I can try a few random restarts with a very simple heuristic.
        
        # Actually, the LUW graph is 8192.
        # If I can find a graph with n=400, that's a huge improvement.
        # But implementing a correct local search for girth is error-prone.
        
        # I will return the LUW graph to ensure validity.
        return luw_construct()

    return luw_construct()
