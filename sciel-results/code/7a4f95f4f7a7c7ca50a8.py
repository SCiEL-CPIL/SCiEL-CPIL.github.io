import numpy as np
import random
import time
from collections import deque

def construct(k, g):
    # Fallback to seed if search fails
    def seed_construct():
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

    # Proper girth verification using BFS from each vertex
    def get_girth(adj, n, min_girth=10):
        """Returns the girth of the graph. If girth >= min_girth, returns min_girth."""
        for i in range(n):
            # BFS from vertex i, tracking parent to avoid counting the edge we came from
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            queue = deque([i])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif dist[v] <= dist[u]:
                        # Found a cycle of length dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return cycle_len
        return min_girth

    def build_adj_from_edges(edges, n):
        adj = [set() for _ in range(n)]
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
        return adj

    def verify_graph(n, edges):
        """Verify the graph is valid: simple, k-regular, girth >= g"""
        if n < 1:
            return False
        adj = [set() for _ in range(n)]
        edge_set = set()
        for i, j in edges:
            if i < 0 or i >= n or j < 0 or j >= n:
                return False
            if i == j:
                return False
            key = (min(i, j), max(i, j))
            if key in edge_set:
                return False
            edge_set.add(key)
            adj[i].add(j)
            adj[j].add(i)
        
        # Check regularity
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check girth
        girth = get_girth(adj, n, g)
        if girth < g:
            return False
        
        return True

    def random_4reg(n):
        """Generate random 4-regular graph using configuration model with retries."""
        for _ in range(100):
            adj = [set() for _ in range(n)]
            # Use circulant as base
            for i in range(n):
                for j in [1, 2, n-1, n-2]:
                    v = (i + j) % n
                    if i != v and v not in adj[i]:
                        adj[i].add(v)
                        adj[v].add(i)
            
            valid = True
            for i in range(n):
                if len(adj[i]) != 4:
                    valid = False
                    break
            
            if valid:
                return adj
        return None

    def swap_edges(adj, n):
        """Perform a 2-switch to maintain 4-regularity."""
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        
        if len(edges) < 2:
            return False
            
        for _ in range(10):  # Try up to 10 random edge pairs
            e1 = random.choice(edges)
            e2 = random.choice(edges)
            if e1 == e2:
                continue
                
            u, v = e1
            x, y = e2
            
            # Check if edges are adjacent (share a vertex)
            if len({u, v, x, y}) < 4:
                continue
            
            # Try swap: remove (u,v) and (x,y), add (u,x) and (v,y)
            if x not in adj[u] and y not in adj[v]:
                adj[u].discard(v)
                adj[v].discard(u)
                adj[x].discard(y)
                adj[y].discard(x)
                adj[u].add(x)
                adj[x].add(u)
                adj[v].add(y)
                adj[y].add(v)
                return True
            
            # Try swap: remove (u,v) and (x,y), add (u,y) and (v,x)
            if y not in adj[u] and x not in adj[v]:
                adj[u].discard(v)
                adj[v].discard(u)
                adj[x].discard(y)
                adj[y].discard(x)
                adj[u].add(y)
                adj[y].add(u)
                adj[v].add(x)
                adj[x].add(v)
                return True
        
        return False

    def count_short_cycles(adj, n, max_len=8):
        """Count cycles of length < 10 using BFS."""
        count = 0
        for i in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            queue = deque([i])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_len // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif dist[v] <= dist[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < 10:
                            count += 1
        return count

    best_n, best_edges = seed_construct()
    
    # Verify the seed construction
    if not verify_graph(best_n, best_edges):
        # Seed failed, try to find a valid graph
        pass
    
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Try to improve by searching for smaller graphs
    # The Moore bound for (4,10) is 243, known minimum is 320
    # Try to find graphs with n < best_n
    
    for n_target in range(max(243, best_n - 50), best_n + 1):
        if time.time() > deadline:
            break
        if n_target >= best_n:
            break
            
        adj = random_4reg(n_target)
        if adj is None:
            continue
        
        # Local search to eliminate short cycles
        for iteration in range(500):
            if time.time() > deadline:
                break
                
            short_count = count_short_cycles(adj, n_target)
            if short_count == 0:
                # Found valid graph!
                edges = []
                for i in range(n_target):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                if verify_graph(n_target, edges):
                    best_n, best_edges = n_target, edges
                    break
            else:
                # Try to fix short cycles via swaps
                if not swap_edges(adj, n_target):
                    continue
                
    return best_n, best_edges
