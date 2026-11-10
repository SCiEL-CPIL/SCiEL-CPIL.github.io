import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    time_limit = 115
    
    target_N = 1116
    
    def generate_random_regular_graph(N, k):
        # Configuration Model
        stubs = []
        for i in range(N):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        
        # Pair stubs
        edges = []
        for i in range(0, N * k, 2):
            edges.append((stubs[i], stubs[i+1]))
            
        # Build adjacency with counts
        adj = [set() for _ in range(N)]
        # We need to handle multi-edges.
        # Let's just build a list of edges and then resolve.
        
        # Resolve self-loops and multi-edges
        # This is a heuristic.
        # For simplicity, we will just generate a random simple k-regular graph using a different method:
        # 1. Start with a cycle.
        # 2. Add random edges to satisfy degree k.
        
        adj = [set() for _ in range(N)]
        # Base cycle
        for i in range(N):
            adj[i].add((i+1) % N)
            adj[(i+1) % N].add(i)
            
        # Add k-2 edges
        # We need to add (k-2)/2 pairs of edges? No, we need to increase degree by k-2.
        # Total degree deficit is N * (k-2).
        # We add edges one by one.
        
        # To ensure we don't get stuck, we can pick random u, v with deg < k.
        
        for _ in range(N * (k-2) // 2):
            # Pick u, v
            # We need to ensure we can complete the graph.
            # Greedy: pick u with min degree, v with min degree.
            # But random is better for girth.
            
            # Pick random u, v
            # Retry if invalid
            for _ in range(100):
                u = random.randint(0, N-1)
                v = random.randint(0, N-1)
                if u == v: continue
                if len(adj[u]) >= k: continue
                if len(adj[v]) >= k: continue
                if v in adj[u]: continue
                
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                break
            else:
                # Failed to find valid edge
                return None
        
        # Check degrees
        for i in range(N):
            if len(adj[i]) != k:
                return None
                
        return [list(s) for s in adj]

    def find_short_cycle(adj, g):
        N = len(adj)
        for s in range(N):
            depth = {s: 0}
            parent = {s: None}
            queue = deque([s])
            while queue:
                u = queue.popleft()
                if depth[u] >= g - 1:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if v in depth:
                        if depth[u] + depth[v] + 1 < g:
                            return [(u, v)]
                    else:
                        depth[v] = depth[u] + 1
                        parent[v] = u
                        queue.append(v)
        return None

    adj = generate_random_regular_graph(target_N, k)
    if adj is None:
        return None, None

    for _ in range(50000):
        if time.time() - start_time > time_limit:
            break
        cycle = find_short_cycle(adj, g)
        if cycle is None:
            break
        u, v = cycle[0]
        # Swap
        for _ in range(20):
            x = random.randint(0, target_N - 1)
            y = random.randint(0, target_N - 1)
            if x == y or x in (u, v) or y in (u, v): continue
            if y not in adj[x]: continue
            if x in adj[u] or y in adj[v]: continue
            
            # Swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            break

    if find_short_cycle(adj, g) is not None:
        return None, None

    edge_list = []
    for i in range(target_N):
        for j in adj[i]:
            if i < j:
                edge_list.append((i, j))
    return target_N, edge_list
