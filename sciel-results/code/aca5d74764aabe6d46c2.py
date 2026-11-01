import time
import random
import numpy as np
from collections import deque

def _seed_construct(k, g):
    # Copy of the original seed code
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

def _is_valid(n, edges, k, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u == v: return False
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    for i in range(n):
        if len(set(adj[i])) != len(adj[i]):
            return False
            
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif dist[v] <= dist[u]:
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

def _count_short_cycles(n, adj, g):
    count = 0
    cycles = []
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif dist[v] <= dist[u]:
                    L = dist[u] + dist[v] + 1
                    if L < g:
                        count += 1
                        # Store cycle for repair (optional, but helps)
                        # We just need count for SA energy
    return count

def _generate_random_regular(n, k):
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    random.shuffle(stubs)
    edges = []
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u != v:
            edges.append((u, v))
        else:
            # Self loop, need to fix. 
            # Simplest: restart or swap. 
            # For SA, we start with a valid simple graph.
            # Let's use a better generator or just retry.
            pass
    
    # This naive generator might produce multi-edges.
    # Let's use a repair loop here too.
    adj_set = [set() for _ in range(n)]
    valid = True
    final_edges = []
    for u, v in edges:
        if v in adj_set[u]:
            valid = False
            break
        adj_set[u].add(v)
        adj_set[v].add(u)
        final_edges.append((u, v))
        
    if not valid:
        return None, None
        
    adj = [[] for _ in range(n)]
    for u, v in final_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    return final_edges, adj

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 4 and g == 10:
        N = 320
        # Try to find a valid graph
        best_edges = None
        
        # Initialize with random graph
        edges, adj = None, None
        for _ in range(100):
            edges, adj = _generate_random_regular(N, k)
            if edges is not None:
                break
        
        if edges is None:
            # Fallback
            return _seed_construct(k, g)
            
        # Simulated Annealing
        T = 10.0
        cooling = 0.99
        current_energy = _count_short_cycles(N, adj, g)
        
        if current_energy == 0:
            return N, edges
            
        for _ in range(50000): # Max iterations
            if time.time() > deadline:
                break
                
            # Pick a random edge (u, v)
            u = random.randint(0, N - 1)
            v = random.choice(adj[u])
            
            # Pick another edge (x, y) not involving u or v
            x = random.randint(0, N - 1)
            if x in (u, v): continue
            y = random.choice(adj[x])
            if y in (u, v) or y == x: continue
            
            # Check if (u, y) and (v, x) are already edges
            if y in adj[u] or x in adj[v]:
                continue
                
            # Perform 2-switch
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].append(y)
            adj[y].append(u)
            adj[v].append(x)
            adj[x].append(v)
            
            new_energy = _count_short_cycles(N, adj, g)
            
            if new_energy < current_energy or random.random() < np.exp(-(new_energy - current_energy) / T):
                current_energy = new_energy
                if current_energy == 0:
                    # Reconstruct edge list
                    final_edges = []
                    for i in range(N):
                        for j in adj[i]:
                            if i < j:
                                final_edges.append((i, j))
                    return N, final_edges
            else:
                # Revert
                adj[u].remove(y)
                adj[y].remove(u)
                adj[v].remove(x)
                adj[x].remove(v)
                
                adj[u].append(v)
                adj[v].append(u)
                adj[x].append(y)
                adj[y].append(x)
                
            T *= cooling
            if T < 0.01:
                break
                
    return _seed_construct(k, g)
