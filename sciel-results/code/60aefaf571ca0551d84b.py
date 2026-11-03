import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Fallback: LUW Graph
    seed_n, seed_edges = luw_construct(k, g)
    
    best_n = seed_n
    best_edges = seed_edges
    
    # 2. Heuristic Search
    # We want to find a graph with n < best_n.
    # Known best is 320.
    # We will try to find a graph of size 320.
    
    # If k != 4 or g != 10, the logic might need adjustment, 
    # but the task is specific to this case.
    
    # Try a few candidate sizes.
    # Start with 320. If found, return.
    # If not, try 330, 340...
    # If we fail, return seed.
    
    candidate_sizes = [320, 310, 300, 350, 400]
    
    for n_target in candidate_sizes:
        if time.time() > deadline - 5:
            break
        if n_target >= best_n:
            continue
            
        # Allocate time for this size
        # If we have lots of time, give more.
        time_limit = min(30, deadline - time.time() - 5)
        if time_limit <= 0: break
        
        result = search_for_graph(k, g, n_target, time.time() + time_limit)
        if result is not None:
            best_n, best_edges = result
            # Found one. Can we do better?
            # Try smaller sizes if time permits?
            # For now, break and return to ensure we return a valid result.
            # 320 is likely the global min.
            break
            
    return best_n, best_edges

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

def search_for_graph(k, g, n, deadline):
    # Generate random k-regular graph
    # Using configuration model
    # Then repair if not simple?
    # Or use edge swaps from a random simple k-regular graph.
    
    # Simplest: Start with random edges, ensure degree k.
    # This is hard.
    # Better: Start with a known k-regular graph, e.g., circulant.
    # C_n(1, 2, ..., k/2)
    # For k=4, C_n(1, 2).
    
    # Generate C_n(1, 2)
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
        edges.append((i, (i + 2) % n))
    
    # Remove duplicates if any (for small n, 1 and 2 might overlap? No, n>=5)
    # Ensure simple
    edge_set = set()
    unique_edges = []
    for u, v in edges:
        if u > v: u, v = v, u
        if (u, v) not in edge_set:
            edge_set.add((u, v))
            unique_edges.append((u, v))
    edges = unique_edges
    
    # Check if it's k-regular
    # C_n(1, 2) is 4-regular for n > 4.
    
    # Simulated Annealing
    current_energy = calculate_girth(edges, n, g)
    best_energy = current_energy
    best_edges = edges if current_energy >= g else None
    
    T = 1.0
    while time.time() < deadline:
        # Random edge swap
        if len(edges) < 2: break
        i = random.randint(0, len(edges) - 1)
        j = random.randint(0, len(edges) - 1)
        if i == j: continue
        
        u1, v1 = edges[i]
        u2, v2 = edges[j]
        
        # Try swap: (u1, v2) and (v1, u2)
        new_u1, new_v1 = u1, v2
        new_u2, new_v2 = v1, u2
        
        # Check validity
        if new_u1 == new_v1 or new_u2 == new_v2: continue
        if new_u1 > new_v1: new_u1, new_v1 = new_v1, new_u1
        if new_u2 > new_v2: new_u2, new_v2 = new_v2, new_u2
        
        if new_u1 == new_u2 and new_v1 == new_v2: continue
        if (new_u1, new_v1) in edge_set and (new_u1, new_v1) != (u1, v1) and (new_u1, new_v1) != (u2, v2): continue
        if (new_u2, new_v2) in edge_set and (new_u2, new_v2) != (u1, v1) and (new_u2, new_v2) != (u2, v2): continue
        
        # Apply swap
        edge_set.remove((u1, v1))
        edge_set.remove((u2, v2))
        edge_set.add((new_u1, new_v1))
        edge_set.add((new_u2, new_v2))
        
        edges[i] = (new_u1, new_v1)
        edges[j] = (new_u2, new_v2)
        
        new_energy = calculate_girth(edges, n, g)
        
        if new_energy < best_energy:
            best_energy = new_energy
            if new_energy >= g:
                best_edges = edges[:]
                # Found valid graph
                return n, best_edges
            else:
                best_edges = edges[:]
        
        # Accept
        delta = new_energy - current_energy
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_energy = new_energy
        else:
            # Revert
            edge_set.remove((new_u1, new_v1))
            edge_set.remove((new_u2, new_v2))
            edge_set.add((u1, v1))
            edge_set.add((u2, v2))
            edges[i] = (u1, v1)
            edges[j] = (u2, v2)
            
        T *= 0.995
        if T < 0.01: T = 0.01
        
    if best_energy >= g:
        return n, best_edges
    return None, None

def calculate_girth(edges, n, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_girth = float('inf')
    
    for u in range(n):
        # BFS from u
        visited = [-1] * n
        parent = [-1] * n
        visited[u] = 0
        q = deque([u])
        
        while q:
            node = q.popleft()
            dist = visited[node]
            if dist > g // 2: break # Optimization
            
            for neighbor in adj[node]:
                if visited[neighbor] == -1:
                    visited[neighbor] = dist + 1
                    parent[neighbor] = node
                    q.append(neighbor)
                elif neighbor != parent[node]:
                    # Found a cycle
                    cycle_len = dist + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth >= g:
                        return min_girth
        if min_girth >= g:
            return min_girth
            
    return min_girth
