import time
import random
from collections import defaultdict
import heapq

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose_once(edges, n, rng, deadline):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    remaining = len(edges)
    
    while remaining > 0:
        if time.time() > deadline:
            break
        
        # Find a long cycle using greedy randomized strategy
        best_cycle = None
        best_len = 0
        
        # Try starting from each vertex
        for start in range(n):
            if not adj[start]:
                continue
            
            for _ in range(20): # Increased restarts
                path = [start]
                visited = {start}
                current = start
                
                while True:
                    neighbors = list(adj[current])
                    unvisited = [v for v in neighbors if v not in visited]
                    
                    if not unvisited:
                        if len(path) >= 3 and start in adj[current]:
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = path[:]
                        break
                    else:
                        # Heuristic: Pick neighbor with highest degree to encourage longer cycles
                        # Use rng to break ties or randomize slightly
                        max_deg = -1
                        chosen = None
                        # Add some randomness to avoid getting stuck in local optima
                        if rng.random() < 0.2:
                            chosen = rng.choice(unvisited)
                        else:
                            for v in unvisited:
                                deg = len(adj[v])
                                if deg > max_deg:
                                    max_deg = deg
                                    chosen = v
                            
                            if chosen is None:
                                chosen = rng.choice(unvisited)
                                
                        path.append(chosen)
                        visited.add(chosen)
                        current = chosen
        
        if best_cycle:
            m = len(best_cycle)
            for i in range(m):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(best_cycle)
        else:
            edge_removed = False
            for u in range(n):
                for v in list(adj[u]):
                    if u < v:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        remaining -= 1
                        pieces.append([u, v])
                        edge_removed = True
                        break
                if edge_removed:
                    break
            if not edge_removed:
                break
    
    return pieces

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    
    num_attempts = 60 # Increased attempts
    base_seed = 42
    
    for i in range(num_attempts):
        if time.time() > deadline - 0.5:
            break
        
        rng = random.Random(base_seed + i)
        pieces = _decompose_once(edges, n, rng, deadline)
        cnt = len(pieces)
        
        if cnt < best_count:
            best_count = cnt
            best = pieces
    
    return best

def construct():
    start_time = time.time()
    total_budget = 140.0
    per_graph_budget = total_budget / 3.0
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        
        graph_deadline = start_time + per_graph_budget * (i + 1)
        if graph_deadline > start_time + total_budget - 3.0:
            graph_deadline = start_time + total_budget - 3.0
        
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
    
    return results
