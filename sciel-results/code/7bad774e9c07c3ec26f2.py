import time
from random import Random
import sys

sys.setrecursionlimit(10000)

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(edges, time_limit):
    if not edges:
        return []
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    def build_adj(eset):
        adj = {}
        for u, v in eset:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj

    adj = build_adj(edges_set)
    
    def find_longest_cycle(adj, eset):
        best_cycle = None
        best_len = 0
        n_nodes = len(adj)
        start_time = time.time()
        timeout = False
        
        # Sort starts by degree descending?
        starts = sorted(adj.keys(), key=lambda x: len(adj[x]), reverse=True)
        
        for start in starts:
            if timeout:
                break
            if not adj[start]:
                continue
            
            def dfs(node, path, visited):
                nonlocal best_cycle, best_len, timeout
                
                if time.time() - start_time > time_limit:
                    timeout = True
                    return

                # Pruning
                if len(path) + (n_nodes - len(visited)) <= best_len:
                    return

                # Check close
                if start in adj.get(node, set()) and len(path) >= 3:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                
                for neighbor in adj.get(node, set()):
                    if timeout:
                        return
                    if neighbor == start:
                        continue
                    if neighbor not in visited:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, path, visited)
                        path.pop()
                        visited.remove(neighbor)
                        
            dfs(start, [start], {start})
            
        return best_cycle

    while edges_set:
        # Allow 10% of total time for each cycle extraction? 
        # Or just check if we have enough time left.
        if time.time() > time_limit - 1:
            break
            
        # Allocate a small chunk of time for the search, e.g., 1 second or 5% of remaining?
        # For N=30, it should be fast. Let's give it 2 seconds per cycle extraction.
        cycle = find_longest_cycle(adj, 2.0)
        
        if not cycle:
            break
        
        for i in range(len(cycle) - 1):
            u, v = cycle[i], cycle[i+1]
            edge = tuple(sorted((u, v)))
            edges_set.discard(edge)
            if v in adj.get(u, set()):
                adj[u].discard(v)
            if u in adj.get(v, set()):
                adj[v].discard(u)
            
            if u in adj and not adj[u]:
                del adj[u]
            if v in adj and not adj[v]:
                del adj[v]
        
        pieces.append(cycle)

    for e in sorted(edges_set):
        pieces.append([e[0], e[1]])
        
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 140
    results = []
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for i, spec in enumerate(specs):
        if time.time() > deadline - 5:
            # Not enough time, return trivial decomposition
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = _instance(*spec)
        
        # Estimate time remaining for this graph
        time_remaining = deadline - time.time() - 5 # 5s buffer
        # Allocate time for decomposition
        alloc_time = min(20, time_remaining / 3) # At least 3 graphs left? No, we are in loop.
        # Better: just give it a reasonable chunk, say 30s, but cap by remaining.
        alloc_time = min(30, time_remaining)
        
        pieces = _decompose(edges, alloc_time)
        results.append(pieces)
        
    return results
