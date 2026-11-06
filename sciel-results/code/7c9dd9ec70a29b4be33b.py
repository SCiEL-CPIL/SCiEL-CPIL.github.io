import time
import random
from collections import defaultdict

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(adj, edges_set, time_limit=0.5):
        if not edges_set:
            return None
        
        best_cycle = None
        best_len = 0
        start_time = time.time()
        
        edges_list = list(edges_set)
        random.shuffle(edges_list)
        
        for u, v in edges_list:
            if time.time() - start_time > time_limit:
                break
                
            visited = {u}
            path = [u]
            removed_edge = (min(u, v), max(u, v))
            
            def dfs(node, target):
                nonlocal best_cycle, best_len
                if time.time() - start_time > time_limit:
                    return
                
                if node == target:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                    return
                
                # Heuristic: Try neighbors with higher degree first
                neighbors = list(adj[node])
                # Sort by degree descending, then shuffle to break ties randomly
                neighbors.sort(key=lambda x: len(adj[x]), reverse=True)
                random.shuffle(neighbors)
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        e = (min(node, neighbor), max(node, neighbor))
                        if e in edges_set and e != removed_edge:
                            visited.add(neighbor)
                            path.append(neighbor)
                            dfs(neighbor, target)
                            path.pop()
                            visited.remove(neighbor)
                
                return

            dfs(u, v)
            
        return best_cycle

    def decompose(edges):
        n = 0
        for u, v in edges:
            n = max(n, u, v)
        n += 1
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        edges_set = set(tuple(sorted(e)) for e in edges)
        pieces = []
        
        graph_time_limit = 40.0
        start_graph_time = time.time()
        
        while edges_set:
            if time.time() - start_graph_time > graph_time_limit:
                for e in edges_set:
                    pieces.append(list(e))
                break
            
            cycle = find_longest_cycle(adj, edges_set, time_limit=0.5)
            if cycle:
                pieces.append(cycle)
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i+1]
                    e = (min(u, v), max(u, v))
                    if e in edges_set:
                        edges_set.remove(e)
                        adj[u].remove(v)
                        adj[v].remove(u)
                u, v = cycle[-1], cycle[0]
                e = (min(u, v), max(u, v))
                if e in edges_set:
                    edges_set.remove(e)
                    adj[u].remove(v)
                    adj[v].remove(u)
            else:
                for e in edges_set:
                    pieces.append(list(e))
                break
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
        
    return results
