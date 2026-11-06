import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]
    
    def decompose_graph(edges, time_limit):
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set(edges)
        pieces = []
        
        def remove_cycle_edges(cycle):
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i+1]
                if (u, v) in remaining_edges:
                    remaining_edges.remove((u, v))
                elif (v, u) in remaining_edges:
                    remaining_edges.remove((v, u))
            u, v = cycle[-1], cycle[0]
            if (u, v) in remaining_edges:
                remaining_edges.remove((u, v))
            elif (v, u) in remaining_edges:
                remaining_edges.remove((v, u))
            
            # Rebuild adjacency
            for u in adj:
                adj[u] = set()
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
        
        def find_best_cycle():
            """
            Find a cycle that maximizes the number of edges removed.
            Uses DFS with backtracking to find the longest cycle within time limits.
            """
            if not remaining_edges:
                return None
            
            # Identify candidate start vertices: those with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                return None
            
            # Sort candidates by degree (descending) to find large cycles faster
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            best_cycle = None
            best_len = 0
            
            # Time check inside the search
            search_deadline = time.time() + 0.3
            
            for start in candidates[:8]:
                if time.time() > search_deadline:
                    break
                
                # DFS to find cycles starting from 'start'
                # We want the longest cycle.
                # Path: [start, ..., current]
                # To close, current must connect to start.
                
                visited = {start}
                path = [start]
                
                def dfs(current, path):
                    nonlocal best_cycle, best_len, search_deadline
                    
                    if time.time() > search_deadline:
                        return
                    
                    # Check if we can close the cycle
                    if len(path) >= 3:
                        if (current, start) in remaining_edges or (start, current) in remaining_edges:
                            # Valid cycle found
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = list(path)
                    
                    # Extend path
                    for neighbor in adj[current]:
                        if neighbor == start:
                            # Only allowed if we are closing the cycle (handled above)
                            continue
                        if neighbor not in visited:
                            visited.add(neighbor)
                            path.append(neighbor)
                            dfs(neighbor, path)
                            path.pop()
                            visited.remove(neighbor)
                            if time.time() > search_deadline:
                                return

                dfs(start, path)
                
            if best_cycle:
                return best_cycle
            
            # Fallback: Find any cycle (e.g., triangle)
            for u in list(adj.keys()):
                if len(adj[u]) < 2:
                    continue
                for v in list(adj[u]):
                    if v <= u: continue
                    for w in list(adj[v]):
                        if w == u or w == v: continue
                        if (u, w) in remaining_edges or (w, u) in remaining_edges:
                            # Triangle u-v-w-u
                            cycle = [u, v, w]
                            return cycle
                            
            return None

        # Main loop: extract cycles
        while remaining_edges:
            if time.time() > time_limit:
                break
                
            cycle = find_best_cycle()
            
            if cycle:
                pieces.append(cycle)
                remove_cycle_edges(cycle)
            else:
                # No cycle found. 
                # If the remaining graph is a forest, we must take edges individually.
                for e in list(remaining_edges):
                    pieces.append(list(e))
                    remaining_edges.remove(e)
                break

        return pieces

    results = []
    for i, edges in enumerate(graphs):
        # Allocate time proportional to complexity
        time_limits = [30.0, 40.0, 60.0]
        
        current_limit = time_limits[i]
        # Ensure we don't exceed global deadline
        global_limit = deadline - time.time()
        if global_limit < current_limit:
            current_limit = max(1.0, global_limit)
            
        time_limit = time.time() + current_limit
        
        best_pieces = decompose_graph(edges, time_limit)
        results.append(best_pieces)
        
        # If time is critical, fallback to trivial for remaining
        if time.time() > deadline - 2.0:
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break
            
    return results

if __name__ == "__main__":
    res = construct()
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        edge_set = set(edges)
        
        used_edges = set()
        valid = True
        for piece in res[i]:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k+1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            else:
                valid = False
                break
                
        if valid and used_edges != edge_set:
            valid = False
            
        print(f"Graph {i}: Valid={valid}, Pieces={len(res[i])}")
