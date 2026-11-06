import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
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
        
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set(edges)
        pieces = []
        
        def get_cycle_from_path(path):
            # path is a list of vertices [v0, v1, ..., vk]
            # Check if edge (vk, v0) exists
            if len(path) < 3:
                return None
            u, v = path[0], path[-1]
            if (u, v) in remaining_edges or (v, u) in remaining_edges:
                # Verify all edges in path exist
                for i in range(len(path) - 1):
                    a, b = path[i], path[i+1]
                    if (a, b) not in remaining_edges and (b, a) not in remaining_edges:
                        return None
                return path
            return None

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
            
            # Update adjacency
            for u in cycle:
                adj[u] = set()
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)

        def find_longest_cycle_greedy():
            if not remaining_edges:
                return None
            
            # Try to find a cycle by greedy path extension
            # Start from a vertex with high degree
            best_cycle = None
            best_len = 0
            
            # Limit searches to avoid timeout
            num_starts = min(50, len(remaining_edges))
            start_vertices = list(adj.keys())
            random.shuffle(start_vertices)
            
            for start in start_vertices[:num_starts]:
                if len(adj[start]) < 2:
                    continue
                
                # Try multiple paths from start
                for _ in range(10):
                    path = [start]
                    visited = {start}
                    current = start
                    
                    # Greedy extension
                    while True:
                        neighbors = list(adj[current])
                        # Prefer unvisited
                        unvisited = [n for n in neighbors if n not in visited]
                        
                        if not unvisited:
                            break
                            
                        # Heuristic: pick neighbor that leads to most options?
                        # Or just random
                        next_v = random.choice(unvisited)
                        path.append(next_v)
                        visited.add(next_v)
                        current = next_v
                        
                        # Check if we can close cycle
                        if len(path) >= 3 and (path[0], current) in remaining_edges:
                            cycle = get_cycle_from_path(path)
                            if cycle:
                                if len(cycle) > best_len:
                                    best_len = len(cycle)
                                    best_cycle = cycle
                            break
                        
                        # Check if we hit a dead end (no unvisited neighbors)
                        if not adj[current]:
                            break
                            
            return best_cycle

        def find_cycles_in_component(vertices):
            # This is a placeholder for a more robust search if greedy fails
            pass

        # Main loop: extract cycles
        while remaining_edges:
            if time.time() > time_limit:
                break
                
            cycle = find_longest_cycle_greedy()
            
            if cycle:
                pieces.append(cycle)
                remove_cycle_edges(cycle)
            else:
                # No cycle found, pick an edge
                # Try to find a path of length 2? 
                # For now, just take one edge
                edge = next(iter(remaining_edges))
                pieces.append(list(edge))
                remaining_edges.remove(edge)
                u, v = edge
                adj[u].discard(v)
                adj[v].discard(u)
                if not adj[u]:
                    del adj[u]
                if not adj[v]:
                    del adj[v]
                    
        return pieces

    results = []
    for i, edges in enumerate(graphs):
        # Allocate time proportional to complexity, but ensure each gets some
        # Simple heuristic: equal time split, with minimum 1s per graph
        time_per_graph = (deadline - time.time()) / (len(graphs) - i)
        # Add a small buffer
        time_limit = time.time() + max(1.0, time_per_graph * 0.9)
        
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
    # Quick sanity check
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        # Re-generate to check
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
                # Check distinct vertices
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                # Check edges
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
