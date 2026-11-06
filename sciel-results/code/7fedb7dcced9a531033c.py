import time
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        """
        Decompose graph into edge-disjoint simple paths covering all edges.
        Strategy: Greedy extension from endpoints, tracking used edges.
        """
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Start with each edge as a path
        paths = [[u, v] for u, v in edges]
        used_edges = set(edge_set)
        
        changed = True
        max_iterations = 500
        iteration = 0
        while changed and iteration < max_iterations:
            if time.time() > deadline:
                break
            changed = False
            iteration += 1
            
            # Try to extend each path at its endpoints using unused edges
            for i in range(len(paths)):
                if time.time() > deadline:
                    break
                path = paths[i]
                if len(path) < 2:
                    continue
                
                # Try extending at the start
                start_v = path[0]
                next_v = path[1]
                for nb in sorted(adj[start_v]):
                    if nb == next_v:
                        continue
                    e = tuple(sorted((start_v, nb)))
                    if e in edge_set and e not in used_edges and nb not in path:
                        paths[i] = [nb] + path
                        used_edges.add(e)
                        changed = True
                        break
                
                # Try extending at the end
                end_v = path[-1]
                prev_v = path[-2]
                for nb in sorted(adj[end_v]):
                    if nb == prev_v:
                        continue
                    e = tuple(sorted((end_v, nb)))
                    if e in edge_set and e not in used_edges and nb not in path:
                        paths[i] = path + [nb]
                        used_edges.add(e)
                        changed = True
                        break
            
            # Try to merge paths that share an endpoint
            merged = True
            while merged:
                if time.time() > deadline:
                    break
                merged = False
                for i in range(len(paths)):
                    if time.time() > deadline:
                        break
                    for j in range(len(paths)):
                        if i == j:
                            continue
                        end_i = paths[i][-1]
                        start_j = paths[j][0]
                        if end_i == start_j:
                            merged_path = paths[i] + paths[j][1:]
                            if len(merged_path) == len(set(merged_path)):
                                paths[i] = merged_path
                                paths.pop(j)
                                merged = True
                                break
                        end_j = paths[j][-1]
                        start_i = paths[i][0]
                        if end_j == start_i:
                            merged_path = paths[j] + paths[i][1:]
                            if len(merged_path) == len(set(merged_path)):
                                paths[i] = merged_path
                                paths.pop(j)
                                merged = True
                                break
                    if merged:
                        break
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
    
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results
