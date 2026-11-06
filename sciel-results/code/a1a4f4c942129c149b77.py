"""Path decompositions of three fixed connected graphs (Gallai's conjecture)."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def greedy_path_decomposition(edges, max_iter=200):
    """Find edge-disjoint paths covering all edges using greedy + local search."""
    adj = defaultdict(list)
    for e in edges:
        adj[e[0]].append(e[1])
        adj[e[1]].append(e[0])
    
    used = set()
    paths = []
    
    # Build list of edges
    edge_list = list(edges)
    
    for _ in range(max_iter):
        # Greedy: pick an unused edge and try to extend
        found = False
        for i, e in enumerate(edge_list):
            if e in used:
                continue
            # Start a new path with this edge
            path = [e[0], e[1]]
            used.add(e)
            found = True
            
            # Try to extend from both ends
            extended = True
            while extended:
                extended = False
                # Extend from the end (last vertex)
                last_v = path[-1]
                for nb in adj[last_v]:
                    for e2 in edges:
                        if e2 in used:
                            continue
                        if (e2[0] == last_v and e2[1] == nb) or (e2[1] == last_v and e2[0] == nb):
                            if nb not in path:  # simplicity check
                                used.add(e2)
                                path.append(nb)
                                extended = True
                                break
                    if extended:
                        break
                if extended:
                    continue
                
                # Extend from the beginning
                first_v = path[0]
                for nb in adj[first_v]:
                    for e2 in edges:
                        if e2 in used:
                            continue
                        if (e2[0] == first_v and e2[1] == nb) or (e2[1] == first_v and e2[0] == nb):
                            if nb not in path:  # simplicity check
                                used.add(e2)
                                path.insert(0, nb)
                                extended = True
                                break
                    if extended:
                        break
            
            paths.append(path)
        
        if not found:
            break
    
    # If we didn't cover all edges, add remaining edges as single-edge paths
    for e in edges:
        if e not in used:
            paths.append([e[0], e[1]])
            used.add(e)
    
    return paths

def matching_based_decomposition(edges, deadline):
    """Use vertex-disjoint matching to minimize path count."""
    adj = defaultdict(list)
    for e in edges:
        adj[e[0]].append(e[1])
        adj[e[1]].append(e[0])
    
    used_edges = set()
    paths = []
    
    # Greedy pairing: at each vertex, try to pair edges to form longer paths
    # Use a simple heuristic: repeatedly pick a path and extend it
    
    remaining = set(edges)
    
    while remaining and time.time() < deadline:
        # Pick an edge from remaining
        start_edge = next(iter(remaining))
        remaining.discard(start_edge)
        
        path = [start_edge[0], start_edge[1]]
        
        # Extend greedily
        extended = True
        while extended and time.time() < deadline:
            extended = False
            # Try to extend from both ends
            for end_idx in [0, -1]:
                if end_idx == 0:
                    curr_v = path[0]
                    insert_at = 0
                else:
                    curr_v = path[-1]
                    insert_at = None
                
                for nb in adj[curr_v]:
                    found = False
                    for e in list(remaining):
                        if (e[0] == curr_v and e[1] == nb) or (e[1] == curr_v and e[0] == nb):
                            if nb not in path:
                                remaining.discard(e)
                                if insert_at == 0:
                                    path.insert(0, nb)
                                else:
                                    path.append(nb)
                                extended = True
                                found = True
                                break
                    if found:
                        break
            # Check time
            if time.time() >= deadline:
                break
        
        paths.append(path)
    
    # Add any remaining edges as single-edge paths
    for e in remaining:
        paths.append([e[0], e[1]])
    
    return paths

def improve_with_local_search(paths, edges, adj, deadline):
    """Try to merge paths by finding overlapping vertices."""
    # This is a placeholder for potential improvement
    return paths

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Keep 30s safety margin
    
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: each edge as its own path
            edges = instance(*spec)
            results.append([[e[0], e[1]] for e in edges])
            continue
        
        edges = instance(*spec)
        adj = defaultdict(list)
        for e in edges:
            adj[e[0]].append(e[1])
            adj[e[1]].append(e[0])
        
        # Try greedy decomposition
        paths = matching_based_decomposition(edges, deadline)
        
        # Validate: all edges covered, paths are simple, use graph edges
        used_edges = set()
        valid = True
        for path in paths:
            if len(path) < 2:
                valid = False
                break
            # Check simplicity
            if len(set(path)) != len(path):
                valid = False
                break
            # Check edges exist
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in edges:
                    valid = False
                    break
                if e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            if not valid:
                break
        
        if not valid or used_edges != set(edges):
            # Fallback to per-edge paths
            paths = [[e[0], e[1]] for e in edges]
        
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
