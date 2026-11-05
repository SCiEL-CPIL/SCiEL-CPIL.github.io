"""Path decomposition for Gallai's conjecture instances via greedy and iterative merging."""
import time
from collections import defaultdict
from random import Random

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _greedy_decompose(edge_set, adj, rng, deadline):
    """One greedy pass: build maximal simple paths, return list of paths."""
    remaining = set(edge_set)
    paths = []
    
    # Build remaining adjacency
    rem_adj = defaultdict(list)
    for u, v in remaining:
        rem_adj[u].append(v)
        rem_adj[v].append(u)
    
    def rem_degree(v):
        return len(rem_adj[v]) if v in rem_adj else 0
    
    while remaining and time.time() < deadline:
        # Choose start vertex: prefer odd-degree in remaining graph
        rem_odd = [u for u in rem_adj if rem_degree(u) % 2 == 1]
        if rem_odd:
            candidates = sorted(rem_odd, key=lambda u: -rem_degree(u))[:3]
            start = rng.choice(candidates)
        else:
            candidates = [u for u in rem_adj if rem_degree(u) > 0]
            if not candidates:
                break
            candidates = sorted(candidates, key=lambda u: -rem_degree(u))[:3]
            start = rng.choice(candidates)
        
        # Build a maximal simple path
        path = [start]
        in_path = {start}
        current = start
        
        while True:
            neighbors = [nb for nb in rem_adj[current] 
                         if (min(current, nb), max(current, nb)) in remaining]
            if not neighbors:
                break
            
            non_cycle = [nb for nb in neighbors if nb not in in_path]
            if non_cycle:
                non_cycle.sort(key=lambda v: -rem_degree(v))
                if rng.random() < 0.3 and len(non_cycle) > 1:
                    next_v = non_cycle[rng.randrange(1, min(3, len(non_cycle)))]
                else:
                    next_v = non_cycle[0]
            else:
                break
            
            key = (min(current, next_v), max(current, next_v))
            remaining.discard(key)
            path.append(next_v)
            in_path.add(next_v)
            
            if next_v in rem_adj[current]:
                rem_adj[current].remove(next_v)
            if current in rem_adj[next_v]:
                rem_adj[next_v].remove(current)
            if not rem_adj[current]:
                del rem_adj[current]
            if not rem_adj[next_v]:
                del rem_adj[next_v]
            
            current = next_v
        
        if len(path) > 1:
            paths.append(path)
    
    # Handle any leftover edges
    while remaining and time.time() < deadline:
        e = next(iter(remaining))
        u, v = e
        paths.append([u, v])
        remaining.discard(e)
        if v in rem_adj.get(u, []):
            rem_adj[u].remove(v)
        if u in rem_adj.get(v, []):
            rem_adj[v].remove(u)
        if u in rem_adj and not rem_adj[u]:
            del rem_adj[u]
        if v in rem_adj and not rem_adj[v]:
            del rem_adj[v]
    
    return paths

def _merge_paths(paths, adj, deadline):
    """
    Iteratively merge paths if they share an endpoint and the union is simple.
    Returns a new list of paths with reduced count.
    """
    if len(paths) <= 1:
        return paths
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_paths = []
        used = [False] * len(paths)
        
        for i in range(len(paths)):
            if used[i]:
                continue
            
            path_i = paths[i]
            start_i = path_i[0]
            end_i = path_i[-1]
            
            merged = False
            for j in range(len(paths)):
                if j == i or used[j]:
                    continue
                
                path_j = paths[j]
                start_j = path_j[0]
                end_j = path_j[-1]
                
                # Option 1: i ends at j's start
                if end_i == start_j:
                    candidate = path_i + path_j[1:]
                    if len(set(candidate)) == len(candidate):
                        new_paths.append(candidate)
                        used[i] = True
                        used[j] = True
                        merged = True
                        changed = True
                        break
                
                # Option 2: i ends at j's end
                if end_i == end_j:
                    candidate = path_i + list(reversed(path_j[1:]))
                    if len(set(candidate)) == len(candidate):
                        new_paths.append(candidate)
                        used[i] = True
                        used[j] = True
                        merged = True
                        changed = True
                        break
                
                # Option 3: i starts at j's start
                if start_i == start_j:
                    candidate = list(reversed(path_i[1:])) + path_j
                    if len(set(candidate)) == len(candidate):
                        new_paths.append(candidate)
                        used[i] = True
                        used[j] = True
                        merged = True
                        changed = True
                        break

                # Option 4: i starts at j's end
                if start_i == end_j:
                    candidate = path_j + path_i[1:]
                    if len(set(candidate)) == len(candidate):
                        new_paths.append(candidate)
                        used[i] = True
                        used[j] = True
                        merged = True
                        changed = True
                        break

            if not merged:
                new_paths.append(path_i)
                used[i] = True
        
        paths = new_paths
    
    return paths

def decompose_paths(edges, deadline):
    """Decompose edges into edge-disjoint simple paths, minimizing path count."""
    if not edges:
        return []
    
    edge_set = set()
    adj = defaultdict(list)
    for e in edges:
        u, v = e
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    
    # Compute odd vertices
    degree = {u: len(adj[u]) for u in adj}
    odd_vertices = [u for u in degree if degree[u] % 2 == 1]
    min_paths = max(1, len(odd_vertices) // 2)
    
    best_paths = None
    best_count = float('inf')
    
    # Multi-start greedy + merge
    max_restarts = 50
    for trial in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = Random(trial * 7919 + 13)
        raw_paths = _greedy_decompose(edge_set, adj, rng, deadline)
        
        # Merge
        merged_paths = _merge_paths(raw_paths, adj, deadline)
        
        count = len(merged_paths)
        if count < best_count:
            best_count = count
            best_paths = merged_paths
            
        if best_count == min_paths:
            break
    
    if best_paths is None:
        # Fallback: single-edge paths
        best_paths = [[u, v] for u, v in edge_set]
    
    return best_paths

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback: return single-edge paths for remaining graphs
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        paths = decompose_paths(edges, deadline)
        results.append(paths)
    
    return results
