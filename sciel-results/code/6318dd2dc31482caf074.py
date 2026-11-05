"""Path decomposition for Gallai's conjecture instances via multi-start greedy with best-of selection."""
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
    """One greedy pass: build maximal simple paths, return (paths, remaining_edges)."""
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
            # Pick randomly among top candidates for diversity
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
            # Get neighbors with unused edges
            neighbors = [nb for nb in rem_adj[current] 
                         if (min(current, nb), max(current, nb)) in remaining]
            if not neighbors:
                break
            
            # Prefer neighbors that are NOT in path (to avoid cycles)
            # Among those, prefer higher degree; if all are in path, pick one
            non_cycle = [nb for nb in neighbors if nb not in in_path]
            if non_cycle:
                non_cycle.sort(key=lambda v: -rem_degree(v))
                # Add some randomness
                if rng.random() < 0.3 and len(non_cycle) > 1:
                    next_v = non_cycle[rng.randrange(1, min(3, len(non_cycle)))]
                else:
                    next_v = non_cycle[0]
            else:
                # All neighbors are in path; pick one to allow cycle (but path must be simple)
                # Actually we can't revisit, so we're stuck
                break
            
            key = (min(current, next_v), max(current, next_v))
            remaining.discard(key)
            path.append(next_v)
            in_path.add(next_v)
            
            # Update rem_adj
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
        elif len(path) == 1:
            # Isolated vertex with no remaining edges? Shouldn't happen if we started from rem_adj
            pass
    
    # Handle any leftover edges (should be rare)
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

def decompose_paths(edges, deadline):
    """Decompose edges into edge-disjoint simple paths, minimizing path count via multi-start."""
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
    
    # Multi-start: try several random seeds, keep best
    best_paths = None
    best_count = float('inf')
    
    # Determine number of restarts based on time budget
    # Each greedy pass is O(E * V) worst case, but fast in practice
    # For graphs of size 20-30, each pass is < 1ms
    # Allow up to ~50 restarts per graph
    max_restarts = 30
    
    for trial in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = Random(trial * 7919 + 13)
        paths = _greedy_decompose(edge_set, adj, rng, deadline)
        
        count = len(paths)
        if count < best_count:
            best_count = count
            best_paths = paths
        
        # Early stop if we hit the theoretical minimum
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
