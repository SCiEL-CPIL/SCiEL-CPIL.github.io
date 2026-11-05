import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_long_cycle(adj, n, max_attempts=20, deadline=None):
    """
    Find a long simple cycle in the graph represented by adj.
    Uses degree-based heuristics to prefer extending through high-degree vertices.
    Returns the cycle as a list of vertices or None.
    """
    best_cycle = None
    best_len = 0
    
    # Try starting from each vertex, prioritizing high-degree vertices
    candidates = [v for v in range(n) if adj[v]]
    if not candidates:
        return None
    
    # Sort by degree descending to find longer cycles earlier
    candidates.sort(key=lambda v: len(adj[v]), reverse=True)
    
    for start in candidates:
        if deadline is not None and time.time() > deadline:
            break
            
        for _ in range(max_attempts):
            if deadline is not None and time.time() > deadline:
                break
                
            path = [start]
            visited = {start}
            current = start
            
            while True:
                neighbors = list(adj[current])
                unvisited_neighbors = [v for v in neighbors if v not in visited]
                
                can_close = start in neighbors and len(path) >= 3
                
                if not unvisited_neighbors:
                    if can_close:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    break
                else:
                    # Heuristic: prefer neighbors with higher degree (more options)
                    # to extend the path further
                    scored = [(len(adj[v]), v) for v in unvisited_neighbors]
                    scored.sort(reverse=True)
                    
                    # With some probability, pick randomly to avoid getting stuck
                    if random.random() < 0.3:
                        next_node = random.choice(unvisited_neighbors)
                    else:
                        # Pick from top 3 highest-degree neighbors
                        top_k = min(3, len(scored))
                        next_node = random.choice([scored[i][1] for i in range(top_k)])
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node

    return best_cycle

def _decompose_greedy(edges, n, deadline, rng):
    """Greedy decomposition using improved cycle finding."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining = len(edges)
    
    while remaining > 0:
        if time.time() > deadline:
            break
            
        # Find a long cycle
        cyc = _find_long_cycle(adj, n, max_attempts=10, deadline=deadline)
        
        if cyc:
            # Remove cycle edges
            m = len(cyc)
            for i in range(m):
                a = cyc[i]
                b = cyc[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(cyc)
        else:
            # No cycle found, remove an edge
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

def _local_search_polish(pieces, edges, n, deadline):
    """
    Try to merge single-edge pieces into cycles by finding alternative decompositions.
    This is a local search that tries to replace some single edges with cycles.
    """
    if time.time() > deadline:
        return pieces
    
    # Count single edges
    single_edges = [p for p in pieces if len(p) == 2]
    if len(single_edges) < 2:
        return pieces
    
    # Try to find cycles that use some of these single edges
    # Build adjacency from single edges
    single_adj = defaultdict(set)
    for e in single_edges:
        u, v = e[0], e[1]
        single_adj[u].add(v)
        single_adj[v].add(u)
    
    # Find cycles in the subgraph formed by single edges
    improved_pieces = [p for p in pieces if len(p) != 2]
    remaining_singles = single_edges[:]
    
    # Try to find cycles among the single edges
    used_edges = set()
    for _ in range(50):  # Limited attempts
        if time.time() > deadline:
            break
        
        # Find a cycle in the single-edge subgraph
        cycle = _find_cycle_in_subgraph(single_adj, n, deadline)
        if cycle and len(cycle) >= 3:
            # Remove these edges from remaining_singles
            cycle_edges = set()
            for i in range(len(cycle)):
                a = cycle[i]
                b = cycle[(i + 1) % len(cycle)]
                cycle_edges.add(tuple(sorted((a, b))))
            
            # Check if all these edges are in remaining_singles
            valid = True
            for ce in cycle_edges:
                if ce not in [(tuple(sorted(e))) for e in remaining_singles]:
                    valid = False
                    break
            
            if valid:
                # Remove these edges from remaining_singles
                new_remaining = []
                for e in remaining_singles:
                    if tuple(sorted(e)) not in cycle_edges:
                        new_remaining.append(e)
                remaining_singles = new_remaining
                improved_pieces.append(cycle)
    
    improved_pieces.extend(remaining_singles)
    return improved_pieces

def _find_cycle_in_subgraph(adj, n, deadline):
    """Find a cycle in a subgraph."""
    for start in range(n):
        if not adj[start]:
            continue
        
        for _ in range(10):
            if time.time() > deadline:
                break
                
            path = [start]
            visited = {start}
            current = start
            
            while True:
                neighbors = list(adj[current])
                unvisited = [v for v in neighbors if v not in visited]
                can_close = start in neighbors and len(path) >= 3
                
                if not unvisited:
                    if can_close:
                        return path
                    break
                else:
                    next_node = random.choice(unvisited)
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
    
    return None

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    
    num_attempts = 30
    base_seed = 42
    
    for i in range(num_attempts):
        if time.time() > deadline - 2.0:
            break
            
        rng = random.Random(base_seed + i)
        pieces = _decompose_greedy(edges, n, deadline, rng)
        cnt = len(pieces)
        
        # Apply local search polish
        polished = _local_search_polish(pieces, edges, n, deadline)
        polished_count = len(polished)
        
        if polished_count < best_count:
            best_count = polished_count
            best = polished
        elif cnt < best_count:
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
        
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0
            
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
            
    return results
