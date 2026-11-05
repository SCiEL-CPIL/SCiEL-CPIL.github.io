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

def _find_cycles_greedy(adj, n, rng, max_cycles=200, deadline=None):
    """Find many candidate cycles using randomized greedy walks."""
    cycles = []
    seen = set()
    
    for _ in range(max_cycles):
        if deadline and time.time() > deadline:
            break
        
        # Pick a random start vertex with remaining edges
        candidates = [v for v in range(n) if adj[v]]
        if not candidates:
            break
        
        start = rng.choice(candidates)
        path = [start]
        visited = {start}
        current = start
        
        for _ in range(n * 2):  # max path length
            neighbors = list(adj[current])
            unvisited = [v for v in neighbors if v not in visited]
            
            if unvisited:
                # Prefer unvisited neighbors
                current = rng.choice(unvisited)
                path.append(current)
                visited.add(current)
            else:
                # Try to close the cycle
                if len(path) >= 3 and start in adj[current]:
                    cycle_key = tuple(sorted(path))
                    if cycle_key not in seen:
                        seen.add(cycle_key)
                        cycles.append(path[:])
                    break
                else:
                    break
    
    return cycles

def _set_cover_decompose(edges, n, rng, deadline):
    """Use greedy set cover to select minimum cycles covering all edges."""
    # Build edge set
    edge_set = set(tuple(sorted(e)) for e in edges)
    total_edges = len(edge_set)
    
    if total_edges == 0:
        return []
    
    # Build adjacency for cycle finding
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find many candidate cycles
    candidate_cycles = _find_cycles_greedy(adj, n, rng, max_cycles=300, deadline=deadline)
    
    # Also try to find longer cycles with different strategies
    for attempt in range(20):
        if deadline and time.time() > deadline:
            break
        # Try to find a long cycle starting from a high-degree vertex
        degs = [(len(adj[v]), v) for v in range(n) if adj[v]]
        if not degs:
            break
        degs.sort(reverse=True)
        
        for _, start in degs[:5]:
            if deadline and time.time() > deadline:
                break
            
            # Randomized DFS to find a long cycle
            best_path = None
            for _ in range(5):
                path = [start]
                visited = {start}
                current = start
                
                while True:
                    neighbors = list(adj[current])
                    unvisited = [v for v in neighbors if v not in visited]
                    
                    if unvisited:
                        # Bias toward high-degree unvisited neighbors
                        weighted = [(len(adj[v]), v) for v in unvisited]
                        weighted.sort(reverse=True)
                        # Take from top 3 with some randomness
                        top_k = min(3, len(weighted))
                        current = weighted[rng.randint(0, top_k - 1)][1]
                        path.append(current)
                        visited.add(current)
                    else:
                        if len(path) >= 3 and start in adj[current]:
                            if best_path is None or len(path) > len(best_path):
                                best_path = path[:]
                        break
                
                if best_path and len(best_path) >= 5:
                    break
            
            if best_path:
                cycle_key = tuple(sorted(best_path))
                if cycle_key not in {tuple(sorted(c)) for c in candidate_cycles}:
                    candidate_cycles.append(best_path)
    
    # Greedy set cover: repeatedly pick the cycle that covers the most uncovered edges
    covered_edges = set()
    selected_cycles = []
    remaining = set(edge_set)
    
    while remaining:
        if deadline and time.time() > deadline:
            break
        
        best_cycle = None
        best_count = 0
        
        for cycle in candidate_cycles:
            if not cycle:
                continue
            m = len(cycle)
            cycle_edges = set()
            for i in range(m):
                a = cycle[i]
                b = cycle[(i + 1) % m]
                e = (min(a, b), max(a, b))
                cycle_edges.add(e)
            
            new_covered = len(cycle_edges & remaining)
            if new_covered > best_count:
                best_count = new_covered
                best_cycle = cycle
        
        if best_cycle and best_count > 0:
            m = len(best_cycle)
            for i in range(m):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % m]
                e = (min(a, b), max(a, b))
                if e in remaining:
                    remaining.discard(e)
            selected_cycles.append(best_cycle)
        else:
            # No cycle covers any remaining edge; take single edges
            for e in list(remaining):
                selected_cycles.append(list(e))
            remaining.clear()
    
    return selected_cycles

def _decompose_once(edges, n, rng, deadline):
    """Single decomposition attempt using set cover."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    remaining = len(edges)
    
    # First, try to find a good set of cycles
    candidate_cycles = _find_cycles_greedy(adj, n, rng, max_cycles=150, deadline=deadline)
    
    # Greedy cycle extraction
    covered_edges = set()
    for cycle in sorted(candidate_cycles, key=len, reverse=True):
        if deadline and time.time() > deadline:
            break
        
        m = len(cycle)
        if m < 3:
            continue
        
        # Check if all edges in cycle are still available
        cycle_valid = True
        cycle_edges = []
        for i in range(m):
            a = cycle[i]
            b = cycle[(i + 1) % m]
            e = (min(a, b), max(a, b))
            if e in covered_edges:
                cycle_valid = False
                break
            cycle_edges.append(e)
        
        if cycle_valid and len(cycle_edges) > 0:
            for e in cycle_edges:
                covered_edges.add(e)
            pieces.append(cycle)
            remaining -= len(cycle_edges)
    
    # Add remaining edges as single-edge pieces
    all_edges = set(tuple(sorted(e)) for e in edges)
    for e in all_edges:
        if e not in covered_edges:
            pieces.append(list(e))
    
    return pieces

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    
    num_attempts = 30
    base_seed = 42
    
    for i in range(num_attempts):
        if time.time() > deadline - 0.5:
            break
        
        rng = random.Random(base_seed + i)
        
        # Alternate between set cover and greedy approaches
        if i % 2 == 0:
            pieces = _set_cover_decompose(edges, n, rng, deadline)
        else:
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
