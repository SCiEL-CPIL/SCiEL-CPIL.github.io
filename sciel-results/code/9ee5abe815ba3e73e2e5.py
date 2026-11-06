"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_cycle_heuristic(adj, seed, deadline):
    """
    Find a long cycle using a randomized greedy DFS approach.
    Combines degree-based ordering with randomization for diversity.
    """
    if not adj:
        return []
    
    vertices = [v for v in adj if adj[v]]
    if not vertices:
        return []
    
    rng = random.Random(seed)
    best_cycle = []
    
    # Try multiple starting strategies: high-degree vertices and random
    deg_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
    starts = list(set(deg_sorted[:8] + vertices[:8]))
    rng.shuffle(starts)
    
    for start in starts[:15]:
        if time.time() > deadline:
            break
        if not adj[start]:
            continue
        
        visited = {start}
        path = [start]
        
        # Greedy DFS: at each step, prefer unvisited neighbors with higher degree
        # to keep the path extendable and increase chances of closing a long cycle
        current = start
        while True:
            if time.time() > deadline:
                return best_cycle
            
            # Check if we can close the cycle
            if start in adj[current] and len(path) >= 3:
                if len(path) > len(best_cycle):
                    best_cycle = list(path)
            
            # Get unvisited neighbors, sorted by degree (descending)
            unvisited = [v for v in adj[current] if v not in visited]
            if not unvisited:
                break
            
            # Sort by degree, add some randomness among top candidates
            unvisited.sort(key=lambda v: -len(adj[v]))
            top_k = min(4, len(unvisited))
            # Randomize among top candidates for diversity
            chosen = rng.choice(unvisited[:top_k])
            
            visited.add(chosen)
            path.append(chosen)
            current = chosen
        
        # After path ends, check for any back-edge to form a cycle
        # Look for edges from path[i] to path[j] where i < j and j-i >= 2
        for i in range(len(path) - 1):
            if time.time() > deadline:
                return best_cycle
            for j in range(len(path) - 1, i + 1, -1):
                if path[i] in adj[path[j]] and j - i >= 2:
                    cycle = path[i:j + 1]
                    if len(cycle) > len(best_cycle):
                        best_cycle = list(cycle)
                    break  # Only take the longest back-edge for this i
    
    return best_cycle


def _decompose(edges, n, seed, time_limit=45.0):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []
    
    best_pieces = None
    best_count = len(edges)
    start_time = time.time()
    deadline = start_time + time_limit
    
    def greedy_decompose(remaining_edges, n, seed_offset):
        remaining = set(remaining_edges)
        pieces = []
        adj = _build_adj(remaining)
        
        cycle_counter = 0
        while remaining:
            if time.time() > deadline:
                # Return partial result; caller will handle
                break
            
            # Find a long cycle
            cycle = _find_cycle_heuristic(adj, seed + seed_offset * 1000 + cycle_counter, deadline)
            if cycle:
                # Extract cycle edges
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    cycle_edges.add(e)
                
                # Remove from remaining and adj
                for e in cycle_edges:
                    if e in remaining:
                        remaining.discard(e)
                        u, v = e
                        if v in adj[u]:
                            adj[u].discard(v)
                        if u in adj[v]:
                            adj[v].discard(u)
                
                pieces.append(cycle)
                cycle_counter += 1
            else:
                # No cycle found; remove a single edge
                # Prefer edges where both endpoints have low degree
                # to preserve cycle potential in denser parts
                best_edge = None
                best_priority = float('inf')
                for u, v in remaining:
                    deg_u = len(adj[u]) if u in adj else 0
                    deg_v = len(adj[v]) if v in adj else 0
                    priority = deg_u + deg_v
                    if priority < best_priority:
                        best_priority = priority
                        best_edge = (u, v)
                
                if best_edge is None:
                    best_edge = next(iter(remaining))
                
                u, v = best_edge
                remaining.discard(best_edge)
                if v in adj[u]:
                    adj[u].discard(v)
                if u in adj[v]:
                    adj[v].discard(u)
                pieces.append([u, v])
        
        return pieces
    
    # Multi-start search with diverse random seeds
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        pieces = greedy_decompose(edges, n, restart)
        if not pieces:
            continue
            
        # Verify all edges are covered
        covered = set()
        valid = True
        for piece in pieces:
            if len(piece) == 2:
                u, v = piece
                covered.add((min(u, v), max(u, v)))
            elif len(piece) >= 3:
                for i in range(len(piece)):
                    u = piece[i]
                    v = piece[(i + 1) % len(piece)]
                    covered.add((min(u, v), max(u, v)))
        
        if covered != set(edges):
            valid = False
        
        if valid:
            current_count = len(pieces)
            if current_count < best_count:
                best_count = current_count
                best_pieces = pieces
    
    if best_pieces is None:
        best_pieces = [[u, v] for u, v in edges]
    
    return best_pieces


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, seed, time_limit=45.0)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
