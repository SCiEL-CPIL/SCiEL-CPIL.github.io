"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai style)."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import random

_DEADLINE = 130.0  # safety margin under 150s wall-clock kill

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
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

def _edges_from_cycle(cycle):
    """Return set of edges in a cycle (list of vertices, closing edge implied)."""
    edges = set()
    n = len(cycle)
    for i in range(n):
        u, v = cycle[i], cycle[(i + 1) % n]
        edges.add(tuple(sorted((u, v))))
    return edges

def _hierholzer_decompose(remaining_edges):
    """
    Decompose an Eulerian graph (all vertices have even degree) into edge-disjoint cycles.
    Uses Hierholzer's algorithm.
    Returns list of cycles (each cycle is a list of vertices).
    """
    if not remaining_edges:
        return []
    
    # Build adjacency with edge tracking
    adj = defaultdict(list)
    edge_set = set(remaining_edges)
    for u, v in edge_set:
        adj[u].append(v)
        adj[v].append(u)
    
    cycles = []
    remaining = set(edge_set)
    
    while remaining:
        # Find a vertex with remaining edges
        start = None
        for v in adj:
            if any(tuple(sorted((v, nb))) in remaining for nb in adj[v]):
                start = v
                break
        if start is None:
            break
        
        # Hierholzer's algorithm
        stack = [start]
        circuit = []
        
        while stack:
            v = stack[-1]
            # Find an unused edge from v
            next_v = None
            for nb in adj[v]:
                e = tuple(sorted((v, nb)))
                if e in remaining:
                    next_v = nb
                    break
            
            if next_v is not None:
                # Use this edge
                e = tuple(sorted((v, next_v)))
                remaining.discard(e)
                stack.append(next_v)
            else:
                # Backtrack
                circuit.append(stack.pop())
        
        circuit.reverse()
        
        # circuit is a closed walk; extract cycles
        # Actually, for an Eulerian graph, the circuit IS a cycle (or concatenation)
        # But Hierholzer gives us the full Eulerian circuit. We need to split into simple cycles.
        
        # The circuit visits vertices; since it's Eulerian, it's one big cycle
        # But it may revisit vertices. Let's extract simple cycles.
        # Actually, for a connected Eulerian graph, Hierholzer gives one Eulerian circuit.
        # We can decompose it into simple cycles.
        
        # Simpler approach: the circuit is a sequence where consecutive pairs are edges.
        # Since all degrees are even, the circuit is a single cycle (possibly with repeated vertices).
        # For our purposes, if the circuit has length >= 3, it's a valid "cycle" in the loose sense,
        # but the task requires distinct vertices. So we need simple cycles.
        
        # Let me re-approach: use a simpler cycle extraction for Eulerian graphs.
        pass
    
    # Alternative: iterative simple cycle extraction
    cycles = []
    remaining = set(remaining_edges)
    
    while remaining:
        # Find any cycle in remaining
        adj_rem = defaultdict(list)
        for u, v in remaining:
            adj_rem[u].append(v)
            adj_rem[v].append(u)
        
        # Find a cycle via DFS
        cycle = None
        for start in sorted(adj_rem.keys()):
            if not adj_rem[start]:
                continue
            visited = {start}
            path = [start]
            # DFS
            stack = [(start, iter(adj_rem[start]))]
            while stack:
                node, neighbors = stack[-1]
                advanced = False
                for nb in neighbors:
                    if nb == start and len(path) >= 3:
                        cycle = path[:]
                        break
                    elif nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        stack.append((nb, iter(adj_rem[nb])))
                        advanced = True
                        break
                if not advanced:
                    visited.discard(node)
                    path.pop()
                    stack.pop()
            if cycle:
                break
        
        if not cycle or len(cycle) < 3:
            break
        
        # Remove cycle edges
        cycle_edges = _edges_from_cycle(cycle)
        for e in cycle_edges:
            remaining.discard(e)
        cycles.append(cycle)
    
    return cycles

def _find_longest_cycle_greedy(adj, remaining_edges, max_depth=80, rng=None):
    """
    Try to find a long cycle using randomized DFS.
    Returns a cycle (list of vertices) or None.
    """
    if not remaining_edges:
        return None
    
    if rng is None:
        rng = random.Random()
    
    vertices = list(adj.keys())
    if not vertices:
        return None
    
    best_cycle = None
    best_len = 0
    
    # Try multiple starting points
    num_starts = min(len(vertices), 20)
    starts = rng.sample(vertices, num_starts) if len(vertices) > num_starts else vertices[:]
    
    for start in starts:
        if not adj[start]:
            continue
        
        # Randomize neighbor order
        adj_copy = {v: list(neighs) for v, neighs in adj.items() if v in adj}
        for v in adj_copy:
            rng.shuffle(adj_copy[v])
        
        # DFS to find a cycle back to start
        visited = {start}
        path = [start]
        found = False
        
        # Iterative DFS
        stack = [(start, iter(adj_copy[start]))]
        
        while stack:
            node, neighbors = stack[-1]
            advanced = False
            for nb in neighbors:
                e = tuple(sorted((node, nb)))
                if e not in remaining_edges:
                    continue
                if nb == start and len(path) >= 3:
                    cycle = path[:]
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                    found = True
                    break
                elif nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    stack.append((nb, iter(adj_copy[nb])))
                    advanced = True
                    break
            
            if not advanced:
                visited.discard(node)
                path.pop()
                stack.pop()
        
        if found and best_cycle and len(best_cycle) >= best_len:
            # Found a good cycle, stop early if it's long
            if best_len >= max_depth:
                break
    
    return best_cycle

def _decompose_one(edges, deadline):
    """Decompose a single graph into cycles and single edges, minimizing number of pieces."""
    if not edges:
        return []
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    remaining = set(edges_set)
    pieces = []
    rng = random.Random(42)
    
    # Phase 1: Repeatedly find cycles in the remaining graph
    # Prefer longer cycles to minimize piece count
    iteration = 0
    while remaining and time.time() < deadline:
        iteration += 1
        if iteration % 50 == 0 and time.time() > deadline:
            break
        
        # Build adjacency for remaining
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        # Try to find a long cycle
        cycle = _find_longest_cycle_greedy(adj, remaining, max_depth=50, rng=rng)
        
        if cycle is None or len(cycle) < 3:
            break
        
        # Verify all edges in cycle are in remaining
        cycle_edges = _edges_from_cycle(cycle)
        valid = all(e in remaining for e in cycle_edges)
        
        if not valid:
            # Fallback: try a simple 3-cycle
            cycle = None
            for u in sorted(adj.keys()):
                for v in sorted(adj[u]):
                    if v <= u:
                        continue
                    for w in sorted(adj.get(v, set())):
                        if w <= u:
                            continue
                        if w in adj.get(u, set()):
                            cycle = [u, v, w]
                            break
                    if cycle:
                        break
                if cycle:
                    break
            if cycle is None:
                break
            
            cycle_edges = _edges_from_cycle(cycle)
            if not all(e in remaining for e in cycle_edges):
                break
        
        # Remove cycle edges
        for e in cycle_edges:
            remaining.discard(e)
        pieces.append(cycle)
    
    # Phase 2: Remaining edges as single-edge pieces
    for u, v in sorted(remaining):
        pieces.append([u, v])
    
    # Phase 3: Local search - try to merge two pieces into a larger cycle
    # This is complex; skip for now due to time constraints
    
    return pieces

def _local_search_improve(pieces, edges_set, deadline, max_rounds=10):
    """
    Try to improve the decomposition by merging pieces.
    Two pieces can be merged if they share a vertex and the combined structure forms a cycle.
    For simplicity, try to find a cycle that uses edges from two or more existing pieces.
    """
    if not pieces or time.time() > deadline:
        return pieces
    
    # Build a map from edge to piece index
    edge_to_piece = {}
    for i, piece in enumerate(pieces):
        if len(piece) == 2:
            e = tuple(sorted(piece))
            edge_to_piece[e] = i
        else:
            for j in range(len(piece)):
                u, v = piece[j], piece[(j + 1) % len(piece)]
                e = tuple(sorted((u, v)))
                edge_to_piece[e] = i
    
    # Try to find a new cycle that replaces multiple existing pieces
    # This is a complex search; do a limited version
    remaining_edges = set(edges_set)
    for e in edge_to_piece:
        remaining_edges.discard(e)
    # Actually, all edges are covered. Let's try a different approach.
    
    # For now, return pieces unchanged
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose_one(edges, _DEADLINE)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
