"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
from collections import deque
from scipy.optimize import linear_sum_assignment

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_2factor_cycles(adj, n):
    """Find vertex-disjoint cycles using maximum bipartite matching (2-matching).
    
    Each edge (u,v) in the original graph creates a node in a bipartite graph.
    Left side: edge nodes. Right side: vertex nodes (each vertex appears twice: once as 'in', once as 'out').
    An edge node (u,v) connects to u_out and v_in.
    A maximum matching gives a 2-factor (set of vertex-disjoint cycles).
    """
    # Build edge list
    edge_list = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_list.append((u, v))
    
    if not edge_list:
        return []
    
    m = len(edge_list)
    
    # Build bipartite graph: left = edge indices, right = vertex occurrences
    # For each edge (u,v), it can use u as "out" and v as "in", OR v as "out" and u as "in"
    # We model this as: left node i (edge i) connects to right nodes representing (vertex, role)
    # Right nodes: 0..2n-1 where vertex v has nodes 2v (in) and 2v+1 (out)
    
    # Cost matrix for Hungarian: minimize cost = 0 for valid assignments, large for invalid
    INF_COST = 10**9
    cost = [[INF_COST] * (2 * n) for _ in range(m)]
    
    for i, (u, v) in enumerate(edge_list):
        # Edge (u,v): u can be "out" (connect to 2u+1), v can be "in" (connect to 2v)
        # OR: v can be "out" (connect to 2v+1), u can be "in" (connect to 2u)
        cost[i][2 * u + 1] = 0  # u is out
        cost[i][2 * v] = 0      # v is in
        cost[i][2 * v + 1] = 0  # v is out
        cost[i][2 * u] = 0      # u is in
    
    # Find minimum cost assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Extract matched edges and reconstruct cycles
    # For each matched edge i assigned to column j:
    #   if j = 2v+1: vertex v is "out" for this edge
    #   if j = 2v: vertex v is "in" for this edge
    # Each vertex should have exactly one "in" and one "out" if fully matched
    
    matched_edges = {}  # edge_index -> (u, v, u_role, v_role)
    for i, j in zip(row_ind, col_ind):
        if cost[i][j] == 0:
            u, v = edge_list[i]
            if j == 2 * u + 1:
                # u is out
                matched_edges[i] = (u, v, 'out', 'in')
            elif j == 2 * v:
                # v is in
                matched_edges[i] = (u, v, 'out', 'in')
            elif j == 2 * v + 1:
                # v is out
                matched_edges[i] = (u, v, 'in', 'out')
            elif j == 2 * u:
                # u is in
                matched_edges[i] = (u, v, 'in', 'out')
    
    # Reconstruct cycles from the 2-factor
    # Build directed graph: for each matched edge, determine direction
    # If u_role = 'out', then u -> v (if v_role = 'in')
    # If v_role = 'out', then v -> u (if u_role = 'in')
    
    # Build adjacency: vertex -> next vertex
    next_vertex = {}
    prev_vertex = {}
    
    for i, (u, v, u_role, v_role) in matched_edges.items():
        if u_role == 'out' and v_role == 'in':
            next_vertex[u] = v
            prev_vertex[v] = u
        elif v_role == 'out' and u_role == 'in':
            next_vertex[v] = u
            prev_vertex[u] = v
    
    # Find all cycles
    visited = set()
    cycles = []
    
    for start in next_vertex:
        if start in visited:
            continue
        # Follow the cycle
        cycle = [start]
        visited.add(start)
        current = start
        while True:
            current = next_vertex.get(current)
            if current is None or current == start:
                break
            if current in visited:
                break
            cycle.append(current)
            visited.add(current)
        
        if len(cycle) >= 3 and next_vertex.get(cycle[-1]) == cycle[0]:
            cycles.append(cycle)
    
    return cycles

def _find_cycle_greedy(adj, n, start, rng):
    """Find a cycle starting from start using randomized DFS."""
    visited = {start}
    path = [start]
    stack = [(start, list(adj[start]))]
    found = False
    
    while stack and not found:
        node, neighbors = stack[-1]
        rng.shuffle(neighbors)
        advanced = False
        for nxt in neighbors:
            if nxt == start and len(path) >= 3:
                found = True
                break
            if nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                stack.append((nxt, list(adj[nxt])))
                advanced = True
                break
        if not advanced and not found:
            stack.pop()
            path.pop()
    
    if found:
        return list(path)
    return None

def _decompose(edges, n, rng=None):
    """Decompose edges into cycles and single edges using 2-matching and greedy cycle packing."""
    if rng is None:
        rng = Random(42)
    
    pieces = []
    remaining = set(tuple(sorted(e)) for e in edges)
    
    # Build initial adjacency
    adj = {i: set() for i in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    # Phase 1: Use 2-matching to find vertex-disjoint cycles
    cycles_found = _find_2factor_cycles(adj, n)
    
    # Verify and extract valid cycles
    for cycle in cycles_found:
        valid = True
        cycle_edges = []
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge = tuple(sorted((u, v)))
            if edge not in remaining:
                valid = False
                break
            cycle_edges.append(edge)
        
        if valid and len(cycle) >= 3:
            for edge in cycle_edges:
                remaining.discard(edge)
                u, v = edge
                adj[u].discard(v)
                adj[v].discard(u)
            pieces.append(cycle[:])
    
    # Phase 2: Find more cycles greedily in the remaining graph
    max_iterations = 200
    iters = 0
    while remaining and iters < max_iterations:
        # Find the best cycle among several random attempts
        best_cycle = None
        best_score = 0
        
        num_attempts = min(30, max(5, len(remaining) // 10))
        
        for attempt in range(num_attempts):
            candidates = [v for v in range(n) if len(adj[v]) >= 2]
            if not candidates:
                break
            
            weights = [len(adj[v]) for v in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            start = candidates[0]
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    start = candidates[i]
                    break
            
            cycle = _find_cycle_greedy(adj, n, start, rng)
            if cycle and len(cycle) >= 3:
                score = len(cycle)
                if score > best_score:
                    best_score = score
                    best_cycle = list(cycle)
        
        if best_cycle is None or len(best_cycle) < 3:
            break
        
        # Verify and extract cycle
        valid = True
        cycle_edges = []
        for i in range(len(best_cycle)):
            u = best_cycle[i]
            v = best_cycle[(i + 1) % len(best_cycle)]
            edge = tuple(sorted((u, v)))
            if edge not in remaining:
                valid = False
                break
            cycle_edges.append(edge)
        
        if not valid:
            break
        
        for edge in cycle_edges:
            remaining.discard(edge)
            u, v = edge
            adj[u].discard(v)
            adj[v].discard(u)
        
        pieces.append(best_cycle[:])
        iters += 1
    
    # Phase 3: Remaining edges become single-edge pieces
    for u, v in remaining:
        pieces.append([u, v])
    
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                s = specs[len(results)]
                results.append([list(e) for e in _instance(*s)])
            break
        
        edges = _instance(*spec)
        n = spec[0]
        rng = Random(42 + spec[2])
        pieces = _decompose(edges, n, rng)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
