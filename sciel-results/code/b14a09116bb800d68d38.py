"""Decompose three fixed graphs into cycles and edges, minimizing piece count.
Strategy: Use maximum 2-matching via LP on bipartite double cover to find
maximum collection of edge-disjoint cycles, then handle remaining edges.
"""
import time
import random
from collections import defaultdict, deque

_DEADLINE_SECONDS = 140.0


def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _max_2matching_lp(n, edges):
    """
    Find maximum 2-matching using LP on bipartite double cover.
    
    For each undirected edge (u,v), create two directed arcs: u->v and v->u.
    Variables x_{u,v} in [0,1] for each directed arc.
    
    Maximize: sum of x_{u,v} (counts each undirected edge at most once in a 2-factor)
    
    Constraints:
      - For each vertex v: sum_{u: (u,v) in E} x_{u,v} <= 2 (in-degree <= 2)
      - For each vertex v: sum_{u: (v,u) in E} x_{v,u} <= 2 (out-degree <= 2)
      - For each undirected edge (u,v): x_{u,v} + x_{v,u} <= 1 (at most one direction)
    
    This gives a 2-factor (collection of cycles) in the original graph.
    """
    from scipy.optimize import linprog
    
    m = len(edges)
    # Directed arcs: for each undirected edge i, we have arc i (u->v) and arc m+i (v->u)
    # Variables: x[0..2m-1]
    
    # Objective: maximize sum of x = minimize -sum(x)
    c = [-1.0] * (2 * m)
    
    # Inequality constraints: A_ub @ x <= b_ub
    # For each vertex, in-degree <= 2 and out-degree <= 2
    # For each edge, x_forward + x_backward <= 1
    
    # Build constraint matrix
    # We'll use sparse representation
    import numpy as np
    
    # Count constraints:
    # 2*n for degree constraints (in + out per vertex)
    # m for edge constraints
    num_constraints = 2 * n + m
    b_ub = [2.0] * (2 * n) + [1.0] * m
    
    # Build A_ub as list of (row, col, val) for sparse
    from scipy.sparse import csr_matrix
    
    rows = []
    cols = []
    vals = []
    
    # Map each vertex to its incident edges
    vert_edges = defaultdict(list)  # vertex -> list of (edge_idx, is_forward)
    for i, (u, v) in enumerate(edges):
        # Arc i: u -> v
        vert_edges[u].append((i, 'out'))  # u is source
        vert_edges[v].append((i, 'in'))   # v is target
        # Arc m+i: v -> u
        vert_edges[v].append((m + i, 'out'))  # v is source
        vert_edges[u].append((m + i, 'in'))   # u is target
    
    # In-degree constraints: for vertex v, sum of x_{arc} where v is target <= 2
    # These are rows 0..n-1
    for v in range(n):
        for (ei, direction) in vert_edges[v]:
            if direction == 'in':
                rows.append(v)
                cols.append(ei)
                vals.append(1.0)
    
    # Out-degree constraints: for vertex v, sum of x_{arc} where v is source <= 2
    # These are rows n..2n-1
    for v in range(n):
        for (ei, direction) in vert_edges[v]:
            if direction == 'out':
                rows.append(n + v)
                cols.append(ei)
                vals.append(1.0)
    
    # Edge constraints: for edge i, x_i + x_{m+i} <= 1
    # These are rows 2n..2n+m-1
    for i in range(m):
        rows.append(2 * n + i)
        cols.append(i)
        vals.append(1.0)
        rows.append(2 * n + i)
        cols.append(m + i)
        vals.append(1.0)
    
    A_ub = csr_matrix((vals, (rows, cols)), shape=(num_constraints, 2 * m))
    
    bounds = [(0, 1)] * (2 * m)
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        return set()  # No valid 2-matching found
    
    # Extract selected undirected edges
    selected_edges = set()
    for i in range(m):
        u, v = edges[i]
        if result.x[i] > 0.5 or result.x[m + i] > 0.5:
            selected_edges.add((u, v))
    
    return selected_edges


def _extract_cycles_from_2factor(n, selected_edges):
    """
    Given a set of edges forming a 2-factor (each vertex has degree 0 or 2),
    extract the individual cycles.
    """
    if not selected_edges:
        return []
    
    adj = defaultdict(list)
    for u, v in selected_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    cycles = []
    visited = set()
    
    for start in range(n):
        if start in visited or start not in adj:
            continue
        # Follow the cycle
        cycle = [start]
        visited.add(start)
        prev = -1
        curr = start
        while True:
            nexts = [nb for nb in adj[curr] if nb != prev]
            if not nexts:
                break
            nxt = nexts[0]
            if nxt == start:
                break
            if nxt in visited:
                break
            cycle.append(nxt)
            visited.add(nxt)
            prev = curr
            curr = nxt
        if len(cycle) >= 3:
            cycles.append(cycle)
    
    return cycles


def _greedy_longest_cycle(edges, n):
    """
    Greedy: repeatedly find and remove the longest cycle.
    This tends to produce fewer, longer cycles.
    """
    adj = defaultdict(set)
    deg = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    pieces = []
    remaining_edges = set(edges)
    
    while remaining_edges:
        # Find longest cycle using DFS with backtracking (limited)
        best_cycle = None
        
        # Try starting from high-degree vertices
        candidates = [v for v in range(n) if deg[v] >= 2]
        candidates.sort(key=lambda v: -deg[v])
        
        for start in candidates[:min(len(candidates), 15)]:
            if deg[start] < 2:
                continue
            
            # DFS to find longest cycle starting from start
            stack = [(start, [start], -1)]
            while stack:
                node, path, prev_idx = stack.pop()
                neighbors = [nb for nb in adj[node] if nb != path[-2] if len(path) > 1] if len(path) > 1 else list(adj[node])
                
                # Actually, let me redo this more carefully
                pass
            
            # Simpler: use a proper longest-cycle search via DFS
            # Find all simple cycles of length >= 3 containing start
            # Use iterative deepening or just find any cycle and try to extend
            
            # For efficiency, find cycles via DFS
            found = _find_longest_cycle_dfs(adj, deg, start, remaining_edges)
            if found and (best_cycle is None or len(found) > len(best_cycle)):
                best_cycle = found
        
        if best_cycle:
            pieces.append(best_cycle)
            # Remove edges from cycle
            for i in range(len(best_cycle)):
                u, v = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
                e = tuple(sorted((u, v)))
                remaining_edges.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
                deg[u] -= 1
                deg[v] -= 1
        else:
            # No more cycles; output remaining edges individually
            for e in remaining_edges:
                pieces.append(list(e))
            break
    
    return pieces


def _find_longest_cycle_dfs(adj, deg, start, edge_set):
    """Find a long cycle starting from start using DFS."""
    best = None
    
    # DFS with path tracking
    path = [start]
    on_path = {start}
    
    def dfs(node, depth):
        nonlocal best
        for nb in adj[node]:
            if nb == start and len(path) >= 3:
                if best is None or len(path) > len(best):
                    best = list(path)
            elif nb not in on_path and deg[nb] >= 2:
                on_path.add(nb)
                path.append(nb)
                dfs(nb, depth + 1)
                path.pop()
                on_path.discard(nb)
    
    dfs(start, 0)
    return best


def _decompose_with_2matching(edges, n, time_limit):
    """
    Decompose using maximum 2-matching approach:
    1. Find max 2-factor (cycles) via LP
    2. Extract cycles
    3. Handle remaining edges with greedy longest-cycle
    4. Post-processing: try to merge small cycles/edges into larger cycles
    """
    start_time = time.time()
    
    # Step 1: Find maximum 2-factor
    selected = _max_2matching_lp(n, edges)
    
    # Step 2: Extract cycles from 2-factor
    cycles = _extract_cycles_from_2factor(n, selected)
    
    # Step 3: Remaining edges
    remaining = set(edges) - set(selected)
    
    # Step 4: Greedy longest cycle on remaining
    if remaining:
        more_pieces = _greedy_longest_cycle(list(remaining), n)
        cycles.extend(more_pieces)
    
    # Step 5: Post-processing - try to improve
    # Check if any single edges can be merged into cycles
    pieces = cycles
    pieces = _post_process_merge(pieces, n, set(edges), time_limit - (time.time() - start_time))
    
    return pieces


def _post_process_merge(pieces, n, edge_set, time_limit):
    """
    Try to merge edge-pieces into cycles, and merge cycles if possible.
    """
    start_time = time.time()
    
    # Separate cycles and edges
    cycle_pieces = [p for p in pieces if len(p) >= 3]
    edge_pieces = [p for p in pieces if len(p) == 2]
    
    if not edge_pieces:
        return pieces
    
    # Build adjacency for edge pieces
    edge_set_remaining = set(tuple(sorted(e)) for e in edge_pieces)
    
    # Try to find triangles among edge pieces
    vert_to_edges = defaultdict(list)
    for i, (u, v) in enumerate(edge_pieces):
        vert_to_edges[u].append(i)
        vert_to_edges[v].append(i)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Try to find 3 edge-pieces forming a triangle
        for vert in range(n):
            if time.time() - start_time > time_limit:
                break
            eps_at_vert = vert_to_edges.get(vert, [])
            if len(eps_at_vert) < 2:
                continue
            
            for a_idx in range(len(eps_at_vert)):
                for b_idx in range(a_idx + 1, len(eps_at_vert)):
                    ia = eps_at_vert[a_idx]
                    ib = eps_at_vert[b_idx]
                    if ia >= len(edge_pieces) or ib >= len(edge_pieces):
                        continue
                    ua, va = edge_pieces[ia]
                    ub, vb = edge_pieces[ib]
                    
                    if vert == ua:
                        other_a = va
                    else:
                        other_a = ua
                    if vert == ub:
                        other_b = vb
                    else:
                        other_b = ub
                    
                    if other_a == other_b:
                        continue
                    
                    e_ab = tuple(sorted((other_a, other_b)))
                    # Check if e_ab is in edge_pieces
                    for ic, (xc, yc) in enumerate(edge_pieces):
                        if tuple(sorted((xc, yc))) == e_ab and ic != ia and ic != ib:
                            # Found triangle!
                            cycle = [vert, other_a, other_b]
                            # Verify
                            e1 = tuple(sorted((vert, other_a)))
                            e2 = tuple(sorted((other_a, other_b)))
                            e3 = tuple(sorted((other_b, vert)))
                            if e1 in edge_set and e2 in edge_set and e3 in edge_set:
                                # Replace 3 edges with 1 cycle
                                edge_pieces.remove(edge_pieces[ia])
                                # Need to re-find indices after removal
                                edge_pieces = [ep for ep in edge_pieces if ep not in [edge_pieces[ib], edge_pieces[ic]]]
                                # Rebuild vert_to_edges
                                vert_to_edges = defaultdict(list)
                                for i, (u, v) in enumerate(edge_pieces):
                                    vert_to_edges[u].append(i)
                                    vert_to_edges[v].append(i)
                                cycle_pieces.append(cycle)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        # Try to find 4-cycles among edge pieces (2 edges sharing a vertex, 
        # and 2 more edges completing the cycle)
        if not improved and len(edge_pieces) >= 4:
            # For pairs of non-adjacent edges, check if they can form a 4-cycle
            # This is more complex; skip for now
    
    return cycle_pieces + edge_pieces


def _decompose_graph(edges, n, time_limit):
    """Main decomposition: try 2-matching approach, fallback to greedy."""
    start_time = time.time()
    
    try:
        pieces = _decompose_with_2matching(edges, n, time_limit)
    except Exception:
        pieces = _greedy_longest_cycle(edges, n)
    
    return pieces


def construct():
    """Decompose three fixed graphs into cycles and edges, minimizing piece count."""
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for i, (n, p, seed) in enumerate(specs):
        elapsed = time.time() - start_time
        remaining = _DEADLINE_SECONDS - elapsed
        if remaining < 5:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        graphs_left = len(specs) - i
        graph_time = remaining * 0.9 / max(graphs_left, 1)

        pieces = _decompose_graph(edges, n, graph_time)
        results.append(pieces)

    return results
