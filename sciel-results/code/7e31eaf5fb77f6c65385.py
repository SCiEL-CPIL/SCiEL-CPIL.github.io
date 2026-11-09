"""Decompose graphs into few cycles and edges using maximum 2-factor + augmentation."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from random import Random
from scipy.optimize import linear_sum_assignment

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _max_2factor(adj, n, rng):
    """Find a maximum 2-factor using bipartite matching on the directed version.
    
    Left: n nodes (u_out), Right: n nodes (v_in).
    Arc u->v exists if (u,v) is an edge.
    Cost: 0 if arc exists, large penalty if not.
    We want to maximize the number of matched arcs (minimize unmatched).
    """
    INF = 10**9
    cost = [[INF] * n for _ in range(n)]
    
    for u in range(n):
        for v in adj[u]:
            if u != v:
                cost[u][v] = 0
    
    row_ind, col_ind = linear_sum_assignment(cost)
    
    matched = {}
    num_matched = 0
    for i, j in zip(row_ind, col_ind):
        if cost[i][j] < INF:
            matched[i] = j
            num_matched += 1
    
    if num_matched == 0:
        return []
    
    # Extract directed cycles
    visited = set()
    cycles = []
    for start in range(n):
        if start in visited or start not in matched:
            continue
        path = [start]
        visited.add(start)
        curr = start
        while True:
            nxt = matched.get(curr)
            if nxt is None:
                break
            if nxt == start:
                if len(path) >= 3:
                    cycles.append(path[:])
                break
            if nxt in visited:
                break
            path.append(nxt)
            visited.add(nxt)
            curr = nxt
    
    return cycles


def _find_best_cycle_in_subgraph(adj, n, rng, max_attempts=50):
    """Find a long cycle in the subgraph using randomized DFS."""
    best = None
    best_len = 0
    
    candidates = [v for v in range(n) if len(adj[v]) >= 2]
    if not candidates:
        return None
    
    for _ in range(max_attempts):
        start = rng.choice(candidates)
        cycle = _dfs_cycle(adj, n, start, rng)
        if cycle and len(cycle) > best_len:
            best_len = len(cycle)
            best = cycle
    
    return best


def _dfs_cycle(adj, n, start, rng, max_depth=80):
    """Find a cycle through start using randomized DFS."""
    visited = {start}
    path = [start]
    stack = [(start, list(adj[start]))]
    
    while stack:
        node, neighbors = stack[-1]
        if not neighbors:
            stack.pop()
            path.pop()
            continue
        nxt = neighbors.pop()
        if nxt == start and len(path) >= 3:
            return list(path)
        if nxt not in visited:
            visited.add(nxt)
            path.append(nxt)
            stack.append((nxt, list(adj[nxt])))
            if len(path) > max_depth:
                stack.pop()
                path.pop()
    
    return None


def _decompose(edges, n, deadline):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    remaining = set(tuple(sorted(e)) for e in edges)
    adj = {i: set() for i in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    rng = Random(42)
    pieces = []
    
    # Phase 1: Find maximum 2-factor (cycle cover)
    cycles = _max_2factor(adj, n, rng)
    
    for cycle in cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge = tuple(sorted((u, v)))
            remaining.discard(edge)
            adj[u].discard(v)
            adj[v].discard(u)
    
    pieces.extend([c[:] for c in cycles])
    
    # Phase 2: Iteratively find cycles in remaining edges to replace single edges
    max_rounds = 100
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        if len(remaining) < 3:
            break
        
        sub_adj = {i: set() for i in range(n)}
        for u, v in remaining:
            sub_adj[u].add(v)
            sub_adj[v].add(u)
        
        found_cycle = _find_best_cycle_in_subgraph(sub_adj, n, rng)
        if found_cycle and len(found_cycle) >= 3:
            cycle_edges = set()
            valid = True
            for i in range(len(found_cycle)):
                u = found_cycle[i]
                v = found_cycle[(i + 1) % len(found_cycle)]
                edge = tuple(sorted((u, v)))
                if edge in remaining:
                    cycle_edges.add(edge)
                else:
                    valid = False
                    break
            
            if valid and len(cycle_edges) >= 3:
                for edge in cycle_edges:
                    remaining.discard(edge)
                    u, v = edge
                    adj[u].discard(v)
                    adj[v].discard(u)
                pieces.append(found_cycle[:])
            else:
                break
        else:
            break
    
    # Remaining edges become single-edge pieces
    for u, v in remaining:
        pieces.append([u, v])
    
    return pieces


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
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
        pieces = _decompose(edges, n, deadline)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
