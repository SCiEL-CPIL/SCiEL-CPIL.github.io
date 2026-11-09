"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict, deque
from random import Random

def _instance(n, p, seed):
    """Reproduce the verifier's instance graph exactly."""
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _path_cover_greedy_longest(edges, n, max_iter=50000):
    """
    Greedy path cover: repeatedly find the longest simple path in the
    remaining edge set, remove its edges, repeat until no edges remain.
    Uses DFS with backtracking for longest path (feasible for small graphs).
    """
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    paths = []
    remaining = set(edges)
    iter_count = 0

    while remaining and iter_count < max_iter:
        iter_count += 1
        # Find all vertices with remaining edges
        active = set()
        for u, v in remaining:
            active.add(u)
            active.add(v)
        if not active:
            break

        best_path = None
        best_len = 0

        # Try starting from each active vertex
        for start in sorted(active):
            # DFS to find longest simple path starting from 'start'
            best_from_start = [start]

            def dfs(path, visited):
                nonlocal best_from_start, best_len
                if len(path) > len(best_from_start):
                    best_from_start = list(path)
                for nb in sorted(adj.get(path[-1], [])):
                    if nb not in visited and (tuple(sorted((path[-1], nb)))) in remaining:
                        path.append(nb)
                        visited.add(nb)
                        dfs(path, visited)
                        path.pop()
                        visited.discard(nb)

            dfs([start], {start})
            if len(best_from_start) > best_len:
                best_len = len(best_from_start)
                best_path = best_from_start

        if best_path is None or len(best_path) < 2:
            # Single edge remaining
            for u, v in remaining:
                best_path = [u, v]
                break
            if best_path is None:
                break

        # Remove edges from remaining
        for i in range(len(best_path) - 1):
            u, v = best_path[i], best_path[i + 1]
            edge = tuple(sorted((u, v)))
            remaining.discard(edge)
            adj[u].discard(v)
            adj[v].discard(u)

        paths.append(best_path)

    # If any edges remain (shouldn't happen), add them as single-edge paths
    for u, v in remaining:
        paths.append([u, v])

    return paths


def _path_cover_lp_based(edges, n):
    """
    LP-based approach: model as min-cost flow to find minimum number of paths.
    Each undirected edge becomes two directed arcs. Node balances enforce
    path endpoints. The minimum number of paths = (number of odd-degree vertices)/2
    for a connected graph, but we need to actually construct the paths.
    
    For a connected graph, the minimum number of edge-disjoint paths covering all edges
    is max(1, odd_count / 2) where odd_count is the number of odd-degree vertices.
    We can achieve this by finding a pairing of odd vertices and adding "virtual" 
    edges to make all degrees even, then finding an Eulerian circuit, then removing
    the virtual edges to split into paths.
    """
    # Compute degrees
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
    
    if len(odd_vertices) == 0:
        # All even: find Eulerian circuit, return as one path
        return [_eulerian_path(edges, n)]
    
    # Pair up odd vertices (greedy pairing)
    # Sort odd vertices and pair consecutive ones
    odd_vertices.sort()
    pairs = []
    for i in range(0, len(odd_vertices), 2):
        if i + 1 < len(odd_vertices):
            pairs.append((odd_vertices[i], odd_vertices[i + 1]))
    
    # Add virtual edges to make all degrees even
    all_edges = list(edges)
    for u, v in pairs:
        all_edges.append((min(u, v), max(u, v)))
    
    # Find Eulerian circuit in the augmented graph
    circuit = _eulerian_circuit(all_edges, n)
    
    # Remove virtual edges to get paths
    # The circuit is a list of vertices. Virtual edges connect paired odd vertices.
    # When we traverse a virtual edge, we "break" the circuit there.
    virtual_edge_set = {tuple(sorted((u, v))) for u, v in pairs}
    
    paths = []
    current_path = []
    
    for i in range(len(circuit)):
        current_path.append(circuit[i])
        if i < len(circuit) - 1:
            u, v = circuit[i], circuit[i + 1]
            if tuple(sorted((u, v))) in virtual_edge_set:
                if len(current_path) >= 2:
                    paths.append(current_path)
                current_path = [circuit[i + 1]]
    
    if len(current_path) >= 2:
        paths.append(current_path)
    
    # Handle edge case: if circuit wraps around
    # Actually the Eulerian circuit is a cycle, so we need to handle the wrap-around
    # Let me redo this more carefully
    
    # Re-do: find Eulerian circuit as a cycle
    # The circuit visits all edges exactly once and returns to start
    # Virtual edges split it into paths
    
    return paths


def _eulerian_path(edges, n):
    """Find an Eulerian path (not necessarily circuit) in a graph."""
    if not edges:
        return [0]
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find start vertex (odd degree if exists, else any)
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    start = 0
    for v in range(n):
        if deg[v] % 2 == 1:
            start = v
            break
    
    # Hierholzer's algorithm
    path = []
    stack = [start]
    while stack:
        v = stack[-1]
        if adj[v]:
            nb = adj[v].pop()
            adj[nb].remove(v)
            stack.append(nb)
        else:
            path.append(stack.pop())
    
    path.reverse()
    return path


def _eulerian_circuit(edges, n):
    """Find an Eulerian circuit in a graph where all vertices have even degree."""
    if not edges:
        return [0]
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    start = edges[0][0]
    
    # Hierholzer's algorithm
    circuit = []
    stack = [start]
    while stack:
        v = stack[-1]
        if adj[v]:
            nb = adj[v].pop()
            adj[nb].remove(v)
            stack.append(nb)
        else:
            circuit.append(stack.pop())
    
    circuit.reverse()
    return circuit


def _path_cover_lp_optimal(edges, n):
    """
    Optimal path cover using the fact that min # paths = max(1, odd/2).
    Pair odd vertices optimally (shortest paths between pairs), add virtual edges,
    find Eulerian circuit, split at virtual edges.
    
    For small graphs, we can try all pairings or use a simple heuristic.
    """
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
    
    if len(odd_vertices) == 0:
        return [_eulerian_path(edges, n)]
    
    # Greedy pairing: sort by vertex id and pair consecutive
    # This is suboptimal but fast. For better results, try multiple pairings.
    odd_vertices.sort()
    pairs = []
    for i in range(0, len(odd_vertices) - 1, 2):
        pairs.append((odd_vertices[i], odd_vertices[i + 1]))
    
    # Create augmented graph
    all_edges = list(edges)
    virtual_pairs = []
    for u, v in pairs:
        all_edges.append((min(u, v), max(u, v)))
        virtual_pairs.append(tuple(sorted((u, v))))
    
    virtual_set = set(virtual_pairs)
    
    # Find Eulerian circuit
    circuit = _eulerian_circuit(all_edges, n)
    
    # Split circuit at virtual edges
    paths = []
    current = []
    
    for i in range(len(circuit)):
        current.append(circuit[i])
        if i < len(circuit) - 1:
            u, v = circuit[i], circuit[i + 1]
            if tuple(sorted((u, v))) in virtual_set:
                if len(current) >= 2:
                    paths.append(current)
                current = [circuit[i + 1]]
    
    if len(current) >= 2:
        paths.append(current)
    
    # Handle wrap-around: if the circuit starts and ends at same vertex,
    # and there's a virtual edge crossing the boundary
    if len(circuit) > 1 and circuit[0] == circuit[-1]:
        # Check if first and last segment crosses a virtual edge
        # Actually in a circuit, circuit[0] == circuit[-1] is expected
        # The splitting should handle it correctly
        
        # But we might have an issue: the last vertex is the same as the first,
        # so the "current" at the end might be incomplete
        pass
    
    return paths


def _validate_and_fix(paths, edges, n):
    """Validate paths and fix any issues."""
    edge_set = set(tuple(sorted(e)) for e in edges)
    used_edges = set()
    valid_paths = []
    
    for path in paths:
        if len(path) < 2:
            continue
        # Check simplicity
        if len(set(path)) != len(path):
            continue
        # Check edges exist and are unused
        valid = True
        path_edges = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            e = tuple(sorted((u, v)))
            if e not in edge_set or e in used_edges:
                valid = False
                break
            path_edges.append(e)
        if valid:
            used_edges.update(path_edges)
            valid_paths.append(path)
    
    # Add any missing edges as single-edge paths
    for e in edge_set:
        if e not in used_edges:
            valid_paths.append(list(e))
    
    return valid_paths


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for n, p, seed in specs:
        if time.time() > deadline - 10:
            # Not enough time, use simple fallback
            edges = _instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        
        edges = _instance(n, p, seed)
        
        # Try LP-based optimal path cover
        try:
            paths = _path_cover_lp_optimal(edges, n)
            paths = _validate_and_fix(paths, edges, n)
            results.append(paths)
        except Exception:
            # Fallback to greedy
            try:
                paths = _path_cover_greedy_longest(edges, n)
                paths = _validate_and_fix(paths, edges, n)
                results.append(paths)
            except Exception:
                results.append([list(e) for e in edges])
    
    return results
# EVOLVE-BLOCK-END
