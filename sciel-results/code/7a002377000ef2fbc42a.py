"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
from collections import deque
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _enumerate_cycles(adj, n):
    """
    Enumerate all simple cycles in the graph.
    Each cycle is represented as a list of vertices [v0, v1, ..., vk-1]
    where v0 is the smallest vertex in the cycle.
    """
    cycles = []
    # To avoid duplicates, we only start a cycle from vertex u if u is the smallest
    # vertex in that cycle. This is achieved by only traversing to neighbors > u
    # during the search, except for the closing edge back to u.
    
    for u in range(n):
        # Stack for DFS: (current_node, path)
        stack = [(u, [u])]
        while stack:
            node, path = stack.pop()
            # Try to extend path
            for neighbor in adj[node]:
                if neighbor == u:
                    # Close the cycle
                    if len(path) >= 3:
                        # Check if u is the smallest vertex
                        if all(v >= u for v in path):
                            cycles.append(path[:])
                elif neighbor > u:
                    # Only visit vertices larger than u to ensure u is min
                    if neighbor not in path:
                        stack.append((neighbor, path + [neighbor]))
    return cycles

def _decompose_ilp(edges, n):
    """
    Decompose edges into cycles and single edges using ILP on enumerated cycles.
    """
    if not edges:
        return []
        
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Enumerate cycles
    cycles = _enumerate_cycles(adj, n)
    
    # If no cycles found, return all edges as pieces
    if not cycles:
        return [list(e) for e in edges]
        
    # Map edges to indices
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    num_edges = len(edges)
    num_cycles = len(cycles)
    
    # Build constraint matrix A and costs c
    # We want to minimize -sum(len(cycle)-1)
    # A[i, j] = 1 if edge i is in cycle j
    A = np.zeros((num_edges, num_cycles))
    costs = np.zeros(num_cycles)
    
    for j, cycle in enumerate(cycles):
        costs[j] = -(len(cycle) - 1)
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge = tuple(sorted((u, v)))
            if edge in edge_to_idx:
                A[edge_to_idx[edge], j] = 1
            else:
                # Should not happen if enumeration is correct
                pass
                
    # Define problem
    # Minimize c @ y
    # Subject to A @ y <= 1
    # y in {0, 1}
    
    constraints = LinearConstraint(A, ub=1)
    integrality = np.ones(num_cycles)
    bounds = Bounds(lb=0, ub=1)
    
    # Solve
    res = milp(c=costs, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if not res.success:
        # Fallback to greedy if ILP fails
        return _decompose_greedy_fallback(edges, n)
        
    # Extract selected cycles
    selected_cycles = []
    for j in range(num_cycles):
        if res.x[j] > 0.5:
            selected_cycles.append(cycles[j])
            
    # Determine used edges
    used_edges = set()
    for cycle in selected_cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            used_edges.add(tuple(sorted((u, v))))
            
    # Construct pieces
    pieces = []
    for cycle in selected_cycles:
        pieces.append(cycle[:])
        
    for e in edges:
        if e not in used_edges:
            pieces.append(list(e))
            
    return pieces

def _decompose_greedy_fallback(edges, n):
    """
    Simple greedy fallback if ILP fails or takes too long.
    """
    pieces = []
    remaining = set(tuple(sorted(e)) for e in edges)
    adj = {i: set() for i in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        
    # Greedy: find any cycle, remove, repeat
    # Use a simple DFS to find a cycle
    def find_cycle(start):
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
        return None

    while remaining:
        # Find a start vertex with degree >= 2
        starts = [v for v in range(n) if len(adj[v]) >= 2]
        if not starts:
            break
        # Pick one
        start = starts[0]
        cycle = find_cycle(start)
        if cycle:
            # Remove edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
            pieces.append(cycle)
        else:
            # No more cycles, break
            break
            
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
            # Fallback: return all edges as pieces
            while len(results) < 3:
                s = specs[len(results)]
                results.append([list(e) for e in _instance(*s)])
            break
        
        edges = _instance(*spec)
        n = spec[0]
        
        # Try ILP decomposition
        try:
            pieces = _decompose_ilp(edges, n)
        except Exception:
            pieces = _decompose_greedy_fallback(edges, n)
            
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
