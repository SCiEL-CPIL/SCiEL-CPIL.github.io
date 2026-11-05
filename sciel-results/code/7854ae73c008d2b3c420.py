import time
import random
import sys
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_all_cycles(adj, n, max_cycles=50000):
    """
    Find all simple cycles in the graph.
    Returns a list of cycles, where each cycle is a list of vertices.
    """
    cycles = []
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_set.add((u, v))
    
    # Map edge to index for faster lookup if needed, but frozenset is fine for small n
    # We need to avoid duplicate cycles.
    # A cycle is unique if we start at the smallest vertex and the second vertex is smaller than the last.
    
    def dfs(start, current, path, visited):
        if len(cycles) >= max_cycles:
            return
        
        if current == start and len(path) >= 3:
            # Normalize cycle
            min_idx = path.index(min(path))
            cycle = path[min_idx:] + path[:min_idx]
            # Check orientation: cycle[1] < cycle[-1]
            if len(cycle) >= 3:
                if cycle[1] < cycle[-1]:
                    cycles.append(cycle)
            return
        
        for neighbor in adj[current]:
            if neighbor == start and len(path) >= 3:
                dfs(start, start, path, visited)
            elif neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(start, neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)

    for start in range(n):
        if not adj[start]:
            continue
        visited = {start}
        path = [start]
        dfs(start, start, path, visited)
        
    return cycles

def _solve_one(edges, n, deadline):
    start_time = time.time()
    if time.time() > deadline:
        return [[u, v] for u, v in edges]
        
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find all cycles
    try:
        cycles = _find_all_cycles(adj, n)
    except Exception:
        cycles = []
        
    if not cycles:
        return [[u, v] for u, v in edges]
        
    # Prepare ILP
    num_cycles = len(cycles)
    num_edges = len(edges)
    
    # Map edge to index
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        
    # Objective: Maximize sum(|c| - 1) * x_c
    # Minimize sum((1 - |c|) * x_c)
    c_obj = np.array([1 - len(c) for c in cycles], dtype=float)
    
    # Constraints: For each edge, sum(x_c for c containing edge) <= 1
    # A[i, j] = 1 if cycle j contains edge i
    # We can use sparse matrix
    from scipy.sparse import lil_matrix
    A = lil_matrix((num_edges, num_cycles), dtype=float)
    
    for j, cycle in enumerate(cycles):
        m = len(cycle)
        for i in range(m):
            u = cycle[i]
            v = cycle[(i + 1) % m]
            e = tuple(sorted((u, v)))
            idx = edge_to_idx.get(e)
            if idx is not None:
                A[idx, j] = 1.0
                
    A = A.tocsr()
    
    # Bounds: x_c in [0, 1]
    bounds = Bounds(lb=0, ub=1)
    
    # Constraints: A @ x <= 1
    constraint = LinearConstraint(A, lb=0, ub=1)
    
    try:
        res = milp(c=c_obj, constraints=constraint, integrality=np.ones(num_cycles), bounds=bounds)
        if res.status == 0:
            selected_cycles = [cycles[j] for j in range(num_cycles) if res.x[j] > 0.5]
            
            # Determine remaining edges
            used_edges = set()
            for cycle in selected_cycles:
                m = len(cycle)
                for i in range(m):
                    u = cycle[i]
                    v = cycle[(i + 1) % m]
                    used_edges.add(tuple(sorted((u, v))))
                    
            pieces = []
            for cycle in selected_cycles:
                pieces.append(cycle)
                
            for u, v in edges:
                if (u, v) not in used_edges:
                    pieces.append([u, v])
                    
            return pieces
        else:
            # Fallback to greedy if ILP fails
            pass
    except Exception:
        pass
        
    # Fallback: Greedy
    return _greedy_decompose(edges, n, deadline)

def _greedy_decompose(edges, n, deadline):
    # Simple greedy: find longest cycle, remove, repeat
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    pieces = []
    remaining = len(edges)
    
    while remaining > 0 and time.time() < deadline:
        # Find a cycle
        best_cycle = None
        for start in range(n):
            if not adj[start]:
                continue
            # DFS to find a cycle
            visited = {start}
            path = [start]
            def dfs(current):
                for neighbor in adj[current]:
                    if neighbor == start and len(path) >= 3:
                        return path[:]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        path.append(neighbor)
                        res = dfs(neighbor)
                        if res:
                            return res
                        path.pop()
                        visited.remove(neighbor)
                return None
            
            cycle = dfs(start)
            if cycle:
                if not best_cycle or len(cycle) > len(best_cycle):
                    best_cycle = cycle
        if not best_cycle:
            # No cycle found, take an edge
            for u in range(n):
                for v in list(adj[u]):
                    if u < v:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        remaining -= 1
                        pieces.append([u, v])
                        break
                if remaining == 0:
                    break
            if remaining == 0:
                break
            continue
            
        # Remove cycle
        m = len(best_cycle)
        for i in range(m):
            u = best_cycle[i]
            v = best_cycle[(i + 1) % m]
            adj[u].discard(v)
            adj[v].discard(u)
        remaining -= m
        pieces.append(best_cycle)
        
    # Add remaining edges
    for u in range(n):
        for v in list(adj[u]):
            if u < v:
                pieces.append([u, v])
                
    return pieces

def construct():
    start_time = time.time()
    total_budget = 145.0
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
