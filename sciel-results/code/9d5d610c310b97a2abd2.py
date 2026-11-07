"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

import time
import random
import sys

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _cycle_edges(cycle):
    edges = set()
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        edges.add((min(u, v), max(u, v)))
    return edges

def _find_ham_cycle(adj, n, forbidden_edges, start, time_limit=5.0):
    """Find a Hamiltonian cycle using DFS with Warnsdorff heuristic."""
    t_start = time.time()
    adj_lists = [sorted(adj[i]) for i in range(n)]
    
    def dfs(path, visited, used_edges):
        if time.time() - t_start > time_limit:
            return None
        if len(path) == n:
            last = path[-1]
            first = path[0]
            edge = (min(last, first), max(last, first))
            if first in adj[last] and edge not in forbidden_edges and edge not in used_edges:
                return list(path)
            return None
        
        curr = path[-1]
        candidates = []
        for v in adj_lists[curr]:
            if not visited[v]:
                edge = (min(curr, v), max(curr, v))
                if edge not in forbidden_edges and edge not in used_edges:
                    candidates.append(v)
        
        if not candidates:
            return None
        
        # Warnsdorff: prefer vertices with fewer unvisited neighbors
        def score(v):
            count = 0
            for u in adj_lists[v]:
                if not visited[u] and u != curr:
                    edge = (min(v, u), max(v, u))
                    if edge not in forbidden_edges and edge not in used_edges:
                        count += 1
            return count
        
        candidates.sort(key=score)
        
        for v in candidates:
            edge = (min(curr, v), max(curr, v))
            path.append(v)
            visited[v] = True
            used_edges.add(edge)
            result = dfs(path, visited, used_edges)
            if result is not None:
                return result
            path.pop()
            visited[v] = False
            used_edges.discard(edge)
        return None
    
    path = [start]
    visited = [False] * n
    visited[start] = True
    used_edges = set()
    return dfs(path, visited, used_edges)

def construct():
    t0 = time.time()
    deadline = t0 + 80
    n = P['n']
    adj = _build_adj()
    
    best_cycles = []
    max_cycles = 3  # floor(63/21) = 3, graph is 6-regular with 63 edges
    
    # Strategy: try multiple starting points with increasing time budgets
    starts = list(range(n))
    rng = random.Random(42)
    
    for attempt in range(200):
        if time.time() > deadline - 3:
            break
        if len(best_cycles) >= max_cycles:
            break
        
        # Choose start vertex
        if attempt < n:
            start = attempt
        else:
            start = rng.randint(0, n - 1)
        
        # Build forbidden edge set from found cycles
        forbidden = set()
        for cyc in best_cycles:
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i + 1) % len(cyc)]
                forbidden.add((min(u, v), max(u, v)))
        
        # Time budget: more time for later cycles (harder)
        remaining = deadline - time.time() - 3
        if remaining < 0.5:
            break
        tl = min(remaining, 2.0 + 0.5 * len(best_cycles))
        
        cycle = _find_ham_cycle(adj, n, forbidden, start, tl)
        if cycle is not None:
            # Validate
            valid = True
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if v not in adj[u]:
                    valid = False
                    break
            if valid and len(set(cycle)) == n:
                # Check edge-disjointness
                ce = _cycle_edges(cycle)
                disjoint = True
                for existing in best_cycles:
                    if ce & _cycle_edges(existing):
                        disjoint = False
                        break
                if disjoint:
                    best_cycles.append(cycle)
    
    # Fallback: if we have fewer than 3, try harder with different starts
    if len(best_cycles) < max_cycles and time.time() < deadline - 5:
        for start in range(n):
            if time.time() > deadline - 3:
                break
            if len(best_cycles) >= max_cycles:
                break
            forbidden = set()
            for cyc in best_cycles:
                for i in range(len(cyc)):
                    u = cyc[i]
                    v = cyc[(i + 1) % len(cyc)]
                    forbidden.add((min(u, v), max(u, v)))
            cycle = _find_ham_cycle(adj, n, forbidden, start, 5.0)
            if cycle is not None:
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if v not in adj[u]:
                        valid = False
                        break
                if valid and len(set(cycle)) == n:
                    ce = _cycle_edges(cycle)
                    disjoint = True
                    for existing in best_cycles:
                        if ce & _cycle_edges(existing):
                            disjoint = False
                            break
                    if disjoint:
                        best_cycles.append(cycle)
    
    if not best_cycles:
        # Absolute fallback: return a single valid cycle
        for start in range(n):
            cycle = _find_ham_cycle(adj, n, set(), start, 10.0)
            if cycle is not None:
                return [cycle]
        return [list(range(n))]
    
    return best_cycles
# EVOLVE-BLOCK-END
