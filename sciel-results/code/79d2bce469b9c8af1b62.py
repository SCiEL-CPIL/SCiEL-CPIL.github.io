"""Decomposition of random graphs into few cycles and edges via iterative 2-factor extraction."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_longest_cycle(adj, n, max_depth=30):
    """
    Find the longest cycle in the graph defined by adj (dict of sets).
    Uses DFS with backtracking, trying to find cycles starting from each vertex.
    Returns the cycle as a list of vertices (without repeating the start at the end) or None.
    """
    if not adj:
        return None
    
    best_cycle = None
    best_len = 0
    
    # We can restrict starting vertices to those with degree >= 2
    candidates = [v for v in adj if len(adj[v]) >= 2]
    if not candidates:
        return None
    
    # Sort candidates by degree descending to find long cycles faster
    candidates.sort(key=lambda v: -len(adj[v]))
    
    for start in candidates:
        if len(adj[start]) < 2:
            continue
        
        path = [start]
        visited = {start}
        
        # We need to be careful with recursion depth. 
        # Since n <= 30, recursion depth is safe.
        
        def dfs(current, depth):
            nonlocal best_cycle, best_len
            
            # If we found a cycle longer than current best, update
            # We check if we can close the loop to start
            if start in adj[current]:
                cycle_len = depth
                if cycle_len > best_len:
                    best_len = cycle_len
                    best_cycle = list(path)
            
            # Pruning: even if we visit all remaining nodes, can we beat best_len?
            # Max possible cycle length is n. 
            # If depth + (n - depth) <= best_len, we can't beat it? 
            # Actually, max additional nodes is n - len(visited).
            # Max cycle length from here is depth + (n - len(visited)).
            # If that <= best_len, prune.
            if depth + (n - len(visited)) <= best_len:
                return
            
            for nb in adj[current]:
                if nb == start:
                    # Closing the cycle
                    cycle_len = depth
                    if cycle_len > best_len:
                        best_len = cycle_len
                        best_cycle = list(path)
                elif nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, depth + 1)
                    path.pop()
                    visited.discard(nb)
        
        dfs(start, 1)
        
        # Optimization: If we found a Hamiltonian cycle, we can't do better
        if best_len == n:
            return best_cycle
            
    return best_cycle


def _decompose(edges, deadline, rng):
    """Decompose edge set into cycles and single edges, minimizing piece count."""
    if not edges:
        return []
    
    n = max(max(u, v) for u, v in edges) + 1
    rem = set(edges)
    rem_adj = defaultdict(set)
    for u, v in rem:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    
    pieces = []
    
    # Iteratively extract cycles
    # We use a greedy approach: at each step, find the longest cycle in the remaining graph.
    # This minimizes the number of cycles needed to cover the edges.
    
    while rem and time.time() < deadline:
        # Find the longest cycle in the remaining graph
        best_cycle = _find_longest_cycle(rem_adj, n)
        
        if best_cycle and len(best_cycle) >= 3:
            # Remove edges of best_cycle
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                e = (u, v) if u < v else (v, u)
                if e in rem:
                    rem.discard(e)
                    rem_adj[u].discard(v)
                    rem_adj[v].discard(u)
                    if not rem_adj[u]:
                        del rem_adj[u]
                    if not rem_adj[v]:
                        del rem_adj[v]
            pieces.append(best_cycle)
        else:
            # No more cycles found, take a single edge
            if not rem:
                break
            u, v = next(iter(rem))
            rem.discard((u, v))
            rem_adj[u].discard(v)
            rem_adj[v].discard(u)
            if not rem_adj[u]:
                del rem_adj[u]
            if not rem_adj[v]:
                del rem_adj[v]
            pieces.append([u, v])
    
    # Add remaining edges as single-edge pieces
    for u, v in rem:
        pieces.append([u, v])
    
    return pieces


def construct():
    start_time = time.time()
    # The evaluator kills after 150s. We use a safety margin of 10s.
    deadline = start_time + 140
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = _instance(*spec)
        rng = random.Random(42)
        pieces = _decompose(edges, deadline, rng)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
