import time
import random
import sys
from collections import defaultdict
from functools import lru_cache

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_long_cycle(adj, n, rng, max_restarts=20):
    """
    Find a long simple cycle in the graph represented by adj.
    """
    best_cycle = None
    best_len = 0

    starts = list(range(n))
    rng.shuffle(starts)
    
    for start in starts:
        if not adj[start]:
            continue

        for _ in range(max_restarts):
            path = [start]
            visited = {start}
            current = start

            while True:
                neighbors = list(adj[current])
                unvisited = [v for v in neighbors if v not in visited]
                
                # Check if we can close the cycle to start
                if start in adj[current] and len(path) >= 3:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                    # Close with some probability if path is long enough to allow exploration
                    if len(path) > 6 and rng.random() < 0.3:
                        break
                    if not unvisited:
                        break

                if not unvisited:
                    break

                # Score unvisited neighbors: prefer high degree in remaining graph
                scored = []
                for v in unvisited:
                    unvisited_count = sum(1 for w in adj[v] if w not in visited)
                    score = unvisited_count + rng.random() * 0.5
                    scored.append((score, v))
                
                scored.sort(key=lambda x: x[0], reverse=True)
                
                if rng.random() < 0.8:
                    next_node = scored[0][1]
                else:
                    if len(scored) > 1:
                        next_node = scored[1][1]
                    else:
                        next_node = scored[0][1]
                
                path.append(next_node)
                visited.add(next_node)
                current = next_node

    return best_cycle

def _exact_decompose(edge_set, n):
    """
    Exact decomposition for small edge sets.
    edge_set: frozenset of (u, v) tuples.
    Returns list of pieces (cycles as lists, edges as lists of 2).
    """
    if not edge_set:
        return []
    
    # Memoization key
    key = edge_set
    
    if key in _exact_memo:
        return _exact_memo[key]
        
    # Pick an arbitrary edge
    e = next(iter(key))
    u, v = e
    
    # Option 1: e is a single edge piece
    rest = key - {e}
    best_pieces = [[u, v]] + _exact_decompose(rest, n)
    best_count = len(best_pieces)
    
    # Option 2: e is part of a cycle
    # Find all simple paths from u to v in graph (key - {e})
    # We need to construct the graph for the remaining edges
    adj = defaultdict(set)
    for a, b in key:
        if (a, b) != e and (b, a) != e:
            adj[a].add(b)
            adj[b].add(a)
            
    # DFS to find paths from u to v
    paths = []
    def dfs(node, target, path, visited):
        if node == target:
            if len(path) >= 2: # At least one edge in the path, so cycle length >= 2? No, path u->v means cycle u->...->v->u.
                # Path is [u, ..., v]. Cycle is path + [u].
                # Length of cycle is len(path).
                # We need len(path) >= 2 for a valid cycle (u, v, u is not a simple cycle in simple graph? 
                # In simple graph, u-v is an edge. If we have edge (u,v) and path u->v, 
                # the cycle is u->...->v->u.
                # If path is u->v (direct edge), that's just the edge itself, which we handled in Option 1.
                # So we look for paths of length >= 2 (at least 2 edges in the path part? No, 1 edge in path means u-w-v. Cycle u-w-v-u. Length 3).
                # So we need len(path) >= 2.
                paths.append(path[:])
            return
        
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, target, path, visited)
                path.pop()
                visited.remove(neighbor)
                
    dfs(u, v, [u], {u})
    
    for path in paths:
        # Construct the cycle
        cycle = path + [u]
        # Edges in this cycle
        cycle_edges = set()
        for i in range(len(cycle) - 1):
            a, b = cycle[i], cycle[i+1]
            cycle_edges.add(tuple(sorted((a, b))))
        
        # Check if all edges in cycle are in key
        if cycle_edges.issubset(key):
            rest_edges = key - cycle_edges
            sub_pieces = _exact_decompose(rest_edges, n)
            current_pieces = [cycle] + sub_pieces
            if len(current_pieces) < best_count:
                best_count = len(current_pieces)
                best_pieces = current_pieces
                
    _exact_memo[key] = best_pieces
    return best_pieces

_exact_memo = {}

def _decompose_hybrid(edges, n, rng, deadline):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    remaining_edges = set(edges)
    
    while remaining_edges:
        if time.time() > deadline:
            # Fallback: add remaining as edges
            for u, v in remaining_edges:
                pieces.append([u, v])
            return pieces
        
        m = len(remaining_edges)
        
        if m <= 16:
            # Use exact solver
            # Clear memo? No, keep it, but keys are specific to this subgraph.
            # Actually, clearing memo between restarts is safer to avoid memory bloat, 
            # but keys are frozensets of edges, so collisions are unlikely across different restarts 
            # unless the same subgraph appears.
            exact_pieces = _exact_decompose(frozenset(remaining_edges), n)
            pieces.extend(exact_pieces)
            break
        else:
            # Greedy: find a long cycle
            best_cycle = _find_long_cycle(adj, n, rng, max_restarts=10)
            
            if best_cycle:
                cycle_edges = set()
                for i in range(len(best_cycle)):
                    a = best_cycle[i]
                    b = best_cycle[(i + 1) % len(best_cycle)]
                    e = tuple(sorted((a, b)))
                    cycle_edges.add(e)
                    adj[a].discard(b)
                    adj[b].discard(a)
                    remaining_edges.discard(e)
                
                pieces.append(best_cycle)
            else:
                # No cycle found, remove a single edge
                # This shouldn't happen often if graph is connected enough, but handle it
                e = next(iter(remaining_edges))
                u, v = e
                adj[u].discard(v)
                adj[v].discard(u)
                remaining_edges.discard(e)
                pieces.append([u, v])
                
    return pieces

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    
    base_seed = 42
    attempts = 0
    max_attempts = 100
    
    while time.time() < deadline - 0.5:
        if attempts >= max_attempts:
            break
        
        rng = random.Random(base_seed + attempts)
        # Clear exact memo for each restart to prevent stale data issues?
        # The memo is global. If we call _exact_decompose with a new frozenset, it's fine.
        # But if we want to be safe, we can clear it if it gets too big.
        if len(_exact_memo) > 10000:
            _exact_memo.clear()
            
        pieces = _decompose_hybrid(edges, n, rng, deadline)
        cnt = len(pieces)
        
        if cnt < best_count:
            best_count = cnt
            best = pieces
            
        attempts += 1
    
    if best is None:
        best = [[u, v] for u, v in edges]
        
    return best

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
        
        if time.time() > start_time + total_budget - 3.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
    
    return results
