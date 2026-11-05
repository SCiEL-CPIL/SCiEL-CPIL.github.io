import time
import random
import sys

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycle_heuristic(adj, start, n, rnd, max_depth=30):
    """
    Find a simple cycle starting from 'start'.
    Heuristic: DFS that prefers nodes with higher degree to try to find longer cycles.
    Returns the longest cycle found in the search space explored.
    """
    best_cycle = []
    visited = [False] * n
    path = []
    parent = [-1] * n
    
    # We use a stack for iterative DFS to avoid recursion limits and allow better control
    # But recursive is easier to manage for backtracking. Let's use recursion with a depth limit.
    
    def dfs(u, depth):
        nonlocal best_cycle
        if depth > max_depth:
            return
        
        visited[u] = True
        path.append(u)
        
        # Check if we can close a cycle
        # We need an edge from u to some node in path[1:] (not path[0] if we want simple cycle > 2, 
        # actually path[0] is start, so edge to start closes the cycle)
        # Wait, if we are at u, and v is in path, and (u,v) is an edge, we have a cycle.
        # The cycle is path[index(v):] + [v]
        
        # To find the LONGEST cycle, we should check all neighbors v in path.
        # If v is in path, cycle length = len(path) - index(v)
        
        # However, checking all neighbors in path is O(N).
        
        # Strategy: 
        # 1. If we can close a cycle, record it.
        # 2. Continue DFS to try to make it longer?
        #    If we close a cycle now, we can't extend it further in this branch.
        #    So we should prefer NOT closing the cycle if we can extend the path.
        #    Only close if we are stuck or depth limit reached?
        #    No, a shorter cycle might be better if we can't find a longer one.
        #    But we want the LONGEST.
        
        # Refined Strategy:
        # We want to find a long cycle. 
        # We should try to extend the path first.
        # If we can't extend (all neighbors are visited or in path), then we check for cycles.
        # Actually, if a neighbor is in the path, we *can* close a cycle. 
        # But if we close it, we stop extending.
        # So, we should only consider closing a cycle if we have no other unvisited neighbors?
        # Or if we have visited neighbors that are not in the path (dead ends)?
        
        # Let's try this:
        # 1. Look for unvisited neighbors.
        # 2. If any exist, recurse into them (prefer high degree).
        # 3. If no unvisited neighbors, look for neighbors in the path.
        #    If found, record the cycle.
        
        unvisited_neighbors = [v for v in adj[u] if not visited[v]]
        
        if unvisited_neighbors:
            # Sort by degree descending to try to find long cycles
            unvisited_neighbors.sort(key=lambda x: len(adj[x]), reverse=True)
            for v in unvisited_neighbors:
                dfs(v, depth + 1)
        else:
            # No unvisited neighbors. Check for cycles closing back to path.
            for i, v in enumerate(path):
                if v == u: continue # self loop not possible
                if (u, v) in adj[u] or (v, u) in adj[u]: # adj is symmetric
                    # Found a cycle: path[i:] + [v]
                    cyc_len = len(path) - i
                    if cyc_len > len(best_cycle):
                        best_cycle = path[i:] + [v]
        
        path.pop()
        visited[u] = False

    dfs(start, 0)
    return best_cycle

def _decompose(edges, n, deadline, rnd):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining = len(edges)
    
    while remaining > 0:
        if time.time() > deadline:
            break
            
        # Pick a start vertex
        # Heuristic: Pick a vertex with high degree
        start_candidates = [v for v in range(n) if adj[v]]
        if not start_candidates:
            break
            
        # Randomize to allow multiple attempts in _solve_one
        start = rnd.choice(start_candidates)
        
        # Try to find a long cycle
        cyc = _find_cycle_heuristic(adj, start, n, rnd, max_depth=n)
        
        if cyc and len(cyc) >= 3:
            # Remove cycle edges
            m = len(cyc) - 1
            for i in range(m):
                a = cyc[i]
                b = cyc[i+1]
                if a in adj[b]:
                    adj[a].discard(b)
                    adj[b].discard(a)
                    remaining -= 1
            pieces.append(cyc[:-1])
        else:
            # No cycle found from this start. Try a different start?
            # Or just remove an edge.
            # To be more aggressive, try a few random starts before giving up on cycles for this step.
            found_cycle = False
            for _ in range(5):
                if time.time() > deadline: break
                if not start_candidates: break
                s = rnd.choice(start_candidates)
                cyc = _find_cycle_heuristic(adj, s, n, rnd, max_depth=n)
                if cyc and len(cyc) >= 3:
                    m = len(cyc) - 1
                    for i in range(m):
                        a = cyc[i]
                        b = cyc[i+1]
                        if a in adj[b]:
                            adj[a].discard(b)
                            adj[b].discard(a)
                            remaining -= 1
                    pieces.append(cyc[:-1])
                    found_cycle = True
                    break
            
            if not found_cycle:
                # Remove a random edge
                # Pick a vertex with degree > 0
                v = rnd.choice(start_candidates)
                nbrs = list(adj[v])
                if nbrs:
                    u = rnd.choice(nbrs)
                    adj[v].discard(u)
                    adj[u].discard(v)
                    remaining -= 1
                    pieces.append([v, u])

    # Leftover edges
    for u in range(n):
        for v in list(adj[u]):
            if u < v:
                adj[u].discard(v)
                adj[v].discard(u)
                pieces.append([u, v])

    return pieces

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    base_seed = 1000
    attempt = 0
    while time.time() < deadline:
        rnd = random.Random(base_seed + attempt)
        pieces = _decompose(edges, n, deadline, rnd)
        cnt = len(pieces)
        if cnt < best_count:
            best = pieces
            best_count = cnt
        attempt += 1
        if best_count <= n // 2:
            break
        if attempt > 200: # Limit attempts to ensure we don't spend too long on one
            break
    return best

def construct():
    start_time = time.time()
    total_budget = 140.0
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
