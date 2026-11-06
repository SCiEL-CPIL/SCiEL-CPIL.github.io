"""Decompose random graphs into cycles and edges, minimizing piece count."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(edges_list, time_budget):
    """Decompose edges into cycles (longer preferred) and single edges."""
    if not edges_list:
        return []

    # Build adjacency
    adj = {}
    edge_set = set()
    for u, v in edges_list:
        edge_set.add((u, v))
        if u not in adj:
            adj[u] = set()
        if v not in adj:
            adj[v] = set()
        adj[u].add(v)
        adj[v].add(u)

    active_edges = set(edge_set)
    pieces = []
    start_time = time.time()
    deadline = start_time + time_budget
    rng = Random(42)

    def _find_longest_cycle(start, max_depth=30, attempts=10):
        """Find a long cycle starting from 'start' using randomized DFS."""
        best_cycle = None
        best_len = 0
        
        for _ in range(attempts):
            # Shuffle neighbors for randomization
            adj_copy = {k: list(v) for k, v in adj.items()}
            for k in adj_copy:
                rng.shuffle(adj_copy[k])
                
            # State for DFS
            path = [start]
            used_edges = set()
            visited_in_path = {start}
            
            def dfs(curr, depth):
                nonlocal best_cycle, best_len
                if depth >= max_depth:
                    return
                
                # Check if we can close the cycle
                if len(path) >= 3:
                    close_edge = (min(curr, start), max(curr, start))
                    if close_edge in active_edges and close_edge not in used_edges:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = list(path)
                        # We found a cycle, but we might find a longer one?
                        # For simplicity and speed, we can return here if we want "first found",
                        # but we want "longest". 
                        # However, backtracking to find longer cycles is expensive.
                        # Let's just return the first valid cycle found in this attempt?
                        # Or continue? 
                        # If we continue, we might find a longer one. 
                        # But depth is limited. 
                        # Let's stop this branch if we found a cycle, as extending it 
                        # is not possible in a simple cycle without repeating vertices.
                        return

                for nb in adj_copy.get(curr, []):
                    edge_key = (min(curr, nb), max(curr, nb))
                    if edge_key not in active_edges or edge_key in used_edges:
                        continue
                    if nb == start:
                        # Only allow closing if len >= 3
                        if len(path) >= 3:
                            continue # Handled above
                        else:
                            continue
                    if nb in visited_in_path:
                        continue
                    
                    # Move to nb
                    path.append(nb)
                    used_edges.add(edge_key)
                    visited_in_path.add(nb)
                    
                    dfs(nb, depth + 1)
                    
                    # Backtrack
                    path.pop()
                    used_edges.remove(edge_key)
                    visited_in_path.remove(nb)
                    
                    # If we found a cycle in this branch, we can stop exploring this neighbor?
                    # Not necessarily, but usually the first one is fine for a heuristic.
                    # To save time, if we found a cycle, we can break?
                    # Let's just let it finish or break if we want speed.
                    # Given N<=30, it's fast.
            
            dfs(start, 0)
            
        return best_cycle

    while active_edges:
        if time.time() > deadline:
            # Fallback: add all remaining as edges
            for e in active_edges:
                pieces.append([e[0], e[1]])
            break

        # Find a vertex with degree >= 2 to start cycle search
        start = None
        for u in adj:
            if len(adj[u]) >= 2:
                # Check if edges are actually active
                active_deg = 0
                for v in adj[u]:
                    if (min(u, v), max(u, v)) in active_edges:
                        active_deg += 1
                if active_deg >= 2:
                    start = u
                    break

        if start is None:
            # No cycles left, just edges
            for e in active_edges:
                pieces.append([e[0], e[1]])
            break

        # Try to find a cycle starting from 'start'
        cycle = _find_longest_cycle(start)
        
        if cycle:
            pieces.append(cycle)
            # Remove cycle edges
            for i in range(len(cycle)):
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                edge_key = (min(a, b), max(a, b))
                active_edges.discard(edge_key)
                # Update adj
                if a in adj and b in adj[a]:
                    adj[a].discard(b)
                if b in adj and a in adj[b]:
                    adj[b].discard(a)
        else:
            # Couldn't find a cycle from this start, try another or just take an edge
            # To avoid infinite loops, if we can't find a cycle, we just take an edge
            # But we should try a few other starts first? 
            # For simplicity and speed, if _find_longest_cycle fails, we take an edge.
            # This might suboptimal but ensures termination.
            e = next(iter(active_edges))
            pieces.append([e[0], e[1]])
            active_edges.discard(e)
            u, v = e
            if u in adj and v in adj[u]:
                adj[u].discard(v)
            if v in adj and u in adj[v]:
                adj[v].discard(u)

    return pieces

def construct():
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    total_budget = 110  # Leave 10s margin for 120s limit

    results = []
    per_graph_budget = total_budget / len(specs)

    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining = total_budget - elapsed
        if remaining < 5:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        pieces = _decompose(edges, min(per_graph_budget, remaining - 2))
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
