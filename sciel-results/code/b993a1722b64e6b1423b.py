import time
from collections import deque

_DEADLINE = time.time() + 130.0

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adj(edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj

def _find_longest_cycle(adj, max_search=50):
    """Find a reasonably long cycle in the graph given by adj (dict of sets)."""
    best_cycle = None
    best_len = 0
    nodes = list(adj.keys())
    if not nodes:
        return None
    
    # Try from several starting nodes
    starts = nodes[:min(len(nodes), max_search)]
    
    for start in starts:
        if time.time() > _DEADLINE:
            break
        # BFS to find shortest path back to start via different route
        # Actually, let's do DFS to find any cycle through start
        # Use a simpler approach: for each edge (u,v), if there's a path from v to u not using that edge, we have a cycle
        
        # Simpler: do DFS from start, track parent, when we find a back edge, extract cycle
        visited = {start: None}  # node -> parent
        stack = [(start, iter(sorted(adj.get(start, set()))))]
        found = False
        while stack and not found:
            node, neighbors = stack[-1]
            advanced = False
            for nb in neighbors:
                if nb == visited.get(node, -1) and node != start:
                    continue  # skip parent edge
                if nb not in visited:
                    visited[nb] = node
                    stack.append((nb, iter(sorted(adj.get(nb, set())))))
                    advanced = True
                    break
                elif nb != visited.get(node, -1):
                    # Found a back edge: node -> nb, and nb is already visited (ancestor)
                    # Extract cycle from nb to node via parent pointers, plus edge node-nb
                    if nb == start:
                        # Cycle from start to node back to start
                        cycle = []
                        cur = node
                        while cur != start:
                            cycle.append(cur)
                            cur = visited[cur]
                        cycle.append(start)
                        if len(cycle) > best_len:
                            best_len = len(cycle)
                            best_cycle = cycle
                        found = True
                        break
                    else:
                        # nb is an ancestor but not start - still a valid cycle
                        # Extract from nb to node
                        cycle = []
                        cur = node
                        while cur != nb:
                            cycle.append(cur)
                            cur = visited[cur]
                        cycle.append(nb)
                        if len(cycle) > best_len:
                            best_len = len(cycle)
                            best_cycle = cycle
                        found = True
                        break
            if not advanced:
                stack.pop()
    
    return best_cycle

def _decompose_graph(edges):
    """Decompose edges into cycles and single edges. Returns list of pieces."""
    if not edges:
        return []
    
    # Build adjacency and edge set
    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = _build_adj(edge_set)
    
    pieces = []
    remaining = set(edge_set)
    
    # Iteratively find cycles
    max_iters = 200
    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
        if not remaining:
            break
        
        # Rebuild adj from remaining
        adj = _build_adj(remaining)
        
        # Find a cycle
        cycle = _find_longest_cycle(adj)
        
        if cycle is None or len(cycle) < 3:
            break
        
        # Verify all edges in cycle exist in remaining
        valid = True
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            e = tuple(sorted((u, v)))
            if e not in remaining:
                valid = False
                break
        
        if not valid:
            # Try a simpler cycle-finding: just find any cycle
            cycle = _find_any_cycle(adj)
            if cycle is None or len(cycle) < 3:
                break
            valid = True
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i+1) % len(cycle)]
                e = tuple(sorted((u, v)))
                if e not in remaining:
                    valid = False
                    break
            if not valid:
                break
        
        # Add cycle as a piece and remove its edges
        pieces.append(cycle)
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            e = tuple(sorted((u, v)))
            remaining.discard(e)
    
    # Remaining edges become individual pieces
    for e in remaining:
        pieces.append(list(e))
    
    return pieces

def _find_any_cycle(adj):
    """Find any cycle in the graph. Returns list of vertices or None."""
    if not adj:
        return None
    
    nodes = list(adj.keys())
    for start in nodes:
        if time.time() > _DEADLINE:
            return None
        visited = {start}
        parent = {start: None}
        stack = [start]
        while stack:
            node = stack[-1]
            advanced = False
            for nb in adj.get(node, set()):
                if nb == parent.get(node, -1):
                    continue
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = node
                    stack.append(nb)
                    advanced = True
                    break
                elif nb != parent.get(node, -1):
                    # Found a cycle
                    cycle = []
                    cur = node
                    while cur != nb:
                        cycle.append(cur)
                        cur = parent[cur]
                    cycle.append(nb)
                    if len(cycle) >= 3:
                        return cycle
            if not advanced:
                stack.pop()
    
    return None

def construct():
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    for n, p, seed in specs:
        if time.time() > _DEADLINE:
            # Fallback: just return edges as individual pieces
            edges = _instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        edges = _instance(n, p, seed)
        pieces = _decompose_graph(edges)
        results.append(pieces)
    return results
