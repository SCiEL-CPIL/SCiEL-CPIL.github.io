import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _random_k_regular(n, k, rng):
    """Generate a random k-regular graph using configuration model."""
    if n * k % 2 != 0:
        raise ValueError("n*k must be even")
    
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    rng.shuffle(half_edges)
    
    adj = [set() for _ in range(n)]
    for i in range(0, n * k, 2):
        u, v = half_edges[i], half_edges[i + 1]
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    
    # Fix multi-edges/self-loops
    for _ in range(500):
        bad = False
        for u in range(n):
            if len(adj[u]) != k:
                bad = True
                break
        if not bad:
            break
        
        for u in range(n):
            if len(adj[u]) < k:
                candidates = [v for v in range(n) if v != u and v not in adj[u] and len(adj[v]) < k]
                if candidates:
                    v = rng.choice(candidates)
                    adj[u].add(v)
                    adj[v].add(u)
            elif len(adj[u]) > k:
                v = rng.choice(list(adj[u]))
                adj[u].discard(v)
                adj[v].discard(u)
    
    adj_list = [sorted(adj[i]) for i in range(n)]
    for i in range(n):
        if len(adj_list[i]) != k:
            raise ValueError(f"Vertex {i} has degree {len(adj_list[i])} instead of {k}")
    return adj_list

def _has_short_cycle(adj, n, g):
    """Check if graph has no cycle shorter than g. Returns True if valid."""
    # Non-backtracking BFS from every vertex
    # We only need to check up to depth g/2 - 1 effectively?
    # Actually, any cycle of length < g will be found.
    # Optimization: If we find a cycle, return False immediately.
    
    for start in range(n):
        # visited[state] = depth
        # state = (node, prev)
        visited = {}
        queue = deque()
        
        for nb in adj[start]:
            state = (nb, start)
            visited[state] = 1
            queue.append(state)
            
        while queue:
            node, prev = queue.popleft()
            depth = visited[(node, prev)]
            
            if depth >= g - 1:
                continue
                
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start:
                    # Cycle of length depth + 1
                    if depth + 1 < g:
                        return False
                    continue
                state = (nb, node)
                if state in visited:
                    # We've seen this state before.
                    # The cycle length is depth + 1 (path to state) + (path from state to current? No)
                    # Wait, non-backtracking BFS.
                    # If we reach (nb, node) and it was visited at depth d', 
                    # it means there is a path of length d' from start to nb via node.
                    # And we have a path of length depth from start to node via prev.
                    # This implies a cycle?
                    # Actually, standard girth check:
                    # If we encounter a neighbor that is already visited, and it's not the parent,
                    # we have found a cycle.
                    # The length is depth + 1 + (distance from start to nb in the other path?)
                    # Simpler: Just check if depth + 1 < g.
                    # If we are at depth < g-1 and we see a visited node, it implies a cycle of length <= depth + 1 + 1?
                    # Let's stick to the simple check: if we close a cycle to start, or if we hit a visited node.
                    # If we hit a visited node (nb, node) at depth d_visited, and current depth is d_current.
                    # The cycle length is d_current + 1 + d_visited? No.
                    # The cycle is formed by the path from start to node (length d_current) 
                    # and the path from start to nb (length d_visited) ... this is getting complicated.
                    
                    # Reliable method:
                    # If (nb, node) is in visited, it means we have two non-backtracking paths from start to nb.
                    # One ends at node (current), one ended at some prev_node (visited).
                    # This forms a cycle.
                    # The length is bounded by d_current + d_visited + 1?
                    # To be safe, if d_current + 1 < g, we assume it might be a short cycle.
                    # Actually, if d_current < g - 1, and we hit a visited state, 
                    # the cycle length is at most d_current + 1 + d_visited.
                    # Since d_visited <= d_current (BFS), length <= 2*d_current + 1.
                    # If 2*d_current + 1 < g, it's definitely a short cycle.
                    # If d_current + 1 < g, it *might* be.
                    
                    # Let's just return False if depth + 1 < g and we hit a visited state.
                    # This is conservative (might report short cycle when it's longer? No, BFS finds shortest).
                    # If the shortest non-backtracking path to (nb, node) is d_visited,
                    # and we have a path of length depth to node, then the cycle is length <= depth + 1 + d_visited?
                    # Actually, the cycle is the union of the two paths.
                    # Length is depth + 1 + d_visited.
                    # If depth + 1 + d_visited < g, it's a short cycle.
                    # Since d_visited >= 1, if depth + 2 < g, it's likely short.
                    # Let's just check if depth + 1 < g. If we hit a visited node, 
                    # it means the girth is <= depth + 1 + d_visited.
                    # If depth + 1 < g, it's possible.
                    
                    # To be 100% correct and fast:
                    # If we find a cycle, return False.
                    # The cycle length is at most depth + 1 + visited[state].
                    # If depth + 1 + visited[state] < g, return False.
                    if depth + 1 + visited[state] < g:
                        return False
                else:
                    if depth + 1 < g:
                        visited[state] = depth + 1
                        queue.append(state)
                        
    return True

def _find_short_cycle_path(adj, n, g):
    """Find a specific short cycle. Returns list of vertices or None."""
    for start in range(n):
        visited = {}
        queue = deque()
        for v in adj[start]:
            queue.append((v, start))
            visited[(v, start)] = (1, [start, v])
        
        while queue:
            u, prev = queue.popleft()
            depth, path = visited[(u, prev)]
            if depth >= g - 1:
                continue
            for w in adj[u]:
                if w == prev:
                    continue
                if w == start and depth + 1 >= 3:
                    if depth + 1 < g:
                        return path + [start]
                elif (w, u) not in visited:
                    visited[(w, u)] = (depth + 1, path + [w])
                    queue.append((w, u))
    return None

def _improve_girth_sa(adj, n, k, g, rng, deadline):
    """Simulated Annealing to improve girth."""
    current_adj = [list(s) for s in adj]
    
    # Initial check
    if _has_short_cycle(current_adj, n, g):
        return current_adj
        
    # Get initial energy (count of short cycles approx)
    # We will just use _has_short_cycle to guide.
    # If it has a short cycle, we try to remove it.
    
    consecutive_failures = 0
    temp = 1.0
    cooling = 0.95
    
    while time.time() < deadline:
        cycle = _find_short_cycle_path(current_adj, n, g)
        if not cycle:
            return current_adj
            
        L = len(cycle) - 1
        # Pick an edge from the cycle
        edge_idx = rng.randint(0, L - 1)
        u = cycle[edge_idx]
        v = cycle[(edge_idx + 1) % L]
        
        # Try to swap (u,v) with another edge
        accepted = False
        for _ in range(50):
            # Find a valid edge (x,y) not incident to u or v
            # and such that (u,x), (v,y) or (u,y), (v,x) are not existing edges
            valid_swaps = []
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in adj[x]:
                    if x < y and y != u and y != v:
                        # Check if swap creates multi-edges
                        if (u not in adj[x] and v not in adj[y]) or (u not in adj[y] and v not in adj[x]):
                            valid_swaps.append((x, y))
                            
            if not valid_swaps:
                break
                
            x, y = rng.choice(valid_swaps)
            
            # Try swap 1: (u,x) and (v,y)
            if u not in adj[x] and v not in adj[y]:
                # Apply
                new_adj = [list(s) for s in current_adj]
                new_adj[u].remove(v)
                new_adj[v].remove(u)
                new_adj[x].remove(y)
                new_adj[y].remove(x)
                
                new_adj[u].append(x)
                new_adj[x].append(u)
                new_adj[v].append(y)
                new_adj[y].append(v)
                
                # Check if valid (no multi edges)
                valid = True
                for i in range(n):
                    if len(new_adj[i]) != len(set(new_adj[i])):
                        valid = False
                        break
                if valid and _has_short_cycle(new_adj, n, g):
                    current_adj = new_adj
                    accepted = True
                    consecutive_failures = 0
                    break
                else:
                    # Check if it reduced the "badness"
                    # We don't have a count, so we just accept if it breaks the specific cycle?
                    # _has_short_cycle is binary.
                    # If it's still invalid, we might still accept it if it looks "better"?
                    # Hard to say. Let's just reject if not valid.
                    pass
            
            # Try swap 2: (u,y) and (v,x)
            if u not in adj[y] and v not in adj[x]:
                new_adj = [list(s) for s in current_adj]
                new_adj[u].remove(v)
                new_adj[v].remove(u)
                new_adj[x].remove(y)
                new_adj[y].remove(x)
                
                new_adj[u].append(y)
                new_adj[y].append(u)
                new_adj[v].append(x)
                new_adj[x].append(v)
                
                valid = True
                for i in range(n):
                    if len(new_adj[i]) != len(set(new_adj[i])):
                        valid = False
                        break
                if valid and _has_short_cycle(new_adj, n, g):
                    current_adj = new_adj
                    accepted = True
                    consecutive_failures = 0
                    break
            
        if not accepted:
            consecutive_failures += 1
            if consecutive_failures > 100:
                # Randomize current graph slightly or break
                # Just break to try a new initialization
                break
        else:
            temp *= cooling

    return current_adj

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 115
    
    # Moore bound
    if g % 2 == 0:
        moore = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        moore = 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))
        
    if moore % 2 != 0:
        moore += 1 # n must be even for regular graph if k is even? No, n*k must be even.
        
    # Ensure n*k is even
    start_n = moore
    if (start_n * k) % 2 != 0:
        start_n += 1

    rng = random.Random(42)
    
    best_n = None
    best_edges = None
    
    # Try sizes from Moore bound up to 320
    max_n_to_try = 320
    if start_n > max_n_to_try:
        start_n = max_n_to_try
        
    # First, get a baseline from LUW
    luw_n, luw_edges = _luw_construct(k, g)
    best_n = luw_n
    best_edges = luw_edges
    
    # Try to find smaller graphs
    for n_try in range(start_n, min(luw_n, max_n_to_try + 1)):
        if time.time() > deadline - 10:
            break
            
        if (n_try * k) % 2 != 0:
            continue
            
        # Try a few random initializations for this size
        for init in range(20):
            if time.time() > deadline - 5:
                break
                
            try:
                adj = _random_k_regular(n_try, k, rng)
            except Exception:
                continue
                
            if _has_short_cycle(adj, n_try, g):
                best_n = n_try
                best_edges = [(u, v) for u in range(n_try) for v in adj[u] if u < v]
                return best_n, best_edges
                
            # Try to improve
            improved_adj = _improve_girth_sa(adj, n_try, k, g, rng, deadline - 2)
            
            if _has_short_cycle(improved_adj, n_try, g):
                best_n = n_try
                best_edges = [(u, v) for u in range(n_try) for v in improved_adj[u] if u < v]
                return best_n, best_edges

    return best_n, best_edges

# EVOLVE-BLOCK-END
