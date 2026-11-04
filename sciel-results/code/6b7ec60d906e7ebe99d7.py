"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

_DEADLINE = 120.0
_MARGIN = 5.0

def _validate_graph(n, edges, k, g, max_vertices=None):
    """
    Validate simple k-regular graph with girth >= g.
    Returns True if valid, False otherwise.
    """
    if max_vertices is not None and n > max_vertices:
        return False
        
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # Check girth using non-backtracking BFS from each vertex
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return True

def _luw_graph(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges).
    """
    q = k
    d = g - 5
    if d < 3:
        d = 3
        
    if q == 4:
        table = [[0, 0, 0, 0], 
                 [0, 1, 2, 3], 
                 [0, 2, 3, 1], 
                 [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _count_short_cycles(adj, n, g):
    """
    Count cycles of length < g using non-backtracking walks.
    """
    count = 0
    max_depth = g - 2  # We look for cycles of length < g, so depth < g-1
    
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            if curr == s:
                if depth + 1 < g:
                    count += 1
                continue
                
            if depth >= max_depth:
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return count

def _count_short_cycles_fast(adj, n, g, time_limit=2.0):
    """
    Fast girth check: returns True if girth >= g, False otherwise.
    Uses early termination and time limit.
    """
    t_start = time.time()
    max_depth = g - 2
    
    for s in range(n):
        if time.time() - t_start > time_limit:
            # Return True (assume valid) if we ran out of time - caller should verify
            return True
            
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
                
            if depth >= max_depth:
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return True

def _generate_4_regular(n, seed):
    """
    Generate a random 4-regular simple graph on n vertices using configuration model
    with rejection and retry.
    """
    rng = random.Random(seed)
    for _ in range(50):  # Try up to 50 times
        stubs = []
        for v in range(n):
            for _ in range(4):
                stubs.append(v)
        rng.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
            
        if not valid:
            continue
            
        # Verify degrees
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if all(d == 4 for d in deg):
            return list(edges)
            
    return None

def _improve_girth(n, edges, g, time_limit, rng):
    """
    Use simulated annealing with 2-switch operations to increase girth.
    A 2-switch removes two non-adjacent edges (u1,v1) and (u2,v2) and adds
    (u1,u2) and (v1,v2) or (u1,v2) and (v1,u2).
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
        
    current_cost = _count_short_cycles(adj, n, g)
    if current_cost == 0:
        return list(edges), 0
        
    best_edges = list(edges)
    best_cost = current_cost
    
    t_start = time.time()
    temp = 100.0
    cooling = 0.99
    min_temp = 0.01
    
    # Track iteration count for adaptive cooling
    iterations = 0
    no_improve_count = 0
    
    while time.time() < t_start + time_limit and temp > min_temp:
        iterations += 1
        
        # Pick two random distinct edges
        e1 = rng.choice(edges)
        e2 = rng.choice(edges)
        if e1 == e2:
            continue
            
        u1, v1 = e1
        u2, v2 = e2
        
        # Edges must be non-adjacent (no shared vertices)
        if len({u1, v1, u2, v2}) < 4:
            continue
            
        # Try both possible 2-switches
        for new_e1, new_e2 in [((min(u1, u2), max(u1, u2)), (min(v1, v2), max(v1, v2))),
                                ((min(u1, v2), max(u1, v2)), (min(v1, u2), max(v1, u2)))]:
            # Check new edges don't already exist and aren't loops
            if new_e1 == new_e2:
                continue
            if new_e1 in edge_set or new_e2 in edge_set:
                continue
                
            # Perform the switch
            edge_set.discard(e1)
            edge_set.discard(e2)
            edge_set.add(new_e1)
            edge_set.add(new_e2)
            
            adj[u1].discard(v1)
            adj[v1].discard(u1)
            adj[u2].discard(v2)
            adj[v2].discard(u2)
            
            a1, b1 = new_e1
            a2, b2 = new_e2
            adj[a1].add(b1)
            adj[b1].add(a1)
            adj[a2].add(b2)
            adj[b2].add(a2)
            
            # Update edges list
            edges.remove(e1)
            edges.remove(e2)
            edges.append(new_e1)
            edges.append(new_e2)
            
            # Use fast check first, then exact count if needed
            if _count_short_cycles_fast(adj, n, g, time_limit=0.5):
                # Likely valid, do exact count
                new_cost = _count_short_cycles(adj, n, g)
            else:
                # Definitely invalid, get approximate cost
                new_cost = _count_short_cycles(adj, n, g)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_edges = list(edges)
                no_improve_count = 0
                # Reset temperature on improvement
                temp = max(temp * 1.5, 50.0)
            else:
                no_improve_count += 1
                # Faster cooling when stuck
                if no_improve_count > 100:
                    temp *= 0.95
                    no_improve_count = 0
                
            if new_cost <= current_cost or rng.random() < pow(2.718, -(new_cost - current_cost) / temp):
                current_cost = new_cost
            else:
                # Revert
                edge_set.discard(new_e1)
                edge_set.discard(new_e2)
                edge_set.add(e1)
                edge_set.add(e2)
                
                adj[a1].discard(b1)
                adj[b1].discard(a1)
                adj[a2].discard(b2)
                adj[b2].discard(a2)
                
                adj[u1].add(v1)
                adj[v1].add(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
                
                edges.remove(new_e1)
                edges.remove(new_e2)
                edges.append(e1)
                edges.append(e2)
            break  # Only try one switch type per iteration
            
        temp *= cooling
        
    return best_edges, best_cost

def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # 1. Try LUW Graph
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. If k=4, g=10, try to find a smaller graph
    if k == 4 and g == 10:
        target_n = 320
        rng = random.Random(42)
        
        # Try multiple random starts with simulated annealing
        # Use adaptive number of attempts based on remaining time
        max_attempts = 30
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Check if we have enough time for a meaningful attempt
            time_left = deadline - time.time()
            if time_left < 5:
                break
                
            edges = _generate_4_regular(target_n, seed=1000 + attempt * 13)
            if edges is None:
                continue
                
            # Check if it's already valid (very unlikely)
            if _validate_graph(target_n, edges, k, g):
                best_n = target_n
                best_edges = edges
                break
                
            # Try to improve girth with adaptive time allocation
            # Give more time to later attempts as we learn
            if attempt < 5:
                sa_time = min(time_left - 2, 3.0)
            elif attempt < 15:
                sa_time = min(time_left - 2, 5.0)
            else:
                sa_time = min(time_left - 2, 8.0)
                
            if sa_time < 1:
                break
                
            improved_edges, cost = _improve_girth(target_n, edges, g, sa_time, rng)
            
            if cost == 0:
                if _validate_graph(target_n, improved_edges, k, g):
                    best_n = target_n
                    best_edges = improved_edges
                    break
                    
    return best_n, best_edges

# EVOLVE-BLOCK-END
