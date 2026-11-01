"""
Edge-switching local search for 3-regular graphs of girth >= g.
Starts from a random 3-regular graph near the Moore bound and uses 2-switches
to eliminate short cycles.
"""
import time as _time
import random as _random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """Fast girth check via non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise."""
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _find_shortest_cycle(adj, n, max_girth):
    """Find a cycle of length < max_girth. Returns (cycle_nodes, length) or (None, None)."""
    for start in range(n):
        # Non-backtracking BFS
        dist = {}
        parent = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= max_girth - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    # Found a cycle
                    # Reconstruct cycle
                    cycle = [w]
                    cur_v, cur_prev = v, prev
                    while cur_v != start:
                        cycle.append(cur_v)
                        cur_v, cur_prev = cur_prev, dist.get((cur_v, cur_prev), 0)  # This won't work directly
                        # We need parent pointers
                    return None, None  # Simplified: just return length for now
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return None, None

def _generate_random_3reg(n):
    """Generate a random 3-regular graph on n vertices using configuration model."""
    if n < 4 or n % 2 != 0:
        return None
    
    # Create stubs
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
    
    # Shuffle and pair
    _random.shuffle(stubs)
    edges = set()
    adj = [set() for _ in range(n)]
    
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            # Self-loop, retry
            continue
        if v in adj[u]:
            # Multi-edge, retry
            continue
        adj[u].add(v)
        adj[v].add(u)
        edges.add((min(u,v), max(u,v)))
    
    # Check if all vertices have degree 3
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    return [list(s) for s in adj], sorted(edges)

def _do_2switch(adj, edges_set, u, v, x, y):
    """Perform a 2-switch: remove edges (u,v) and (x,y), add (u,x) and (v,y).
    Returns True if switch was successful (no multi-edges or self-loops created)."""
    if v in adj[x] or y in adj[u] or u == x or v == y:
        return False
    if (u, x) in edges_set or (x, u) in edges_set or (v, y) in edges_set or (y, v) in edges_set:
        return False
    
    # Remove old edges
    adj[u].remove(v)
    adj[v].remove(u)
    adj[x].remove(y)
    adj[y].remove(x)
    edges_set.discard((min(u,v), max(u,v)))
    edges_set.discard((min(x,y), max(x,y)))
    
    # Add new edges
    adj[u].add(x)
    adj[x].add(u)
    adj[v].add(y)
    adj[y].add(v)
    edges_set.add((min(u,x), max(u,x)))
    edges_set.add((min(v,y), max(v,y)))
    
    return True

def _undo_2switch(adj, edges_set, u, v, x, y):
    """Undo a 2-switch."""
    adj[u].remove(x)
    adj[x].remove(u)
    adj[v].remove(y)
    adj[y].remove(v)
    edges_set.discard((min(u,x), max(u,x)))
    edges_set.discard((min(v,y), max(v,y)))
    
    adj[u].add(v)
    adj[v].add(u)
    adj[x].add(y)
    adj[y].add(x)
    edges_set.add((min(u,v), max(u,v)))
    edges_set.add((min(x,y), max(x,y)))

def _count_short_cycles(adj, n, g):
    """Count the number of cycles of length < g using BFS."""
    count = 0
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    count += 1
                    continue
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return count

def _luw_construct(k, g, d_override=None):
    """LUW construction with optional d override."""
    q = k
    if d_override is not None:
        d = d_override
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
    else:
        add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    if k != 3:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw
    
    # Moore bound for k=3, even g: 2*(2^(g/2)-1)
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    # Start from Moore bound, try to find a valid graph
    # n must be even for 3-regular graphs
    n_start = moore if moore % 2 == 0 else moore + 1
    
    best_result = None
    best_n = float('inf')
    
    # Try edge-switching local search
    for n_try in range(n_start, min(n_start + 200, 1000)):
        if _time.time() > deadline - 10:
            break
        
        if n_try >= best_n:
            continue
            
        if n_try % 2 != 0:
            continue
            
        # Try multiple random initial graphs
        max_initials = 5 if n_try < 600 else 3
        for init_idx in range(max_initials):
            if _time.time() > deadline - 5:
                break
                
            _random.seed(hash((n_try, init_idx)))
            result = _generate_random_3reg(n_try)
            if result is None:
                continue
                
            adj, edges = result
            edges_set = set(edges)
            
            # Check initial girth
            if _verify_girth_fast(adj, n_try, g):
                best_n = n_try
                best_result = (n_try, edges)
                return n_try, edges
            
            # Local search with 2-switches
            # We want to minimize the number of short cycles
            # Use simulated annealing or greedy descent
            
            current_short = _count_short_cycles(adj, n_try, g)
            best_short = current_short
            best_adj_state = [list(s) for s in adj]
            best_edges_state = set(edges_set)
            
            # Greedy descent: try random 2-switches, keep if it reduces short cycles
            max_iter = 500 if n_try < 600 else 200
            for it in range(max_iter):
                if _time.time() > deadline - 3:
                    break
                    
                if best_short == 0:
                    break
                    
                # Pick random edges (u,v) and (x,y)
                edge_list = list(edges_set)
                if len(edge_list) < 2:
                    break
                    
                e1, e2 = _random.sample(edge_list, 2)
                u, v = e1
                x, y = e2
                
                # Try the switch
                if _do_2switch(adj, edges_set, u, v, x, y):
                    new_short = _count_short_cycles(adj, n_try, g)
                    if new_short < best_short:
                        best_short = new_short
                        best_adj_state = [list(s) for s in adj]
                        best_edges_state = set(edges_set)
                    else:
                        # Undo
                        _undo_2switch(adj, edges_set, u, v, x, y)
                
                if best_short == 0:
                    break
            
            if best_short == 0:
                best_n = n_try
                best_edges_list = sorted(best_edges_state)
                best_result = (n_try, best_edges_list)
                return n_try, best_edges_list
    
    # Fallback: LUW
    if _time.time() < deadline - 5:
        for d_try in range(6, 12):
            if _time.time() > deadline - 3:
                break
            try:
                n_try, edges_try = _luw_construct(3, g, d_override=d_try)
                if n_try < best_n:
                    adj = [set() for _ in range(n_try)]
                    for u, v in edges_try:
                        adj[u].add(v)
                        adj[v].add(u)
                    adj_list = [list(s) for s in adj]
                    if _verify_girth_fast(adj_list, n_try, g):
                        best_n = n_try
                        best_result = (n_try, edges_try)
                        return n_try, edges_try
            except Exception:
                pass
    
    # Final fallback
    if best_result is not None:
        return best_result
    
    n_luw, edges_luw = _luw_construct(3, g)
    return n_luw, edges_luw
