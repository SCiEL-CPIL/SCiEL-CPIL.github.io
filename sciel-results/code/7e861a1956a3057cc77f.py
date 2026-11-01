"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque
import math
import sys

def _verify_girth_fast(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
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

def _verify_girth_array_fast(adj_list, n, g, max_check=0):
    """Faster girth verification with early exit. max_check=0 means check all."""
    if max_check == 0:
        max_check = n
    for start in range(max_check):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj_list[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _luw_construct(q, g):
    """LUW construction for (q+1)-regular graphs. Returns (n, edges)."""
    if q == 3:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
    else:
        p = q
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = (lambda x,y:(x+y)%p), (lambda x,y:(x-y)%p), (lambda x,y:x*y%p)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(p):
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

def _find_shortest_cycle(adj_list, n):
    """Find the shortest cycle in the graph. Returns (length, cycle_vertices) or (inf, [])."""
    min_len = float('inf')
    min_cycle = []
    for start in range(n):
        dist = {start: 0}
        parent = {start: -1}
        dq = deque([start])
        while dq:
            v = dq.popleft()
            if dist[v] >= min_len - 1:
                continue
            for w in adj_list[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    parent[w] = v
                    dq.append(w)
                elif w != parent[v]:
                    # Found a cycle
                    # Reconstruct cycle
                    cycle = []
                    curr = v
                    while curr != w:
                        cycle.append(curr)
                        curr = parent[curr]
                    cycle.append(w)
                    # The cycle is w -> ... -> v -> w
                    # Actually, the path is start -> ... -> w and start -> ... -> v
                    # We need the path from w to v via parent pointers?
                    # No, parent pointers go towards start.
                    # So path from w to start and v to start.
                    # LCA is not start necessarily.
                    # This BFS finds shortest path from start to all nodes.
                    # A cycle through start has length dist[v] + 1 (if v connected to start? No).
                    # The cycle is formed by edge (v, w) and paths to LCA.
                    # This simple BFS doesn't easily give the cycle length without LCA.
                    # Let's use a different method: BFS from each node to find shortest cycle passing through it?
                    # Or just use the property that shortest cycle in graph is min over all edges (u,v) of dist(u,v) + 1.
                    # But we don't have all pairs dist.
                    pass
    # This function is too slow/complex to implement correctly in a hurry.
    # Let's use a simpler heuristic: Just verify girth. If it fails, we know girth < g.
    # We don't strictly need the cycle for the greedy algorithm if we just do random swaps and check.
    return min_len, min_cycle

def _local_search_3reg(n, g, seed, time_limit):
    """
    Attempt to find a 3-regular graph on n vertices with girth >= g.
    Uses random initialization and 2-switches to break short cycles.
    """
    rng = _random.Random(seed)
    start_time = _time.time()
    
    # Generate random 3-regular graph
    def generate_random_3reg(n, rng):
        stubs = []
        for v in range(n):
            for _ in range(3):
                stubs.append(v)
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, n * 3, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if not valid:
            return None
        # Check connectivity
        visited = set([0])
        stack = [0]
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if w not in visited:
                    visited.add(w)
                    stack.append(w)
        if len(visited) != n:
            return None
        return [sorted(adj[i]) for i in range(n)]

    adj_list = generate_random_3reg(n, rng)
    if adj_list is None:
        return None

    # Local search loop
    while _time.time() - start_time < time_limit:
        if _verify_girth_array_fast(adj_list, n, g):
            edges = []
            for u in range(n):
                for v in adj_list[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
        
        # Find a short cycle to break
        # We don't have an efficient find_shortest_cycle, so we just pick random edges
        # and try to swap them.
        # To be smarter, we can pick edges that are likely in short cycles.
        # But random swaps often work well for girth improvement.
        
        # Pick two random edges (u1, v1) and (u2, v2) such that they are distinct and non-adjacent?
        # 2-switch: remove (u1, v1) and (u2, v2), add (u1, u2) and (v1, v2) (or v1, u2 and v2, u1).
        
        # Collect all edges
        edges_set = set()
        for u in range(n):
            for v in adj_list[u]:
                if u < v:
                    edges_set.add((u, v))
        
        if not edges_set:
            break
            
        # Pick two random edges
        edge_list = list(edges_set)
        e1, e2 = rng.sample(edge_list, 2)
        
        u1, v1 = e1
        u2, v2 = e2
        
        # Check if swap is valid (no loops, no multi-edges)
        # Option 1: (u1, u2) and (v1, v2)
        valid_swap = False
        new_edges = set()
        
        # Try swap type 1
        if u1 != u2 and v1 != v2 and (u1, v2) not in adj_list[u1] and (v1, u2) not in adj_list[v1]:
             # Actually we need to check if (u1, u2) and (v1, v2) exist
             if (u1, u2) not in adj_list[u1] and (v1, v2) not in adj_list[v1]:
                 # Perform swap
                 adj_list[u1].remove(v1)
                 adj_list[v1].remove(u1)
                 adj_list[u2].remove(v2)
                 adj_list[v2].remove(u2)
                 
                 adj_list[u1].append(u2)
                 adj_list[u2].append(u1)
                 adj_list[v1].append(v2)
                 adj_list[v2].append(v1)
                 
                 # Sort for consistency? Not strictly needed for set-based adj, but good for determinism
                 # adj_list[u1].sort()
                 # etc.
                 valid_swap = True
                 
        if not valid_swap:
             # Try swap type 2: (u1, v2) and (v1, u2)
             if u1 != v2 and v1 != u2 and (u1, v2) not in adj_list[u1] and (v1, u2) not in adj_list[v1]:
                 adj_list[u1].remove(v1)
                 adj_list[v1].remove(u1)
                 adj_list[u2].remove(v2)
                 adj_list[v2].remove(u2)
                 
                 adj_list[u1].append(v2)
                 adj_list[v2].append(u1)
                 adj_list[v1].append(u2)
                 adj_list[u2].append(v1)
                 valid_swap = True
                 
        if not valid_swap:
             # Revert if we partially modified? We only modify if valid_swap is True.
             # So if not valid_swap, we did nothing.
             pass
             
    return None

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 115
    
    if k != 3:
        q = k
        n, edges = _luw_construct(q, g)
        return n, edges
    
    # Moore bound for k=3, girth g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    best_result = None
    best_n = float('inf')
    
    # Strategy 1: Local Search for small n
    # Try n from moore to moore + 1000
    # We have ~100s budget.
    # Let's allocate 80s for local search.
    search_time_budget = 80
    search_deadline = start_time + search_time_budget
    
    # We know the smallest known is 936.
    # Let's try a range of n values.
    # If we find a solution, we return it.
    # We should try n values that are multiples of 2 (3-reg requires even n? No, 3*n must be even, so n must be even).
    
    # Start from moore, step by 2.
    # But moore might be 510.
    # Let's try a few specific ranges.
    
    # To avoid wasting time on impossible small n, we can jump to known good ranges?
    # No, let's just try sequentially.
    
    n_try = moore
    if n_try % 2 != 0:
        n_try += 1
        
    # Limit the number of attempts per n to avoid getting stuck
    max_attempts_per_n = 5
    time_per_attempt = 5.0
    
    while n_try < moore + 2000 and _time.time() < search_deadline:
        if n_try % 2 != 0:
            n_try += 1
            continue
            
        # Check if we have time for this n
        if _time.time() + time_per_attempt * max_attempts_per_n > search_deadline:
            break
            
        for attempt in range(max_attempts_per_n):
            if _time.time() > search_deadline:
                break
            seed = n_try * 100 + attempt
            result = _local_search_3reg(n_try, g, seed, time_limit=time_per_attempt)
            if result is not None:
                n_r, edges_r = result
                if n_r < best_n:
                    best_n = n_r
                    best_result = (n_r, edges_r)
                # Found a valid graph, return immediately (it's the smallest n we tried)
                return best_result
        n_try += 2
        
    # Strategy 2: LUW construction as fallback
    if _time.time() < deadline - 10:
        n_luw, edges_luw = _luw_construct(3, g)
        if n_luw < best_n:
            best_n = n_luw
            best_result = (n_luw, edges_luw)
            
    if best_result is None:
        n_luw, edges_luw = _luw_construct(3, g)
        best_result = (n_luw, edges_luw)
        
    return best_result
# EVOLVE-BLOCK-END
