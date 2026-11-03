import time
import math
import random
import numpy as np
from collections import deque

def construct(k, g):
    deadline = time.time() + 115
    
    if k != 3:
        return _luw_fallback(k, g)
        
    r = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    if moore_bound % 2 != 0:
        moore_bound += 1
        
    # Phase 1: Circulant Search (fast, deterministic)
    max_circ_n = 2000
    for n in range(moore_bound, max_circ_n + 2, 2):
        if time.time() > deadline - 10:
            break
        h = n // 2
        for a in range(1, h):
            if math.gcd(h, a) != 1:
                continue
            if _check_girth_circulant_numpy(n, a, g):
                edges = []
                for u in range(n):
                    for s in (a, n - a, h):
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges
                
    # Phase 2: 2-Switch Optimization
    # Start from Moore bound, try increasing n
    # For each n, generate random 3-regular graphs and optimize via 2-switches
    
    best_n = None
    best_edges = None
    
    for n in range(moore_bound, min(moore_bound + 200, 2000), 2):
        if time.time() > deadline - 15:
            break
            
        # Try multiple random starts with 2-switch optimization
        for attempt in range(15):
            if time.time() > deadline - 15:
                break
                
            adj = _generate_random_3_reg(n)
            if adj is None:
                continue
                
            # Optimize via 2-switches to eliminate short cycles
            adj = _optimize_girth_2switch(adj, g, deadline - 15)
            
            # Check if girth >= g
            if _check_girth_nb(adj, g):
                edges = _get_edges_from_adj(adj)
                return n, edges
                
            # Keep track of best (smallest n with valid girth)
            if best_n is None or n < best_n:
                # We didn't achieve girth >= g, but this is our best attempt
                pass
    
    # Phase 3: Fallback - try larger n with more attempts
    for n in range(min(moore_bound + 200, 2000), 3000, 2):
        if time.time() > deadline - 10:
            break
            
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
                
            adj = _generate_random_3_reg(n)
            if adj is None:
                continue
                
            adj = _optimize_girth_2switch(adj, g, deadline - 10)
            
            if _check_girth_nb(adj, g):
                edges = _get_edges_from_adj(adj)
                return n, edges
    
    return _luw_fallback(k, g)

def _check_girth_circulant_numpy(n, a, g):
    h = n // 2
    S = np.array([a, n - a, h])
    neg_idx = [1, 0, 2]
    
    R = np.zeros((3, n), dtype=bool)
    R[0, a % n] = True
    R[1, (n - a) % n] = True
    R[2, h % n] = True
    
    for d in range(2, g):
        R_new = np.zeros((3, n), dtype=bool)
        for i in range(3):
            for j in range(3):
                if j == neg_idx[i]:
                    continue
                R_new[j] |= np.roll(R[i], S[j])
        R = R_new
        if R[:, 0].any():
            return False
            
    return True

def _generate_random_3_reg(n):
    # Configuration model with rejection for loops/multiedges
    for _ in range(50):
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
            
        random.shuffle(stubs)
        
        adj = [set() for _ in range(n)]
        valid = True
        
        for i in range(0, 3*n, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
            
        if valid:
            # Verify degree
            for i in range(n):
                if len(adj[i]) != 3:
                    valid = False
                    break
            if valid:
                return adj
                
    return None

def _optimize_girth_2switch(adj, g, deadline):
    """
    Iteratively find short cycles and eliminate them via 2-switches.
    A 2-switch: if we have edges (a,b) and (c,d), replace with (a,c) and (b,d)
    or (a,d) and (b,c), provided no loops or multiedges are created.
    This preserves the degree sequence.
    """
    n = len(adj)
    target_depth = g - 1  # We want no cycle shorter than g
    
    # Convert to list of lists for faster access
    adj_list = [list(s) for s in adj]
    
    max_iterations = 5000
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Find a short cycle using BFS/non-backtracking
        short_cycle = _find_short_cycle(adj_list, target_depth)
        if short_cycle is None:
            break  # No short cycle found
            
        # Try to break the cycle via 2-switch
        if not _apply_2switch_to_break_cycle(adj_list, short_cycle):
            # Try random 2-switches to escape local minimum
            if not _try_random_2switches(adj_list, n, 10):
                break  # Can't improve further
        
    # Convert back to sets
    return [set(lst) for lst in adj_list]

def _find_short_cycle(adj_list, max_length):
    """
    Find a cycle of length < max_length using non-backtracking BFS.
    Returns the cycle as a list of vertices, or None.
    """
    n = len(adj_list)
    
    for start in range(n):
        # Non-backtracking BFS: state is (current, previous)
        # dist[(u, v)] = length of shortest non-backtracking walk from start to u with last step from v
        visited = {}  # (u, v) -> (prev_state, dist)
        q = deque()
        
        # Initialize: from start, go to each neighbor
        for v in adj_list[start]:
            q.append((v, start, 1))
            visited[(v, start)] = (None, 1)
        
        while q:
            u, prev, d = q.popleft()
            
            if d >= max_length:
                continue
                
            for w in adj_list[u]:
                if w == prev:
                    continue  # Non-backtracking
                
                if w == start and d + 1 < max_length:
                    # Found a cycle of length d + 1
                    # Reconstruct path
                    cycle = [w]
                    state = (u, prev)
                    while state is not None:
                        cu, cp = state
                        cycle.append(cu)
                        # Find previous state
                        prev_state = None
                        for (su, sp), (ps, _) in visited.items():
                            if su == cp and sp == cu:
                                prev_state = ps
                                break
                        state = prev_state
                    return cycle
                    
                key = (w, u)
                if key not in visited:
                    visited[key] = ((u, prev), d + 1)
                    q.append((w, u, d + 1))
                    
    return None

def _apply_2switch_to_break_cycle(adj_list, cycle):
    """
    Given a short cycle, try to break it via a 2-switch.
    Pick two edges on the cycle and try to replace them with cross-edges.
    """
    n = len(adj_list)
    m = len(cycle)
    if m < 3:
        return False
    
    # Remove duplicates while preserving order
    unique_cycle = []
    seen = set()
    for v in cycle:
        if v not in seen:
            unique_cycle.append(v)
            seen.add(v)
    cycle = unique_cycle
    m = len(cycle)
    
    if m < 3:
        return False
    
    # Get edges on the cycle
    cycle_edges = []
    for i in range(m):
        u = cycle[i]
        v = cycle[(i + 1) % m]
        if v in adj_list[u]:
            cycle_edges.append((u, v))
    
    if len(cycle_edges) < 2:
        return False
    
    # Try all pairs of cycle edges for 2-switch
    for i in range(len(cycle_edges)):
        for j in range(i + 1, len(cycle_edges)):
            a, b = cycle_edges[i]
            c, d = cycle_edges[j]
            
            # Ensure we're working with distinct vertices
            if len({a, b, c, d}) < 4:
                continue
                
            # Try replacement: (a,c) and (b,d)
            if c not in adj_list[a] and d not in adj_list[b]:
                # Check no loops
                if a != c and b != d:
                    # Apply
                    adj_list[a].remove(b)
                    adj_list[b].remove(a)
                    adj_list[c].remove(d)
                    adj_list[d].remove(c)
                    
                    adj_list[a].append(c)
                    adj_list[c].append(a)
                    adj_list[b].append(d)
                    adj_list[d].append(b)
                    return True
            
            # Try replacement: (a,d) and (b,c)
            if d not in adj_list[a] and c not in adj_list[b]:
                if a != d and b != c:
                    adj_list[a].remove(b)
                    adj_list[b].remove(a)
                    adj_list[c].remove(d)
                    adj_list[d].remove(c)
                    
                    adj_list[a].append(d)
                    adj_list[d].append(a)
                    adj_list[b].append(c)
                    adj_list[c].append(b)
                    return True
                    
    return False

def _try_random_2switches(adj_list, n, num_attempts):
    """
    Try random 2-switches to escape local minima.
    Returns True if any switch was applied.
    """
    all_edges = []
    for u in range(n):
        for v in adj_list[u]:
            if u < v:
                all_edges.append((u, v))
    
    for _ in range(num_attempts):
        if len(all_edges) < 2:
            break
        i = random.randint(0, len(all_edges) - 1)
        j = random.randint(0, len(all_edges) - 1)
        if i == j:
            continue
            
        a, b = all_edges[i]
        c, d = all_edges[j]
        
        if len({a, b, c, d}) < 4:
            continue
            
        # Try (a,c) and (b,d)
        if c not in adj_list[a] and d not in adj_list[b] and a != c and b != d:
            adj_list[a].remove(b)
            adj_list[b].remove(a)
            adj_list[c].remove(d)
            adj_list[d].remove(c)
            
            adj_list[a].append(c)
            adj_list[c].append(a)
            adj_list[b].append(d)
            adj_list[d].append(b)
            # Rebuild edge list
            all_edges = []
            for u in range(n):
                for v in adj_list[u]:
                    if u < v:
                        all_edges.append((u, v))
            return True
        # Try (a,d) and (b,c)
        if d not in adj_list[a] and c not in adj_list[b] and a != d and b != c:
            adj_list[a].remove(b)
            adj_list[b].remove(a)
            adj_list[c].remove(d)
            adj_list[d].remove(c)
            
            adj_list[a].append(d)
            adj_list[d].append(a)
            adj_list[b].append(c)
            adj_list[c].append(b)
            all_edges = []
            for u in range(n):
                for v in adj_list[u]:
                    if u < v:
                        all_edges.append((u, v))
            return True
            
    return False

def _check_girth_nb(adj, g):
    """
    Check girth >= g using non-backtracking walks from every vertex.
    """
    n = len(adj)
    adj_list = [list(s) for s in adj]
    
    for start in range(n):
        # State: (current, previous) -> depth
        visited = {}
        q = deque()
        
        for v in adj_list[start]:
            q.append((v, start, 1))
            visited[(v, start)] = 1
            
        while q:
            u, prev, d = q.popleft()
            
            if d >= g - 1:
                continue
                
            for w in adj_list[u]:
                if w == prev:
                    continue
                
                if w == start:
                    # Cycle of length d + 1
                    if d + 1 < g:
                        return False
                    continue
                
                key = (w, u)
                if key not in visited:
                    visited[key] = d + 1
                    q.append((w, u, d + 1))
                    
    return True

def _get_edges_from_adj(adj):
    edges = []
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

def _luw_fallback(k, g):
    # Same as before
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
