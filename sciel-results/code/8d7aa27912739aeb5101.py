import time
import sys
import random
from collections import deque

_DEADLINE = 115.0

def construct(k, g):
    t_start = time.time()
    
    # Fallback for general k, g
    if k != 4 or g != 10:
        return _luw(k, g)
    
    # Specific search for (4, 10) cage
    # Moore bound for (4, 10) is 246.
    # Best known is 320.
    
    # Quick circulant search with 2 and 3 steps
    for n in range(246, 400):
        if time.time() - t_start > _DEADLINE - 15:
            break
        max_step = n // 2
        # 2-step circulants
        for s1 in range(1, max_step + 1):
            if time.time() - t_start > _DEADLINE - 15:
                break
            for s2 in range(s1 + 1, max_step + 1):
                if (s1 + s2) % n == 0:
                    continue
                if _check_girth_circulant(n, [s1, s2], 10):
                    edges = _circulant_edges(n, [s1, s2])
                    return n, edges
        # 3-step circulants (only for larger n where 2-step failed)
        if n >= 280:
            for s1 in range(1, max_step + 1):
                if time.time() - t_start > _DEADLINE - 15:
                    break
                for s2 in range(s1 + 1, max_step + 1):
                    if time.time() - t_start > _DEADLINE - 15:
                        break
                    for s3 in range(s2 + 1, max_step + 1):
                        # Check no sum of two equals another (would create short cycles)
                        diffs = [(s1+s2)%n, (s1-s2)%n, (s2-s1)%n, (s1+s3)%n, (s1-s3)%n,
                                 (s3-s1)%n, (s2+s3)%n, (s2-s3)%n, (s3-s2)%n,
                                 (s1+s2+s3)%n, (s1+s2-s3)%n, (s1-s2+s3)%n, (-s1+s2+s3)%n]
                        if 0 in diffs:
                            continue
                        if _check_girth_circulant(n, [s1, s2, s3], 10):
                            edges = _circulant_edges(n, [s1, s2, s3])
                            return n, edges
    
    # Stochastic local search: 2-switches to eliminate short cycles
    result = _stochastic_local_search(k, g, t_start)
    if result is not None:
        return result
    
    # Fallback to LUW
    return _luw(k, g)

def _circulant_edges(n, steps):
    """Generate edge list for circulant graph C(n, steps)."""
    edges = []
    for u in range(n):
        for s in steps:
            v = (u + s) % n
            if u < v:
                edges.append((u, v))
    return edges

def _stochastic_local_search(k, g, t_start):
    """
    Stochastic local search using 2-switches to maximize girth.
    Uses fast local cycle detection instead of full girth recomputation.
    """
    target_sizes = [320, 324, 328, 332, 336, 340]
    
    for n in target_sizes:
        if time.time() - t_start > _DEADLINE - 10:
            break
        
        # Generate random k-regular graph (try multiple times)
        adjacency = None
        for attempt in range(20):
            adjacency = _generate_random_k_regular(n, k)
            if adjacency is not None:
                break
        if adjacency is None:
            continue
        
        # Build adjacency list for faster operations
        adj_list = [[] for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if adjacency[u][v]:
                    adj_list[u].append(v)
                    adj_list[v].append(u)
        
        # Compute initial girth
        girth = _compute_girth_fast(adj_list, n, g)
        
        if girth >= g:
            edges = _adj_list_to_edges(adj_list, n)
            return n, edges
        
        # Simulated annealing with 2-switches
        best_girth = girth
        best_adj_list = [list(row) for row in adj_list]
        current_girth = girth
        
        T_start = 3.0
        T_end = 0.05
        max_iters = 30000
        
        for iteration in range(max_iters):
            if time.time() - t_start > _DEADLINE - 5:
                break
            
            # Generate a random 2-switch from adjacency list
            switch = _generate_2_switch_list(adj_list, n)
            if switch is None:
                continue
            
            u, v, w, x = switch
            # Apply 2-switch: remove (u,v) and (w,x), add (u,w) and (v,x)
            adj_list[u].remove(v)
            adj_list[v].remove(u)
            adj_list[w].remove(x)
            adj_list[x].remove(w)
            adj_list[u].append(w)
            adj_list[w].append(u)
            adj_list[v].append(x)
            adj_list[x].append(v)
            
            # Fast local girth check
            new_girth = _compute_girth_fast(adj_list, n, g)
            
            if new_girth >= g:
                edges = _adj_list_to_edges(adj_list, n)
                return n, edges
            
            # Accept if improvement or with probability exp(-(delta)/T)
            delta = current_girth - new_girth
            if delta <= 0:
                current_girth = new_girth
                if new_girth > best_girth:
                    best_girth = new_girth
                    best_adj_list = [list(row) for row in adj_list]
            else:
                T = T_start * (T_end / T_start) ** (iteration / max_iters)
                if random.random() < (2.718281828 ** (-delta / T)):
                    current_girth = new_girth
                else:
                    # Revert
                    adj_list[u].remove(w)
                    adj_list[w].remove(u)
                    adj_list[v].remove(x)
                    adj_list[x].remove(v)
                    adj_list[u].append(v)
                    adj_list[v].append(u)
                    adj_list[w].append(x)
                    adj_list[x].append(w)
        
        if best_girth >= g:
            edges = _adj_list_to_edges(best_adj_list, n)
            return n, edges
    
    return None

def _generate_random_k_regular(n, k):
    """
    Generate a random k-regular graph using improved configuration model.
    Returns adjacency matrix or None if failed.
    """
    # Create half-edges: each vertex has k stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    # Use a more robust pairing approach
    for attempt in range(50):
        random.shuffle(stubs)
        adjacency = [[0] * n for _ in range(n)]
        valid = True
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                valid = False
                break
            if adjacency[u][v]:
                valid = False
                break
            adjacency[u][v] = 1
            adjacency[v][u] = 1
        if valid:
            return adjacency
    
    return None

def _compute_girth_fast(adj_list, n, g):
    """
    Fast girth computation using BFS from each vertex with early termination.
    Returns the girth (shortest cycle length), or g if no cycle shorter than g.
    """
    min_cycle = g  # We only care about cycles < g
    
    for start in range(n):
        if min_cycle <= 3:
            break
        # BFS to find shortest cycle back to start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] >= min_cycle - 1:
                continue
            
            for v in adj_list[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and dist[v] < dist[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        if min_cycle <= 3:
                            return min_cycle
                    if cycle_len < g:
                        pass  # Could count, but we just need girth
    
    return min_cycle

def _adj_list_to_edges(adj_list, n):
    """Convert adjacency list to edge list (no duplicates)."""
    edges = []
    for u in range(n):
        for v in adj_list[u]:
            if u < v:
                edges.append((u, v))
    return edges

def _generate_2_switch_list(adj_list, n):
    """
    Generate a random valid 2-switch from adjacency list.
    Returns (u, v, w, x) where (u,v) and (w,x) are edges that can be switched.
    """
    # Collect all edges from adjacency list
    edges = []
    for u in range(n):
        for v in adj_list[u]:
            if u < v:
                edges.append((u, v))
    
    if len(edges) < 2:
        return None
    
    # Build adjacency set for O(1) lookup
    adj_set = set()
    for u in range(n):
        for v in adj_list[u]:
            if u < v:
                adj_set.add((u, v))
            else:
                adj_set.add((v, u))
    
    # Try several random pairs
    for _ in range(15):
        e1, e2 = random.sample(edges, 2)
        u, v = e1
        w, x = e2
        
        # Check if the switch is valid (no self-loops, no multi-edges)
        # Switch type 1: (u,w) and (v,x)
        if u != w and v != x:
            key1 = (u, w) if u < w else (w, u)
            key2 = (v, x) if v < x else (x, v)
            if key1 not in adj_set and key2 not in adj_set:
                return (u, v, w, x)
        # Switch type 2: (u,x) and (v,w)
        if u != x and v != w:
            key1 = (u, x) if u < x else (x, u)
            key2 = (v, w) if v < w else (w, v)
            if key1 not in adj_set and key2 not in adj_set:
                return (u, v, x, w)
    
    return None

def _adjacency_to_edges(adjacency, n):
    """Convert adjacency matrix to edge list."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adjacency[u][v]:
                edges.append((u, v))
    return edges

def _compute_girth_and_cycles(adjacency, n, g):
    """
    Compute the girth (shortest cycle length) and count cycles of length < g.
    Returns (girth, cycle_count).
    """
    adj_list = [[] for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if adjacency[u][v]:
                adj_list[u].append(v)
                adj_list[v].append(u)
    girth = _compute_girth_fast(adj_list, n, g)
    return girth, 0

def _check_girth_circulant(n, steps, g):
    """
    Check if circulant graph C(n, steps) has girth at least g.
    Uses BFS from vertex 0 and checks for short cycles.
    """
    # Compute neighbors for each vertex
    def get_neighbors(u):
        nbrs = []
        for s in steps:
            v1 = (u + s) % n
            v2 = (u - s) % n
            nbrs.append(v1)
            nbrs.append(v2)
        return nbrs
    
    # BFS from 0 to depth g//2 - 1
    max_depth = g // 2 - 1  # For g=10, max_depth=4
    dist = [-1] * n
    parent = [-1] * n
    dist[0] = 0
    q = deque([0])
    
    counts = [0] * (max_depth + 1)
    counts[0] = 1
    
    while q:
        u = q.popleft()
        d = dist[u]
        if d >= max_depth:
            continue
        
        nbrs = get_neighbors(u)
        for v in nbrs:
            if dist[v] == -1:
                dist[v] = d + 1
                parent[v] = u
                q.append(v)
                if d + 1 <= max_depth:
                    counts[d + 1] += 1
            else:
                # Check for short cycle
                if v != parent[u] and dist[v] <= d:
                    cycle_len = d + 1 + dist[v]
                    if cycle_len < g:
                        return False
    
    # Check that the BFS tree has the expected number of nodes
    # For a graph with girth >= g, the number of nodes at distance d from 0
    # should be at least k*(k-1)^(d-1) for d >= 1 (if n is large enough)
    k = len(steps) * 2
    for d in range(1, max_depth + 1):
        expected = k * (k - 1) ** (d - 1)
        if counts[d] < expected:
            return False
    
    return True

def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction CD(d, q)."""
    q = k
    if (g - 5) % 2 == 0:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    start = (0, tuple([None] + [0] * d))

    def neighbours(vertex):
        side, x = vertex
        x = list(x)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    index = {start: 0}
    order = [start]
    edges = []

    for vi, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
