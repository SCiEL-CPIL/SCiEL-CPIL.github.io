import time
import random
import numpy as np
from itertools import combinations_with_replacement, product as iproduct

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s kill

    if k != 3 or g != 16:
        return _luw_construct(k, g)

    # Moore bound for k=3, g=16
    moore_bound = 1 + k * sum((k-1)**i for i in range(g//2))

    # Phase 1: Circulant graph construction
    # For a circulant graph on Z_n with connection set S (symmetric, |S|=3),
    # the girth is determined by the shortest non-trivial relation sum of ±s = 0.
    # For k=3, S = {a, -a, b} where 2b ≡ 0 (mod n) (b is self-inverse).
    # Actually for undirected 3-regular: S must be symmetric.
    # If n is odd: S = {a, -a, b} requires b = -b mod n, so 2b ≡ 0, so b=0 (invalid)
    #   unless n is even. For odd n, we can't have a self-inverse element other than 0.
    #   Wait: for odd n, -a ≠ a for all a≠0. So S must have 3 elements that are
    #   symmetric. This means one element must be self-inverse: 2x ≡ 0 mod n.
    #   For odd n, only x=0 works, which is invalid. So odd n doesn't work for k=3 circulant.
    # For even n: b = n/2 is self-inverse. S = {a, -a, n/2} for some a.
    #   But also need: no element is 0, and S has exactly 3 distinct elements.
    #   a ≠ 0, -a ≠ 0, a ≠ n/2, -a ≠ n/2, a ≠ -a (so 2a ≠ 0, i.e., a ≠ n/2).
    #   So a ∈ {1, ..., n-1} \ {n/2}, and we take S = {a, n-a, n/2}.
    #   Also need a ≠ n-a, i.e., 2a ≠ 0 mod n, i.e., a ≠ n/2 (already excluded).
    
    # For even n: try all valid a values, check girth
    # Girth check: shortest cycle in circulant with steps {a, -a, n/2}
    # A cycle is a sequence of steps s_1, ..., s_L (each in {a, -a, n/2}) 
    # summing to 0 mod n, with no backtracking (consecutive steps not canceling).
    
    # Let's search for small even n starting from moore_bound (rounded up to even)
    n_start = moore_bound if moore_bound % 2 == 0 else moore_bound + 1
    # Actually moore_bound for k=3,g=16: 1 + 3*(1+2+4+8+16+32+64) = 1 + 3*127 = 382
    # Wait: 1 + 3*sum_{i=0}^{7} 2^i = 1 + 3*(2^8 - 1) = 1 + 3*255 = 766
    # So moore_bound = 766.
    
    phase1_deadline = start_time + 60
    
    best_n = None
    best_edges = None
    
    for n in range(n_start, n_start + 400):
        if n % 2 != 0:
            continue
        if time.time() > phase1_deadline:
            break
        
        half = n // 2
        # Try various a values
        # Heuristic: try a values that are "spread out" to maximize girth
        # Good candidates: a near n/4, or a with large order
        candidates_a = []
        # Try a = n/4, n/3, etc. (when they're integers)
        if n % 4 == 0:
            candidates_a.append(n // 4)
        if n % 3 == 0:
            candidates_a.append(n // 3)
        # Try golden-ratio-like fractions
        for frac in [0.25, 0.3, 0.333, 0.382, 0.4, 0.414, 0.45, 0.5-0.05, 0.5-0.1]:
            a_val = int(n * frac)
            if 0 < a_val < half:
                candidates_a.append(a_val)
        # Also try small values and values near half
        for a_val in [1, 2, 3, 5, 7, 11, 13, 17]:
            if 0 < a_val < half:
                candidates_a.append(a_val)
        for a_val in [half-1, half-2, half-3, half-5]:
            if 0 < a_val < half:
                candidates_a.append(a_val)
        
        # Deduplicate and sort
        candidates_a = sorted(set(candidates_a))
        
        for a in candidates_a:
            if time.time() > phase1_deadline:
                break
            # Connection set: {a, n-a, half}
            # Check girth via BFS on the non-backtracking state space
            girth = _circulant_girth(n, a, half)
            if girth >= g:
                edges = _circulant_edges(n, a, half)
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                # We found one with this n, no need to try more a for this n
                break
        else:
            # No a worked for this n, continue to next n
            continue
        break  # Found a valid n, stop searching
    
    if best_n is not None:
        return best_n, best_edges
    
    # Phase 2: Broader circulant search with more a values
    phase2_deadline = start_time + 100
    for n in range(n_start, n_start + 400):
        if n % 2 != 0:
            continue
        if time.time() > phase2_deadline:
            break
        half = n // 2
        # Try more a values
        for a in range(1, half):
            if time.time() > phase2_deadline:
                break
            girth = _circulant_girth(n, a, half)
            if girth >= g:
                edges = _circulant_edges(n, a, half)
                return n, edges
    
    # Phase 3: Fallback to LUW
    return _luw_construct(k, g)

def _circulant_girth(n, a, half):
    """Compute the girth of circulant graph on Z_n with steps {a, -a, half}.
    Uses non-backtracking BFS. Returns the girth (shortest cycle length).
    """
    # Steps: +a, -a, +half (which equals -half since 2*half = 0 mod n)
    # Non-backtracking: can't go +a then -a (or vice versa), can't go +half then +half
    #   (since +half then +half = 0, that's a 2-cycle which is invalid for simple graph)
    #   Actually in a simple graph, no 2-cycles. +half is a self-inverse step.
    #   Going +half then +half returns to start, which is a 2-cycle. But in a simple
    #   graph, there's no edge from v to v, so this can't happen.
    #   Wait: v --(half)--> v+half --(half)--> v+2half = v (mod n). 
    #   But v and v+half are connected by an edge, and v+half and v are the same edge.
    #   So this is just traversing the same edge back and forth, which is backtracking.
    #   So non-backtracking forbids: after +a, can't do -a; after -a, can't do +a;
    #   after +half, can't do +half (since it's the same edge).
    
    # State: (position, last_step) where last_step in {0: +a, 1: -a, 2: +half}
    # Start from position 0, try each initial step
    # BFS to find shortest cycle
    
    steps = [a, -a % n, half]
    # Transitions: from (pos, last_step), can go to (pos + steps[s], s) for s != last_step
    #   and also not the "reverse" of last_step
    # Reverse of +a is -a (step 1), reverse of -a is +a (step 0), reverse of +half is +half (step 2)
    reverse = {0: 1, 1: 0, 2: 2}
    
    from collections import deque
    
    for start_step in range(3):
        # BFS: state = (position, last_step), distance = number of steps taken
        # We want to return to position 0 with a different last_step (to close the cycle)
        # Actually: a cycle of length L means after L steps we're back at position 0,
        # and the last step is not the reverse of the first step (non-backtracking)
        
        visited = {}  # (pos, last_step) -> distance
        queue = deque()
        pos = steps[start_step] % n
        visited[(pos, start_step)] = 1
        queue.append((pos, start_step, 1))
        
        while queue:
            pos, last, dist = queue.popleft()
            if dist >= 20:  # Cap to avoid infinite loops; girth 16 is what we need
                continue
            for s in range(3):
                if s == reverse[last]:
                    continue  # non-backtracking
                new_pos = (pos + steps[s]) % n
                new_state = (new_pos, s)
                if new_pos == 0:
                    # We've returned to start!
                    # Cycle length is dist + 1
                    # Need to check: the step s should not be the reverse of start_step
                    # for it to be a valid non-backtracking cycle
                    if s != reverse[start_step]:
                        return dist + 1
                    # If s == reverse[start_step], this would be backtracking, skip
                if new_state in visited:
                    # Found a cycle
                    cycle_len = dist + 1 - visited[new_state]
                    if cycle_len > 1:  # Must be at least 3 for a simple graph
                        return cycle_len
                else:
                    visited[new_state] = dist + 1
                    queue.append((new_pos, s, dist + 1))
    
    return 100  # No short cycle found (shouldn't happen for connected graphs)

def _circulant_edges(n, a, half):
    """Generate edges for circulant graph on Z_n with steps {a, -a, half}."""
    edges = []
    a_mod = a % n
    for v in range(n):
        # Edge to v + a (only add if v < v+a to avoid duplicates)
        u = (v + a_mod) % n
        if v < u:
            edges.append((v, u))
        # Edge to v - a
        u = (v - a_mod) % n
        if v < u:
            edges.append((v, u))
        # Edge to v + half
        u = (v + half) % n
        if v < u:
            edges.append((v, u))
    return edges

def _luw_construct(k, g):
    # Original code
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

def _luw_construct(k, g):
    # Original code
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

def _generate_random_regular_graph(n, k):
    """Generate a random simple k-regular graph using configuration model with retries."""
    if n * k % 2 != 0 or n < k:
        return None
    
    max_retries = 50
    for _ in range(max_retries):
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        
        edges = set()
        adj = [[] for _ in range(n)]
        valid = True
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if (min(u,v), max(u,v)) in edges:
                valid = False
                break
            edges.add((min(u,v), max(u,v)))
            adj[u].append(v)
            adj[v].append(u)
        
        if valid:
            return list(edges)
    
    return None

def _check_girth_fast(edges, n, g):
    """Fast girth check using non-backtracking BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    depth_limit = g - 1
    
    for start in range(n):
        visited = {}
        queue = []
        for nb in adj[start]:
            state = (start, nb)
            visited[state] = 1
            queue.append((start, nb, 1))
        
        while queue:
            u, prev, depth = queue.pop(0)
            if depth >= depth_limit:
                continue
            for v in adj[u]:
                if v == prev:
                    continue
                new_state = (u, v)
                if new_state in visited:
                    cycle_len = depth - visited[new_state]
                    if cycle_len < g:
                        return False
                else:
                    visited[new_state] = depth + 1
                    queue.append((u, v, depth + 1))
    return True

def _improve_girth(edges, n, g, max_iter, time_limit):
    return None
