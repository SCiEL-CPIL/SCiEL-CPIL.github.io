"""Construct a k-regular graph of girth at least g with minimal n."""
import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Moore bound: n >= 2 * sum_{i=0}^{floor((g-1)/2)} (k-1)^i
    # For k=5, g=9: n >= 2*(1+4+16+64+256) = 682
    
    # Try sizes from known best upward
    candidate_sizes = [1116, 1200, 1500, 2000, 2500, 3000]
    
    for n_target in candidate_sizes:
        if time.time() > deadline - 25:
            break
        result = _try_rls(k, g, n_target, deadline - 10)
        if result is not None:
            return result
    
    # Larger fallbacks
    for n_target in [4000, 5000, 6000]:
        if time.time() > deadline - 8:
            break
        result = _try_rls(k, g, n_target, deadline - 3)
        if result is not None:
            return result
    
    # Last resort: LUW construction
    return _luw_fallback(k, g)


def _luw_fallback(k, g):
    """Falls back to the Lazebnik-Ustimenko-Woldar construction."""
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
    n = len(order)
    return n, edges


def _try_rls(k, g, n, deadline):
    """Try to construct a k-regular graph of girth >= g with n vertices using RLS."""
    # Generate random k-regular graph using adjacency list representation
    adj_list = _random_k_regular_adjlist(n, k)
    if adj_list is None:
        return None
    
    # Build adjacency set for fast lookup
    adj_set = [set(neighs) for neighs in adj_list]
    
    max_iterations = 20000
    iter_count = 0
    while iter_count < max_iterations:
        if time.time() > deadline:
            break
        iter_count += 1
        
        # Find a short cycle using non-backtracking BFS
        cycle = _find_short_cycle_adjlist(adj_list, adj_set, g, n)
        if cycle is None:
            # Verify thoroughly with non-backtracking walks
            if _verify_girth_adjlist(adj_list, adj_set, g, n):
                edges = []
                for u in range(n):
                    for v in adj_list[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
        
        # Break the cycle by swapping edges
        if not _break_cycle_adjlist(adj_list, adj_set, cycle, k, n):
            # Try a random perturbation
            _random_swap_adjlist(adj_list, adj_set, k, n)
    
    return None


def _random_k_regular_adjlist(n, k):
    """Generate a random k-regular graph using configuration model with adjacency lists."""
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None
    
    # Configuration model
    half_edges = []
    for v in range(n):
        for _ in range(k):
            half_edges.append(v)
    
    random.shuffle(half_edges)
    
    # Pair up half-edges
    adj_list = [set() for _ in range(n)]
    used = [False] * len(half_edges)
    
    for i in range(len(half_edges) // 2):
        while used[i]:
            i += 1
        j = i + 1
        while j < len(half_edges) and used[j]:
            j += 1
        if j >= len(half_edges):
            return None
        
        u = half_edges[i]
        v = half_edges[j]
        if u == v:
            return None
        if v in adj_list[u]:
            return None
        
        adj_list[u].add(v)
        adj_list[v].add(u)
        used[i] = True
        used[j] = True
    
    # Verify all vertices have degree k
    for v in range(n):
        if len(adj_list[v]) != k:
            return None
    
    return [list(s) for s in adj_list]


def _find_short_cycle_adjlist(adj_list, adj_set, g, n):
    """Find a cycle of length < g using non-backtracking BFS. Returns list of vertices or None."""
    for start in range(n):
        # State: (current, previous) -> distance
        visited = {}
        parent = {}  # (current, prev) -> (prev, prevprev)
        queue = deque()
        
        for nb in adj_list[start]:
            state = (nb, start)
            visited[state] = 1
            parent[state] = None
            queue.append(state)
        
        while queue:
            u, prev = queue.popleft()
            dist = visited[(u, prev)]
            if dist >= g - 1:
                continue
            for v in adj_list[u]:
                if v == prev:
                    continue
                if v == start and dist + 1 < g:
                    # Found a cycle back to start
                    # Reconstruct path
                    cycle = [start]
                    state = (u, prev)
                    while state is not None:
                        cu, cp = state
                        cycle.append(cu)
                        state = parent[state]
                    return cycle
                new_state = (v, u)
                if new_state not in visited:
                    visited[new_state] = dist + 1
                    parent[new_state] = (u, prev)
                    queue.append(new_state)
                elif parent.get((u, prev)) != (v, u) and parent.get(new_state) != (u, prev):
                    # Found a cycle - two different paths to same state
                    # Reconstruct
                    cycle = [u, v]
                    a = (u, prev)
                    b = (v, u)
                    while a != b and a is not None and b is not None:
                        a = parent.get(a)
                        b = parent.get(b)
                    if a is not None:
                        # Reconstruct from meeting point
                        ca = a
                        while ca != (u, prev):
                            cu, cp = ca
                            cycle.append(cu)
                            ca = parent.get(ca)
                        cb = a
                        while cb != (v, u):
                            cu, cp = cb
                            cycle.append(cu)
                            cb = parent.get(cb)
                    if len(cycle) >= 3:
                        return cycle
    return None


def _verify_girth_adjlist(adj_list, adj_set, g, n):
    """Verify that the graph has girth at least g using non-backtracking walks."""
    for start in range(n):
        visited = set()
        queue = deque()
        for nb in adj_list[start]:
            visited.add((nb, start))
            queue.append((nb, start, 1))
        
        while queue:
            u, prev, dist = queue.popleft()
            if dist >= g - 1:
                continue
            for v in adj_list[u]:
                if v == prev:
                    continue
                if v == start and dist + 1 < g:
                    return False
                state = (v, u)
                if state not in visited:
                    visited.add(state)
                    queue.append((v, u, dist + 1))
    return True


def _break_cycle_adjlist(adj_list, adj_set, cycle, k, n):
    """Try to break a cycle by edge swapping."""
    # Find edges on the cycle
    cycle_edges = []
    m = len(cycle)
    for i in range(m):
        u = cycle[i]
        v = cycle[(i + 1) % m]
        if v in adj_set[u]:
            cycle_edges.append((u, v))
    
    if not cycle_edges:
        return False
    
    # Try to swap one edge
    for u, v in cycle_edges:
        # Find w1 neighbor of u (not v), w2 neighbor of v (not u)
        # such that u-w2 and v-w1 don't exist
        w1_list = [w for w in adj_list[u] if w != v]
        w2_list = [w for w in adj_list[v] if w != u]
        
        for w1 in w1_list:
            for w2 in w2_list:
                if w1 == w2:
                    continue
                if w2 in adj_set[u] or w1 in adj_set[v]:
                    continue
                # Perform swap
                adj_set[u].discard(v)
                adj_set[v].discard(u)
                adj_set[u].add(w2)
                adj_set[w2].add(u)
                adj_set[v].add(w1)
                adj_set[w1].add(v)
                # Update adj_list
                adj_list[u].remove(v)
                adj_list[u].append(w2)
                adj_list[v].remove(u)
                adj_list[v].append(w1)
                adj_list[w2].append(u)
                adj_list[w1].append(v)
                return True
    
    return False


def _random_swap_adjlist(adj_list, adj_set, k, n):
    """Perform a random edge swap to maintain regularity."""
    for _ in range(50):
        u = random.randint(0, n - 1)
        if not adj_list[u]:
            continue
        v = random.choice(adj_list[u])
        
        # Find w1 neighbor of u (not v), w2 neighbor of v (not u)
        w1_list = [w for w in adj_list[u] if w != v]
        w2_list = [w for w in adj_list[v] if w != u]
        
        if not w1_list or not w2_list:
            continue
        
        w1 = random.choice(w1_list)
        w2 = random.choice(w2_list)
        
        if w1 == w2:
            continue
        if w2 in adj_set[u] or w1 in adj_set[v]:
            continue
        
        # Perform swap
        adj_set[u].discard(v)
        adj_set[v].discard(u)
        adj_set[u].add(w2)
        adj_set[w2].add(u)
        adj_set[v].add(w1)
        adj_set[w1].add(v)
        adj_list[u].remove(v)
        adj_list[u].append(w2)
        adj_list[v].remove(u)
        adj_list[v].append(w1)
        adj_list[w2].append(u)
        adj_list[w1].append(v)
        return
    
    # No valid swap found
