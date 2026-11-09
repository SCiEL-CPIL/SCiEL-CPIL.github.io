"""Stochastic local search for 4-regular graphs with girth at least g.
Starts with random 4-regular graphs and uses 2-switches to eliminate
short cycles, minimizing vertex count."""
import time
import sys
import random
from collections import deque

_DEADLINE = 115.0  # safety margin below the 120s evaluator kill


def construct(k, g):
    t_start = time.time()

    if k != 4:
        return _luw(k, g)

    # Moore bound for (4,10): 2 + 2*(3+9+27+81) = 2 + 2*120 = 246
    moore_bound = 2 + 2 * sum((k - 1) ** i for i in range(1, g // 2))
    
    best_n = None
    best_edges = None

    # Try increasing n from Moore bound
    for n in range(moore_bound, 500, 2):  # n must be even for 4-regular
        if time.time() - t_start > _DEADLINE:
            break
        
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() - t_start > _DEADLINE:
                break
            
            adj = _generate_random_k_regular(n, k)
            if adj is None:
                continue
            
            # Run local search to eliminate short cycles
            adj = _local_search(adj, n, k, g, t_start, _DEADLINE)
            
            # Check if girth >= g
            if _check_girth(adj, n, g):
                edges = _adjacency_to_edges(adj)
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                break  # Found a valid graph for this n, move to next n

    if best_n is not None:
        return best_n, best_edges
    
    # Fallback to LUW
    return _luw(k, g)


def _generate_random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    # Create stubs
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    
    random.shuffle(stubs)
    
    # Pair up stubs
    adj = [[0] * n for _ in range(n)]
    valid = True
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            valid = False
            break
        if adj[u][v] >= 1:  # no multi-edges
            valid = False
            break
        adj[u][v] = 1
        adj[v][u] = 1
    
    if not valid:
        return None
    
    return adj


def _find_short_cycles(adj, n, g):
    """Find all cycles of length < g. Returns list of cycles (as vertex lists)."""
    cycles = []
    for start in range(n):
        # BFS/DFS to find cycles through start
        # Use non-backtracking walks
        visited = set()
        stack = [(start, [start], -1)]  # (current, path, prev)
        
        while stack:
            node, path, prev = stack.pop()
            for nb in range(n):
                if adj[node][nb] and nb != prev:
                    if nb == start and len(path) >= 3:
                        if len(path) < g:
                            cycles.append(list(path))
                    elif nb not in path:
                        stack.append((nb, path + [nb], node))
    
    return cycles


def _check_girth(adj, n, g):
    """Check if graph has girth >= g using non-backtracking walks from every vertex."""
    for start in range(n):
        # BFS with non-backtracking constraint
        # State: (node, prev)
        visited = set()
        visited.add((start, -1))
        queue = deque([(start, -1, 0)])  # (node, prev, depth)
        
        while queue:
            node, prev, depth = queue.popleft()
            if depth >= g - 1:
                continue
            for nb in range(n):
                if adj[node][nb] and nb != prev:
                    if nb == start and depth + 1 < g:
                        return False  # Found cycle shorter than g
                    if (nb, node) not in visited:
                        visited.add((nb, node))
                        queue.append((nb, node, depth + 1))
    
    return True


def _local_search(adj, n, k, g, t_start, deadline):
    """Apply 2-switches to eliminate short cycles."""
    adj = [row[:] for row in adj]
    
    for iteration in range(2000):
        if time.time() - t_start > deadline - 5:
            break
        
        # Find a short cycle
        cycle = _find_one_short_cycle(adj, n, g)
        if cycle is None:
            break
        
        # Try to break the cycle with a 2-switch
        L = len(cycle)
        if L < 4:
            # Triangle - can't 2-switch, try removing and reconnecting
            continue
        
        switched = False
        # Try different pairs of edges to switch
        for i in range(L):
            for j in range(i + 2, L):
                if j == i + L - 1 and L > 4:
                    continue  # adjacent edges (wrap around)
                if j == i + 1:
                    continue  # adjacent edges
                
                a1, a2 = cycle[i], cycle[(i + 1) % L]
                b1, b2 = cycle[j], cycle[(j + 1) % L]
                
                # Check if new edges are valid
                if not adj[a1][b2] and not adj[a2][b1]:
                    # Perform switch
                    adj[a1][a2] = 0
                    adj[a2][a1] = 0
                    adj[b1][b2] = 0
                    adj[b2][b1] = 0
                    adj[a1][b2] = 1
                    adj[b2][a1] = 1
                    adj[a2][b1] = 1
                    adj[b1][a2] = 1
                    switched = True
                    break
            if switched:
                break
        
        if not switched:
            # Can't break this cycle, try another one or give up
            if iteration > 100:
                break
    
    return adj


def _find_one_short_cycle(adj, n, g):
    """Find one cycle of length < g using BFS from each vertex."""
    for start in range(n):
        # BFS with non-backtracking
        visited = set()
        visited.add((start, -1))
        queue = deque()
        for nb in range(n):
            if adj[start][nb]:
                queue.append((nb, start, 1, [start, nb]))
        
        while queue:
            node, prev, depth, path = queue.popleft()
            if depth >= g - 1:
                continue
            for nb in range(n):
                if adj[node][nb] and nb != prev:
                    if nb == start and depth + 1 < g:
                        return path
                    if (nb, node) not in visited:
                        visited.add((nb, node))
                        queue.append((nb, node, depth + 1, path + [nb]))
    
    return None


def _adjacency_to_edges(adj):
    """Convert adjacency matrix to edge list."""
    edges = []
    n = len(adj)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append((i, j))
    return edges


def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction CD(d, q)."""
    q = k
    # girth = d + 5 for odd d, so d = g - 5 for odd g, g - 4 for even g
    # Actually the paper says girth is at least d + 5.
    # For g=10, we want d+5 >= 10, so d >= 5. Use d=5.
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

    # Build the rule
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Use BFS to find the connected component
    from collections import deque as dq
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
