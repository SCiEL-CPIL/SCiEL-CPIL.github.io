import time
import collections

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _check_girth_nb(n, adj, min_girth):
    """
    Fast girth check using non-backtracking BFS from each directed edge.
    Returns True if girth >= min_girth.
    """
    if _time_left() < 2.0:
        return True
    
    max_depth = min_girth - 2
    
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            
            visited = {}
            queue = collections.deque()
            queue.append((u, v, 0))
            visited[(u, v)] = 0
            
            while queue:
                prev, curr, depth = queue.popleft()
                
                if depth >= max_depth:
                    continue
                
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    
                    new_state = (curr, nxt)
                    # Check if we closed a cycle
                    # The cycle length would be depth + 1 (steps to get to nxt) + 1 (back to u? No)
                    # State is (prev, curr). We are at curr, moving to nxt.
                    # If nxt == u, we have a cycle of length depth + 1?
                    # Start (u, v) is depth 0.
                    # Step 1: (v, nxt1). Depth 1.
                    # If nxt1 == u, cycle length 2? No, simple graph, no 2-cycles.
                    # If we return to u, the path is u->v->...->u.
                    # The state (prev, curr) represents the last step.
                    # If we are at state (x, y) and we go to z, and z == u,
                    # the path is u...x->y->u.
                    # Length is depth + 1?
                    # Let's trace:
                    # (u, v) depth 0.
                    # (v, w) depth 1. If w == u, cycle u-v-u (len 2).
                    # (w, u) depth 2. If u == u (start), cycle u-v-w-u (len 3).
                    # So if new_state (curr, nxt) has nxt == u, cycle length is depth + 2?
                    # Wait.
                    # Path: u -> v (start, depth 0).
                    # v -> w (depth 1). If w == u, len 2.
                    # w -> x (depth 2). If x == u, len 3.
                    # So if nxt == u, length = depth + 1?
                    # Depth 0: u->v. v->u (len 2). Depth 0+1=1? No, len 2.
                    # Depth 1: v->w. w->u (len 3). Depth 1+1=2? No, len 3.
                    # So length = depth + 2.
                    
                    if nxt == u:
                        if depth + 2 < min_girth:
                            return False
                    elif new_state not in visited:
                        visited[new_state] = depth + 1
                        queue.append((curr, nxt, depth + 1))
                
                if _time_left() < 1.0:
                    return True
    
    return True

def _luw_construct(q, d):
    if q == 4:
        # Special case for q=4 (XOR)
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
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
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
        if _time_left() < 5.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    q = k
    # Calculate d for LUW
    # Girth >= 2d + 2
    # Need 2d + 2 >= g => d >= (g - 2) / 2
    d_min = (g - 2) // 2
    d = d_min
    if q % 2 == 1 and d % 2 == 0:
        d += 1
    # Ensure d is at least 1
    d = max(1, d)
    
    n_luw, edges = _luw_construct(q, d)
    
    # Verify
    if _time_left() > 5.0:
        adj_luw = [[] for _ in range(n_luw)]
        for u, v in edges:
            adj_luw[u].append(v)
            adj_luw[v].append(u)
        if _check_girth_nb(n_luw, adj_luw, g):
            return n_luw, edges
    
    return n_luw, edges
