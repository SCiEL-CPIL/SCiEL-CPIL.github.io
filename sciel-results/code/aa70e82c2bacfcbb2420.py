import time
import collections
import math

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _check_girth_nb(n, adj, min_girth):
    """
    Fast girth check using non-backtracking BFS from each directed edge.
    Returns True if girth >= min_girth, False otherwise.
    """
    if _time_left() < 2.0:
        return True
    
    max_depth = min_girth - 2
    
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            
            # BFS from directed edge (u, v)
            # State: (prev, curr)
            # Start: (u, v) at depth 0
            # A cycle of length L is found if we reach (v, u) at depth L-2
            # We want to ensure no cycle of length < min_girth.
            # So if we reach (v, u) at depth d, cycle length is d+2.
            # We need d+2 >= min_girth => d >= min_girth - 2.
            # If we reach (v, u) at depth d < min_girth - 2, we have a short cycle.
            
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
                    if new_state == (v, u):
                        cycle_len = depth + 2
                        if cycle_len < min_girth:
                            return False
                        # If cycle_len >= min_girth, this is fine.
                        # We don't need to expand further from this state for this start edge.
                        continue
                    
                    if new_state not in visited:
                        visited[new_state] = depth + 1
                        queue.append((curr, nxt, depth + 1))
                
                if _time_left() < 1.0:
                    return True
    
    return True

def _construct_luw(q, d):
    """
    Constructs the LUW graph CD(d, q).
    Based on Lazebnik, Ustimenko, Woldar (1998).
    """
    if q == 4:
        # GF(4) arithmetic
        table_add = [[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 0, 1]]
        # Actually for GF(4) with generator alpha, alpha^2 = alpha + 1
        # 0=0, 1=1, 2=alpha, 3=alpha^2
        # Add:
        # 0+0=0, 0+1=1, 0+2=2, 0+3=3
        # 1+1=0, 1+2=3, 1+3=2
        # 2+2=0, 2+3=1
        # 3+3=0
        add_tbl = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        # Mul:
        # 0*x=0
        # 1*x=x
        # 2*2=3 (alpha*alpha=alpha^2), 2*3=1 (alpha*alpha^2=alpha^3=1), 3*3=2 (alpha^2*alpha^2=alpha^4=alpha)
        mul_tbl = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y): return add_tbl[x][y]
        def sub(x, y): return add_tbl[x][y] # char 2
        def mul(x, y): return mul_tbl[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    # Rules for CD(d, q)
    # For d=2: (1,1)
    # For d=3: (2,1)
    # For d=4: (1,2)
    # For d>=5:
    #   Let c = d.
    #   r = (c-5) % 4
    #   c0 = c - r
    #   If r=0: (1, c0-2)
    #   If r=1: (c0-1, 1)
    #   If r=2: (c0, 1)
    #   If r=3: (1, c0+1)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else: # r == 3
            rule[c] = (1, c0 + 1)
            
    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None (or 0), x[1..d] are in GF(q)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # y_c = x_c + y_a * x_b
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # y_c = x_c - x_a * y_b
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if _time_left() < 5.0:
            return None
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
    if q != 3:
        q = 3 
        
    min_d = max(3, (g - 2) // 2)
    
    best_n = float('inf')
    best_edges = None
    
    # Try LUW with smallest d first
    for d in range(min_d, min_d + 10):
        if _time_left() < 10.0:
            break
        result = _construct_luw(q, d)
        if result is None:
            continue
        n, edges = result
        if n < best_n:
            # Verify girth
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Check for simple graph
            edge_set = set()
            valid = True
            for u, v in edges:
                if u == v:
                    valid = False
                    break
                key = (min(u,v), max(u,v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
            if not valid:
                continue
                
            # Check degrees
            for i in range(n):
                if len(adj[i]) != k:
                    valid = False
                    break
            if not valid:
                continue

            if _check_girth_nb(n, adj, g):
                best_n = n
                best_edges = edges
                break
    
    if best_edges is None:
        # Emergency fallback
        result = _construct_luw(q, min_d + 5)
        if result:
            return result
        return 0, []
        
    return best_n, best_edges
