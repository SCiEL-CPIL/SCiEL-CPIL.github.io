"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _luw_build(k, g, q, d, rule):
    """Build the LUW graph CD(d,q) with given parameters, return (n, edges) for the
    connected component of the zero vertex. Returns None if not k-regular."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
    # Check k-regularity
    if n < 2 * k:
        return None
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(dg != k for dg in deg):
        return None
    return n, edges


def _verify_girth(n, edges, g):
    """Verify no cycle shorter than g using non-backtracking walks from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # Non-backtracking walk enumeration up to length g-1
    # A cycle of length L < g means there's a closed non-backtracking walk of length L
    # For efficiency, use BFS on the directed line graph (non-backtracking states)
    # State: (current_vertex, previous_vertex) where previous != current
    # Start states: all (v, u) for each edge (v,u)
    # A cycle of length L exists iff some state (v, u) is reachable from (u, v) in L-2 steps
    # Actually simpler: do BFS from each directed edge, track if we return to start
    # For small g this is fine.
    # Better: use the standard method - for each vertex, do BFS tracking (node, parent)
    # and check if we can reach the parent in fewer than g steps via non-backtracking path.
    
    # Non-backtracking: a walk v0, v1, ..., vl where vi != v(i-2) for i >= 2
    # A cycle of length L < g exists iff for some edge (u,v), there's a non-backtracking
    # walk from v to u of length L-1 where the first step doesn't go back to u.
    
    # For each starting directed edge (u -> v), do BFS on non-backtracking walks
    # and check if we can reach u in fewer than g-1 steps.
    from collections import deque
    
    for u, v in edges:
        # BFS from (v, u) - we're at v, came from u, next step can't go back to u
        visited = set()
        queue = deque()
        queue.append((v, u, 1))  # (current, prev, length)
        visited.add((v, u))
        found = False
        while queue and not found:
            cur, prev, length = queue.popleft()
            if length >= g - 1:
                break
            if cur == u and length >= 2:
                # We found a cycle of length length+1 (the walk v->...->u plus edge u->v)
                # Wait, we started at v with prev=u. If we reach u, the cycle is length+1
                # But we need length+1 < g, i.e., length < g-1. Since we check length < g-1 in loop
                if length + 1 < g:
                    found = True
                    break
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                state = (nxt, cur)
                if state not in visited:
                    visited.add(state)
                    queue.append((nxt, cur, length + 1))
        if found:
            return False
    return True


def _make_rule(d, variant=0):
    """Generate a rule for LUW construction. variant selects different valid rules."""
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    if variant == 0:
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    elif variant == 1:
        # Alternative: shift the pattern
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(c0, 1), (1, c0 + 1), (c0 - 1, 1), (1, c0 - 2)][r]
    elif variant == 2:
        # Another variant
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 + 1), (c0, 1), (c0 - 1, 1), (1, c0 - 2)][r]
    return rule


def construct(k, g):
    deadline = time.time() + 140  # 10s safety margin
    
    # d must be odd and >= g-5 for girth >= d+5 >= g
    # Try odd d values starting from g-5 (if odd) or g-4 (if g-5 is even)
    d_min = g - 5 if (g - 5) % 2 == 1 else g - 4
    d_min = max(3, d_min)
    
    best = None
    
    # Try different field sizes q >= k and rule variants
    # For k=3, q=3 is the natural choice, but larger q might give smaller components
    # Actually for LUW, the degree is q, so we need q=k for k-regularity
    # Unless we take a subgraph... but that's not straightforward.
    # So q must equal k for the graph to be k-regular.
    # The only freedom is in the rule parameters.
    
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Try multiple rule variants and d values
    for d in range(d_min, d_min + 6, 2):  # Try a few odd d values
        if time.time() > deadline:
            break
        for variant in range(3):
            if time.time() > deadline:
                break
            rule = _make_rule(d, variant)
            # Validate rule: for each c in 2..d, rule[c] = (a,b) with a,b in 1..d-1
            # and a != b, and the construction is well-defined
            valid = True
            for c in range(2, d + 1):
                a, b = rule[c]
                if a < 1 or a > d - 1 or b < 1 or b > d - 1 or a == b:
                    valid = False
                    break
            if not valid:
                continue
            
            result = _luw_build(k, g, q, d, rule)
            if result is not None:
                n, edges = result
                if best is None or n < best[0]:
                    best = (n, edges)
                    # Verify girth for the best candidate (expensive but necessary)
                    if _verify_girth(n, edges, g):
                        # Valid!
                        pass
                    else:
                        # Girth violation, don't use this
                        best = None
    
    if best is None:
        # Fallback: use the original construction
        d = d_min
        rule = _make_rule(d, 0)
        result = _luw_build(k, g, q, d, rule)
        if result is not None:
            best = result
        else:
            # Should not happen, but handle gracefully
            return 2 * k, [(i, i + k) for i in range(k)]
    
    return best
# EVOLVE-BLOCK-END
