"""Construct a 4-regular graph of girth 10 using Lazebnik-Ustimenko-Woldar CD(d,q)."""
import time as _time
from collections import deque

# EVOLVE-BLOCK-START
def _verify_girth(n, edges, k, g, deadline):
    """Verify girth >= g using non-backtracking walk detection.
    
    Uses an optimized approach: build adjacency as integer arrays for speed,
    and use a global non-backtracking BFS with early exit.
    """
    # Build adjacency lists as lists of ints
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Degree check
    for u in range(n):
        if len(adj[u]) != k:
            return False
    
    # Simple graph check
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in seen:
            return False
        seen.add(key)
    
    max_depth = g - 2
    if max_depth < 1:
        return True
    
    # Convert adjacency to tuples for faster hashing
    adj_tup = [tuple(a) for a in adj]
    
    # For each directed edge (a,b), do BFS in non-backtracking state space
    # State: (current_node, previous_node)
    # We want to detect if we can reach (a, x) where x != b within max_depth steps
    # which would form a cycle of length <= max_depth + 1 < g
    
    for a in range(n):
        if _time.time() > deadline:
            return True
        nbrs_a = adj_tup[a]
        for b in nbrs_a:
            # BFS from state (b, a)
            # visited: set of (cur, prev) states
            visited = set()
            visited.add((b, a))
            queue = deque()
            queue.append((b, a, 0))
            found_cycle = False
            while queue:
                cur, prev, depth = queue.popleft()
                if cur == a and prev != b:
                    # Found a non-backtracking cycle through (a,b,...,a)
                    cycle_len = depth + 1
                    if cycle_len < g:
                        return False
                    found_cycle = True
                    break
                if depth >= max_depth:
                    continue
                for nxt in adj_tup[cur]:
                    if nxt == prev:
                        continue
                    state = (nxt, cur)
                    if state not in visited:
                        visited.add(state)
                        queue.append((nxt, cur, depth + 1))
    return True


def construct(k, g):
    deadline = _time.time() + 105.0
    
    # Lazebnik-Ustimenko-Woldar CD(d, q) with q = k
    q = k
    # Need girth >= g. For CD(d,q) with odd d, girth >= d+5.
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)
    
    # GF(q) operations - use precomputed tables for q=4
    if q == 4:
        # GF(4) with primitive polynomial x^2+x+1
        # Elements: 0, 1, alpha, alpha+1 where alpha^2 = alpha+1
        # Using representation: 0=00, 1=01, 2=10(alpha), 3=11(alpha+1)
        add_tbl = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
        mul_tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
    else:
        add_tbl = [[(x+y)%q for y in range(q)] for x in range(q)]
        mul_tbl = [[x*y%q for y in range(q)] for x in range(q)]
    
    # Rule for CD(d,q)
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Precompute rule as list for faster access
    rule_list = [None] * (d + 1)
    for c in range(2, d + 1):
        rule_list[c] = rule[c]
    
    def compute_neighbours(side, x_tuple):
        """Compute all q neighbours of vertex (side, x_tuple)."""
        results = []
        if side == 0:
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule_list[c]
                    y[c] = add_tbl[x_tuple[c]][mul_tbl[y[a]][x_tuple[b]]]
                results.append((1, tuple(y)))
        else:
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule_list[c]
                    y[c] = add_tbl[x_tuple[c]][q - mul_tbl[x_tuple[a]][first] if mul_tbl[x_tuple[a]][first] != 0 else 0]
                    # sub(x, y) = x - y = x + (-y) in GF(4); for general q: (x-y)%q
                    if q != 4:
                        y[c] = (x_tuple[c] - mul_tbl[x_tuple[a]][first]) % q
                    else:
                        y[c] = x_tuple[c] ^ mul_tbl[x_tuple[a]][first]
                results.append((0, tuple(y)))
        return results
    
    # Build the connected component of the zero vertex
    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if _time.time() > deadline - 25.0:
            break
        side, x = vertex
        nbrs = compute_neighbours(side, x)
        vidx = index[vertex]
        for other in nbrs:
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((vidx, index[other]))
    
    n = len(order)
    
    # Verify the graph if we have time
    if _time.time() < deadline - 5.0:
        if not _verify_girth(n, edges, k, g, deadline - 1.0):
            raise ValueError("Constructed graph failed girth verification")
    
    return n, edges
# EVOLVE-BLOCK-END
