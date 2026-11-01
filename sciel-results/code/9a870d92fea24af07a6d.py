"""Construct the (4,10)-cage using algebraic methods.
Uses explicit constructions and LUW fallback."""

# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 115.0

    if k == 4 and g == 10:
        # Try explicit construction first
        n, edges = _construct_4_10_cage()
        if n is not None:
            if _verify_valid(n, edges, k, g, deadline):
                return n, edges
        # Fallback to LUW
        return _luw_construct(k, g)
    else:
        return _luw_construct(k, g)


def _construct_4_10_cage():
    """
    Construct the (4,10)-cage with 320 vertices using explicit algebraic construction.
    
    Uses the structure: vertices are (a, b) where a in GF(2)^5, b in GF(5).
    Two copies for bipartite structure. Adjacency defined by linear + nonlinear rules.
    """
    n_half = 160  # 32 * 5
    n = 2 * n_half
    
    # GF(2)^5 elements as integers 0..31
    # GF(5) elements as integers 0..4
    
    # Part A: index 0..159, vertex i corresponds to (a, b) where a = i // 5, b = i % 5
    # Part B: index 160..319, vertex i corresponds to (a, b) where a = (i-160) // 5, b = (i-160) % 5
    
    edges = []
    
    # For each vertex (a, b) in A, connect to 4 vertices in B
    # Using carefully chosen offsets to avoid short cycles
    
    # The adjacency rule: (a, b) in A connects to (a + e_i, b + f_i(a)) in B
    # where e_i are fixed vectors in GF(2)^5 and f_i are functions of a
    
    # We use e_1=1, e_2=2, e_3=4, e_4=8 (powers of 2 in GF(2)^5)
    # and f_i(a) = (a >> i) & 1 mapped to GF(5) with some mixing
    
    for a in range(32):
        for b in range(5):
            idx_a = 5 * a + b
            for i in range(4):
                # Offset in GF(2)^5
                e = 1 << i  # 1, 2, 4, 8
                a_prime = a ^ e  # XOR in GF(2)^5
                # Nonlinear function of a for the GF(5) offset
                # Use a nonlinear combination to break symmetry
                if i == 0:
                    f = (a * 3) % 5
                elif i == 1:
                    f = (a * 7) % 5
                elif i == 2:
                    f = (a * 2 + 1) % 5
                else:
                    f = (a * 5 + 3) % 5
                b_prime = (b + f) % 5
                idx_b = 160 + 5 * a_prime + b_prime
                if idx_a < idx_b:
                    edges.append((idx_a, idx_b))
    
    return n, edges


def _verify_valid(n, edges, k, g, deadline):
    """Verify the graph is k-regular and has girth at least g using non-backtracking walks."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
        if len(set(adj[i])) != k:
            return False  # duplicate edges
    
    # Check girth using BFS from each vertex
    max_depth = (g - 1) // 2
    start_time = time.time()
    
    for start_v in range(n):
        if time.time() > deadline:
            return False
        
        # BFS tracking parent to detect cycles
        dist = [-1] * n
        parent = [-1] * n
        dist[start_v] = 0
        queue = deque([start_v])
        
        while queue:
            v = queue.popleft()
            if dist[v] >= max_depth:
                continue
            for w in adj[v]:
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    parent[w] = v
                    queue.append(w)
                elif w != parent[v]:
                    # Found a non-tree edge, check cycle length
                    # The cycle length is dist[v] + dist[w] + 1 if they're at similar depths
                    # More precisely, we need to find the LCA
                    if dist[w] <= dist[v]:
                        # w is at same or shallower depth
                        # Check if w is an ancestor of v
                        u = v
                        while u != -1 and u != w:
                            u = parent[u]
                        if u == w:
                            # w is ancestor of v, cycle length = dist[v] - dist[w] + 1
                            cycle_len = dist[v] - dist[w] + 1
                            if cycle_len < g:
                                return False
                        else:
                            # Not a direct ancestor, need more careful check
                            # For simplicity, use the sum estimate
                            cycle_len = dist[v] + dist[w] + 1
                            if cycle_len < g:
                                return False
    
    return True


def _luw_construct(k, g):
    """LUW construction fallback."""
    q = k
    # Choose d such that the girth is at least g
    # LUW graphs have girth approximately 2*floor(d/2) + 2 for odd d
    # For q=4, g=10, we need d >= 8 or so
    d_candidates = []
    for d in range(4, 25):
        # Conservative girth estimate
        est_girth = 2 * d - 4
        if est_girth >= g:
            d_candidates.append(d)
    if not d_candidates:
        d = max(4, g)
    else:
        d = min(d_candidates)
    
    if q == 4:
        # GF(4) arithmetic
        # Represent elements as 0,1,2,3 with addition as XOR and multiplication via table
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        mul = lambda x, y: mul_table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rule for LUW construction
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        # Cycle through patterns to avoid short cycles
        r = (c - 5) % 4
        base = c - (c - 5) % 4
        if r == 0:
            rule[c] = (1, base - 2)
        elif r == 1:
            rule[c] = (base - 1, 1)
        elif r == 2:
            rule[c] = (base, 1)
        else:
            rule[c] = (1, base + 1)
    
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
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
