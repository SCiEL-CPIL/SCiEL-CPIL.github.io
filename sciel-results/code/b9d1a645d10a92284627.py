"""
Constructs a 3-regular graph of girth 16 with minimal vertices.
Uses a targeted search for Cayley graphs on cyclic groups Z_N,
fixing the odd-degree connection set bug in the seed program.
"""
import time
import random
import math
from collections import deque

_DEADLINE = 140.0

def _moore_bound(k, g):
    if k <= 1:
        return 0
    if g % 2 == 1:
        return 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))
    else:
        return 2 * sum((k - 1) ** i for i in range(g // 2))

def _girth_of_cayley(N, S, g):
    """
    Shortest non-trivial zero-sum length of S-elements.
    S is a list of distinct nonzero residues mod N, closed under inverses.
    """
    inv = {}
    for s in S:
        inv[s] = (-s) % N
    
    # BFS on (sum, last)
    # State: (current_sum, last_step)
    # Start: (0, 0) where 0 represents no previous step
    start = (0, -1)
    dist = {start: 0}
    q = deque([start])
    
    while q:
        sm, last = q.popleft()
        d = dist[(sm, last)]
        if d >= g - 1:
            break
            
        for s in S:
            # No immediate backtracking
            if last != -1 and s == inv[last]:
                continue
                
            ns = (sm + s) % N
            nd = d + 1
            
            if ns == 0 and nd >= 3:
                return nd
                
            state = (ns, s)
            if state not in dist:
                dist[state] = nd
                q.append(state)
                
    return None

def _make_edges(N, S):
    edges = []
    seen = set()
    for u in range(N):
        for s in S:
            v = (u + s) % N
            if u < v:
                key = (u, v)
                if key not in seen:
                    seen.add(key)
                    edges.append((u, v))
    return edges

def _search_cyclic_k3(g, deadline):
    """
    Search for a valid 3-regular Cayley graph on Z_N with girth >= g.
    For k=3, S must be {N/2, v, N-v} where 2*(N/2)=0 mod N.
    """
    mb = _moore_bound(3, g)
    # Start from Moore bound, ensure even N
    start_N = mb if mb % 2 == 0 else mb + 1
    # Search up to a reasonable limit
    max_N = 2000
    
    for N in range(start_N, max_N + 1, 2):
        if time.time() > deadline:
            return None
        
        u = N // 2
        # v must be such that S = {u, v, N-v} are distinct
        # u != v and u != N-v and v != N-v
        # v != N-v implies 2v != 0 mod N. Since N is even, v != N/2 = u.
        # So we just need v not equal to u.
        
        for v in range(1, N // 2):
            if v == u:
                continue
            S = [u, v, N - v]
            # Check distinctness
            if len(set(S)) != 3:
                continue
            
            # Check girth
            if _girth_of_cayley(N, S, g) is None:
                return N, S
                
    return None

def _luw_fallback(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction as a fallback.
    """
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = lambda x, y: x ^ y
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

def construct(k, g):
    t0 = time.time()
    deadline = t0 + (_DEADLINE - 10.0)
    
    # Specific optimization for the known task: k=3, g=16
    if k == 3 and g == 16:
        result = _search_cyclic_k3(g, deadline)
        if result:
            N, S = result
            edges = _make_edges(N, S)
            return N, edges
    
    # General fallback for other k, g or if search fails
    # For k=3, g=16, if search fails, use LUW
    # Note: LUW is very large, but valid.
    return _luw_fallback(k, g)
