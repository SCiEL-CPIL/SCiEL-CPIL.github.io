import time
import random
from collections import deque

def construct(k, g):
    if k != 5 or g != 9:
        # Fallback for other parameters
        return _seed_construct(k, g)

    deadline = time.time() + 110
    p = 11
    identity = (1, 0, 0, 1)
    
    # Generate SL(2, p)
    elements = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    if (a * d - b * c) % p == 1:
                        elements.append((a, b, c, d))
                        
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    n_elem = len(elements)
    
    def mul(e1, e2):
        a1, b1, c1, d1 = e1
        a2, b2, c2, d2 = e2
        return (
            (a1*a2 + b1*c2) % p,
            (a1*b2 + b1*d2) % p,
            (c1*a2 + d1*c2) % p,
            (c1*b2 + d1*d2) % p
        )
        
    def inv(e):
        a, b, c, d = e
        return (d, (-b) % p, (-c) % p, a)
        
    involutions = []
    for e in elements:
        if e == identity: continue
        if mul(e, e) == identity:
            involutions.append(e)
            
    def check_girth_and_conn(S):
        visited = {identity: (0, -1)}
        q = deque([identity])
        min_cycle = 100
        
        while q:
            curr = q.popleft()
            d, parent = visited[curr]
            if d >= 8: continue
            for s in S:
                nxt = mul(curr, s)
                if nxt not in visited:
                    visited[nxt] = (d + 1, curr)
                    q.append(nxt)
                else:
                    pd, pp = visited[nxt]
                    if nxt == pp:
                        continue
                    # Potential cycle
                    # The cycle length is at most d + 1 + pd
                    # If this is < 9, we have a short cycle.
                    if d + 1 + pd < min_cycle:
                        min_cycle = d + 1 + pd
                        
        if len(visited) < n_elem:
            return False, min_cycle
        return min_cycle >= g, min_cycle

    random.seed(42)
    attempts = 0
    while time.time() < deadline and attempts < 1000:
        attempts += 1
        h = random.choice(involutions)
        a = random.choice(elements)
        if a == h or a == identity or mul(a, a) == identity:
            continue
        b = random.choice(elements)
        if b == h or b == identity or b == a or b == inv(a) or mul(b, b) == identity:
            continue
            
        S = [h, a, inv(a), b, inv(b)]
        if len(set(S)) < 5:
            continue
            
        is_valid, _ = check_girth_and_conn(S)
        if is_valid:
            edges = []
            for i, u in enumerate(elements):
                for s in S:
                    v = mul(u, s)
                    j = elem_to_idx[v]
                    if i < j:
                        edges.append((i, j))
            return n_elem, edges

    return _seed_construct(k, g)

def _seed_construct(k, g):
    # Paste seed code here
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
