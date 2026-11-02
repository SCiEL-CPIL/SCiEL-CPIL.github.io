import time
import math
from math import gcd

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 120
    # Moore bound for k=3, g=16 is 766
    # We search for circulant graphs C(n; {a, -a, n/2})
    # n must be even.
    
    # Calculate Moore bound
    # 1 + k * sum((k-1)^i for i in range(0, (g-2)//2))
    # For g=16, (g-2)//2 = 7.
    # sum = 1 + 2 + ... + 2^7 = 255.
    # 1 + 3*255 = 766.
    
    start_n = 766
    if start_n % 2 != 0:
        start_n += 1
        
    # We will search n up to some limit or until deadline
    # The LUW graph has ~354k vertices.
    # Even n=2000 is a huge improvement.
    max_n = 10000
    
    for n in range(start_n, max_n, 2):
        if time.time() > deadline - 10:
            break
            
        # Try all a
        # a must be coprime to n/2 for connectivity
        half_n = n // 2
        for a in range(1, half_n):
            if time.time() > deadline - 5:
                break
            if gcd(a, half_n) != 1:
                continue
                
            # Check girth
            # We check if there is a simple cycle of length < g passing through 0.
            # Since the graph is vertex-transitive, this is sufficient.
            if _has_short_cycle_circulant(n, a, g):
                continue
            
            # Construct edges
            edges = []
            step1 = a
            step2 = -a % n
            step3 = half_n
            
            for i in range(n):
                # Add edges (i, i+s) for s in {step1, step2, step3}
                # To avoid duplicates, only add if i < (i+s) % n ?
                # Or just add all and deduplicate?
                # Better: add (i, j) if i < j.
                j1 = (i + step1) % n
                j2 = (i + step2) % n
                j3 = (i + step3) % n
                
                if i < j1:
                    edges.append((i, j1))
                if i < j2:
                    edges.append((i, j2))
                if i < j3:
                    edges.append((i, j3))
            
            return n, edges

    # Fallback to LUW if no circulant found
    return _luw_fallback(k, g)

def _has_short_cycle_circulant(n, a, g):
    # Check for simple cycle of length < g through 0
    # DFS
    target_len = g
    # visited in current path
    # We can use a set or a list since depth is small
    
    # Optimization: 
    # A cycle of length L corresponds to s1 + ... + sL = 0
    # where si in {a, -a, n/2}
    # And partial sums are distinct (except last=0)
    
    # We can just do DFS on the graph
    # Nodes are 0..n-1
    # Start at 0
    # Depth limit g-1 (since cycle length L means path of length L from 0 to 0)
    # Actually, if we find a path of length L < g from 0 to 0 with no intermediate 0, it's a cycle.
    
    # Since n can be large, but depth is small, DFS is fine.
    
    # We can prune if we hit a node we've seen in the current path.
    
    path = [0]
    visited = {0}
    
    def dfs(node, depth):
        if depth >= target_len:
            return False
        for step in (a, -a % n, n // 2):
            nxt = (node + step) % n
            if nxt == 0:
                # Found a cycle of length depth + 1
                if depth + 1 < target_len:
                    return True
                else:
                    continue
            if nxt in visited:
                continue
            visited.add(nxt)
            path.append(nxt)
            if dfs(nxt, depth + 1):
                return True
            path.pop()
            visited.remove(nxt)
        return False
        
    return dfs(0, 0)

def _luw_fallback(k, g):
    # Original LUW implementation
    deadline = time.time() + 120
    q = k
    d = g - 5
    if d % 2 == 0:
        d = g - 4
    d = max(3, d)

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
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
