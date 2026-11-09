# EVOLVE-BLOCK-START
import itertools
import time

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    
    Strategy: 
    1. Try to find a circulant graph on n vertices (starting from the lower bound 
       up to the known optimal 320 for k=4, g=10) that satisfies the girth constraint.
    2. If a circulant is found, return it.
    3. If not, fall back to the Lazebnik-Ustimenko-Woldar (L-U-W) construction 
       which is guaranteed to be valid but potentially larger.
    
    This approach uses a combinatorial search over circulant structures, 
    which is distinct from the direct algebraic construction of the seed.
    """
    if k == 4 and g == 10:
        # Known optimal is 320. Lower bound is 243.
        # We search for a circulant graph in this range.
        # A 4-regular circulant graph is defined by shifts S = {a, -a, b, -b} mod n.
        # We need to ensure simplicity (a != b, a != 0, etc.) and girth >= 10.
        
        start_time = time.time()
        time_limit = 100  # Leave some time for fallback
        
        # Try n from 244 to 320 (even numbers)
        for n in range(244, 321, 2):
            if time.time() - start_time > time_limit:
                break
            # Search for shifts a, b
            # a and b must be distinct and not 0, n/2 (if n even, n/2 is self-loop in undirected? 
            # No, shift n/2 connects i to i+n/2. It's an edge. 
            # We need 4 distinct neighbors.
            # Neighbors of 0: a, n-a, b, n-b.
            # Must be distinct: a != b, a != n-b, n-a != b, n-a != n-b.
            # i.e., a != b, a + b != n, a != n/2, b != n/2.
            
            found = False
            for a in range(1, n // 2):
                if time.time() - start_time > time_limit:
                    break
                for b in range(a + 1, n // 2):
                    # Check distinctness
                    if a == b or a + b == n:
                        continue
                    if a == n // 2 or b == n // 2:
                        continue
                    
                    # Check girth
                    # BFS from 0
                    # State: (vertex, parent) to avoid backtracking
                    # We want to find if any cycle of length < g exists.
                    # A cycle of length L corresponds to a path of length L from 0 to 0 
                    # where the first and last edges are not the same (in terms of direction).
                    
                    # Simulate non-backtracking walks
                    # visited: dict of (vertex, prev_vertex) -> distance
                    # Actually, for girth, we just need to see if we can return to 0 
                    # with length < g without immediately backtracking.
                    
                    # Let's do a BFS on the state space (v, p)
                    # Start states: (a, 0), (n-a, 0), (b, 0), (n-b, 0) at dist 1.
                    # Target: reach (0, u) where u is a neighbor of 0, and dist >= 2.
                    # If we reach (0, u) at dist L, we have a cycle of length L.
                    # We need min L >= g.
                    
                    from collections import deque
                    q = deque()
                    # (current_vertex, previous_vertex, distance)
                    # Start from neighbors of 0
                    neighbors_0 = [a, n - a, b, n - b]
                    # Ensure neighbors are distinct
                    if len(set(neighbors_0)) < 4:
                        continue
                    
                    for nb in neighbors_0:
                        q.append((nb, 0, 1))
                    
                    visited = set()
                    min_cycle = float('inf')
                    
                    while q:
                        v, p, dist = q.popleft()
                        if (v, p) in visited:
                            continue
                        visited.add((v, p))
                        
                        # If we are at 0, we found a cycle
                        if v == 0:
                            # The cycle length is dist + 1 (the edge back to the start neighbor? 
                            # No, the path is 0 -> ... -> p -> 0.
                            # The length is dist + 1?
                            # Let's trace:
                            # dist=1: v=nb, p=0.
                            # If v=0, dist must be >= 2.
                            # Path: 0 -> v1 -> ... -> v_{dist} = 0.
                            # The edge is (v_{dist-1}, 0).
                            # The cycle is 0-v1-...-v_{dist-1}-0.
                            # Length is dist.
                            if dist < min_cycle:
                                min_cycle = dist
                            # We can stop exploring this branch if we just want min cycle
                            # But we need to check if ANY cycle < g exists.
                            # If we find one < g, this (a,b) is invalid.
                            if min_cycle < g:
                                break
                            continue
                        
                        # Expand to neighbors of v, excluding p
                        for nb in [v + a, v - a, v + b, v - b]:
                            nb = nb % n
                            if nb == p:
                                continue
                            if nb == 0:
                                # Found a cycle of length dist + 1
                                if dist + 1 < min_cycle:
                                    min_cycle = dist + 1
                                    if min_cycle < g:
                                        break
                                # Do not add (0, v) to queue if we are just checking for existence?
                                # Actually, we can stop if we found a short one.
                            else:
                                if (nb, v) not in visited:
                                    q.append((nb, v, dist + 1))
                    
                    if min_cycle >= g:
                        # Valid circulant found
                        edges = []
                        for i in range(n):
                            for s in [a, b]:
                                j = (i + s) % n
                                if i < j:
                                    edges.append((i, j))
                                j = (i - s) % n
                                if i < j:
                                    edges.append((i, j))
                        # Ensure no duplicates
                        edges = list(set(edges))
                        return n, edges

        # If no circulant found, fall back to L-U-W
        pass

    # Fallback: L-U-W Construction
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
# EVOLVE-BLOCK-END
