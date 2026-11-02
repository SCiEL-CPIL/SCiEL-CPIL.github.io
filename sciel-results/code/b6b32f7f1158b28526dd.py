import random
import time
from collections import deque

def _find_cage_generators(k, g, max_time=100):
    """
    Search for a set of k generators in Z_n such that the Cayley graph
    has girth at least g. Returns (n, generators) or None.
    """
    start_time = time.time()
    # Moore bound for k=5, g=9 is 428.
    # Known smallest is 1116.
    # We search n from 430 to 1500.
    for n in range(430, 1501):
        if time.time() - start_time > max_time:
            break
        # Try several random generator sets
        for _ in range(20):
            # Pick k generators, ensure they are distinct, not 0, and symmetric
            # We pick k/2 pairs if k is even, or (k-1)/2 pairs and one self-inverse?
            # For k=5, we need 5 generators.
            # In a Cayley graph on an abelian group, if x is a generator, -x is also.
            # So we pick 2 pairs {a, -a}, {b, -b} and one element c such that c = -c (order 2).
            # Or we can just pick 5 random elements and ensure the set is symmetric.
            # Easier: Pick 2 random non-zero elements a, b.
            # Generators: a, -a, b, -b, c.
            # c must be order 2 (2c = 0 mod n). So c = n/2 if n is even.
            # If n is odd, there is no element of order 2.
            # So for k=5 (odd), n must be even?
            # If n is odd, we can't have a symmetric set of size 5?
            # Wait, the set of generators S must be symmetric: x in S => -x in S.
            # If k is odd, the size of S is odd.
            # This implies there is at least one element x such that x = -x, i.e., 2x = 0.
            # So n must be even.
            if n % 2 != 0:
                continue
            
            c = n // 2
            # Pick a, b from 1 to n-1, a != b, a != c, b != c, a != -a, b != -b
            # a != -a => 2a != 0 => a != n/2 (already ensured)
            # a != b, a != -b
            while True:
                a = random.randint(1, n - 1)
                b = random.randint(1, n - 1)
                if a == c or b == c:
                    continue
                if a == b or a == n - b:
                    continue
                # Check if {a, n-a, b, n-b, c} are distinct
                s = {a, n - a, b, n - b, c}
                if len(s) == 5:
                    break
            
            gens = [a, n - a, b, n - b, c]
            
            # Check girth
            if _check_girth(n, gens, g):
                return n, gens
    
    return None

def _check_girth(n, gens, g):
    """
    Check if the Cayley graph of Z_n with generators gens has girth >= g.
    Uses non-backtracking BFS.
    """
    # Adjacency list
    adj = [[] for _ in range(n)]
    for u in range(n):
        for v in gens:
            w = (u + v) % n
            adj[u].append(w)
    
    # Non-backtracking BFS from every vertex
    # To speed up, we can just check from a few vertices? No, must be all.
    # But n is small (< 1500).
    # We can use a global visited array with timestamps to avoid clearing.
    # State: (current, prev)
    # We want to find the shortest cycle.
    # A cycle of length L corresponds to a non-backtracking walk of length L that returns to start.
    
    # Optimization: We only need to check up to depth g-1.
    # If we don't find a cycle back to start in < g steps, girth >= g.
    
    # We can do BFS from each vertex s.
    # Queue: (node, prev, dist)
    # Visited: set of (node, prev) to avoid loops in BFS?
    # Actually, for girth, we just need to know if we can return to s.
    # Standard non-backtracking BFS:
    # From s, start with neighbors.
    # If a neighbor is s, cycle of length 2 (not possible in simple graph with no loops/multi-edges? 
    # Wait, if a=0, loop. We excluded 0.
    # If a = -a, i.e., 2a=0, then a is order 2.
    # Walk s -> s+a -> s. Length 2.
    # But simple graph has no multi-edges.
    # If a is a generator, edge (s, s+a).
    # If -a is also a generator, edge (s+a, s).
    # This is the same edge.
    # So a walk s -> s+a -> s is a backtracking walk.
    # Non-backtracking forbids this.
    # So we don't count length 2.
    
    for s in range(n):
        # BFS
        # Queue stores (node, prev_node)
        # We track distance implicitly by levels.
        q = deque()
        for nb in adj[s]:
            q.append((nb, s))
        
        dist = 1
        while q and dist < g:
            new_q = deque()
            for node, prev in q:
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == s:
                        # Found a cycle of length dist+1
                        # Wait, if nb == s, the path is s -> ... -> node -> s.
                        # Length is dist + 1.
                        # We require girth >= g, so cycle length >= g.
                        # If dist + 1 < g, return False.
                        if dist + 1 < g:
                            return False
                    else:
                        new_q.append((nb, node))
            q = new_q
            dist += 1
            
    return True

def construct(k, g):
    # If k=5 and g=9, try to find a small cage.
    if k == 5 and g == 9:
        res = _find_cage_generators(5, 9, max_time=110)
        if res:
            n, gens = res
            edges = []
            for u in range(n):
                for v in gens:
                    w = (u + v) % n
                    if u < w:
                        edges.append((u, w))
            return n, edges
    
    # Fallback: CD graph (same as seed)
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
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
