"""Voltage graph (circulant) construction for 5-regular graphs of girth >= 9."""
import numpy as np
from itertools import combinations, permutations, product

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g using a circulant/voltage graph approach.
    
    For the specific case of k=5, g=9, we use a known construction.
    For general cases, we attempt to find a suitable connection set.
    """
    if k == 5 and g == 9:
        return _construct_5_9()
    
    # Fallback: use Lazebnik-Ustimenko-Woldar style construction (same as seed)
    return _construct_luw(k, g)


def _construct_luw(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction (fallback from seed)."""
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


def _construct_5_9():
    """
    Construct a 5-regular graph of girth >= 9.
    
    We use a voltage graph approach: a circulant graph on Z_n with connection set S of size 5.
    The key constraint is that no non-trivial product of connection elements (corresponding to
    closed walks of length < 9) should equal the identity.
    
    For a circulant graph C(n, S), the girth is determined by the shortest non-trivial relation
    among elements of S ∪ S^{-1}.
    
    We search for a suitable n and connection set S = {s1, s2, s3, s4, s5} where:
    - All si are distinct non-zero elements of Z_n
    - si ≠ sj^{-1} for all i,j (to avoid 2-cycles and loops)
    - No product of fewer than 9 elements from S ∪ S^{-1} equals 0 in Z_n
    
    For n=1116 (known best), we can use a specific construction.
    However, finding such a set is hard. Let's use a known small example.
    
    Actually, let's try to construct using the known (5,9)-cage candidates.
    The smallest known has 1116 vertices. Let's try to build it.
    
    A simpler approach: use a known 5-regular graph of girth 9.
    The (5,9)-cage is not fully known, but there are constructions with n=1116.
    
    Let's try a different approach: use a bipartite double cover or other known construction.
    
    For now, let's use a heuristic search for a circulant graph.
    """
    # Try to find a circulant graph C(n, S) with girth >= 9
    # n must be at least the Moore bound: 1 + 5 + 5*4 + 5*4^2 + 5*4^3 = 1 + 5 + 20 + 80 + 320 = 426
    # For odd girth, the Moore bound is different.
    # For g=9 (odd), Moore bound: 1 + 5*(1 + 4 + 4^2 + 4^3) = 1 + 5*(1+4+16+64) = 1 + 5*85 = 426
    
    # Let's search for n and S
    # We need n such that there exist 5 distinct non-zero elements s1..s5 in Z_n
    # with no short relations.
    
    # This is a hard search problem. Let's use a known result.
    # The (5,9)-cage has n=1116 vertices.
    
    # Let's try to construct using a known method.
    # One approach: use the incidence graph of a design or a known algebraic construction.
    
    # For now, let's return a valid but possibly non-optimal graph.
    # We can use the LUW construction which gives a valid graph.
    
    # Actually, let's try a simple circulant construction.
    # For C(n, S) to have girth >= 9, we need that the shortest cycle is at least 9.
    # This means no product of < 9 elements from S ∪ S^{-1} equals 0.
    
    # Let's try n = 1116 and find a suitable S.
    n = 1116
    
    # We need to find S = {s1, s2, s3, s4, s5} such that:
    # 1. All si are distinct non-zero
    # 2. si ≠ -sj for all i,j (no 2-cycles)
    # 3. No product of < 9 elements from S ∪ S^{-1} equals 0
    
    # This is a constraint satisfaction problem. Let's try a greedy approach.
    
    # For simplicity, let's use a known construction if available, or fall back.
    
    # Let's try to use a random search with verification.
    import random
    
    def check_girth(n, S, max_girth=8):
        """Check if C(n, S) has girth > max_girth using BFS on the Cayley graph."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        for i in range(n):
            for s in S:
                j = (i + s) % n
                adj[i].add(j)
                adj[j].add(i)
        
        # BFS from each vertex to find shortest cycle
        # For efficiency, only check from a few vertices
        for start in range(min(n, 20)):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            for node in queue:
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                    elif nb != parent[node]:
                        # Found a cycle
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len <= max_girth:
                            return False
        return True
    
    # Search for S
    random.seed(42)
    for attempt in range(100):
        # Pick 5 random distinct non-zero elements
        candidates = list(range(1, n))
        random.shuffle(candidates)
        S = candidates[:5]
        
        # Check no 2-cycles: si != -sj
        valid = True
        for i in range(5):
            for j in range(5):
                if (S[i] + S[j]) % n == 0 and i != j:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        
        # Check girth
        if check_girth(n, S):
            # Found a valid S
            edges = []
            for i in range(n):
                for s in S:
                    j = (i + s) % n
                    if i < j:
                        edges.append((i, j))
            return n, edges
    
    # Fallback: use LUW construction
    return _construct_luw(5, 9)
