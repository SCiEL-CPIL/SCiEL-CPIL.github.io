import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    if k == 4 and g == 10:
        # Try to find the (4,10)-cage as a bipartite circulant on 160+160
        # The known (4,10)-cage is a circulant graph
        # Let's search for connection sets that give girth 10
        
        N = 160
        # Try various connection sets
        # The (4,10)-cage is known to have a specific structure
        # Let's try the known construction
        
        # First, try some known candidate sets
        candidates = [
            [1, 2, 50, 100],
            [1, 3, 50, 100],
            [1, 5, 50, 100],
            [1, 7, 50, 100],
            [1, 11, 50, 100],
            [1, 13, 50, 100],
            [1, 17, 50, 100],
            [1, 19, 50, 100],
            [1, 23, 50, 100],
            [1, 29, 50, 100],
            [1, 31, 50, 100],
            [1, 37, 50, 100],
            [1, 41, 50, 100],
            [1, 43, 50, 100],
            [1, 47, 50, 100],
            [1, 49, 50, 100],
            [1, 51, 50, 100],
            [1, 53, 50, 100],
            [1, 57, 50, 100],
            [1, 59, 50, 100],
            [1, 61, 50, 100],
            [1, 63, 50, 100],
            [1, 67, 50, 100],
            [1, 69, 50, 100],
            [1, 71, 50, 100],
            [1, 73, 50, 100],
            [1, 77, 50, 100],
            [1, 79, 50, 100],
            [1, 81, 50, 100],
            [1, 83, 50, 100],
            [1, 87, 50, 100],
            [1, 89, 50, 100],
            [1, 91, 50, 100],
            [1, 93, 50, 100],
            [1, 97, 50, 100],
            [1, 99, 50, 100],
            [1, 101, 50, 100],
            [1, 103, 50, 100],
            [1, 107, 50, 100],
            [1, 109, 50, 100],
            [1, 111, 50, 100],
            [1, 113, 50, 100],
            [1, 117, 50, 100],
            [1, 119, 50, 100],
            [1, 121, 50, 100],
            [1, 123, 50, 100],
            [1, 127, 50, 100],
            [1, 129, 50, 100],
            [1, 131, 50, 100],
            [1, 133, 50, 100],
            [1, 137, 50, 100],
            [1, 139, 50, 100],
            [1, 141, 50, 100],
            [1, 143, 50, 100],
            [1, 147, 50, 100],
            [1, 149, 50, 100],
            [1, 151, 50, 100],
            [1, 153, 50, 100],
            [1, 157, 50, 100],
        ]
        
        for S in candidates:
            if time.time() > deadline:
                break
            if _check_girth_circulant(N, S, g):
                return _build_circulant_edges(N, S)
        
        # If not found, try a broader search
        # The (4,10)-cage might not be a simple bipartite circulant
        # Let's use the LUW construction as fallback
        return _construct_luw(k, g)
    
    return _construct_luw(k, g)

def _check_girth_circulant(N, S, g):
    """Check if bipartite circulant with connection set S has girth >= g."""
    # Check for cycles of length < g using BFS from each vertex
    # But for circulant, we can use the group structure
    
    # For a bipartite circulant, cycles correspond to relations in the connection set
    # A 2m-cycle exists if s1 - s2 + s3 - ... + s_{2m-1} - s_{2m} = 0 (mod N)
    # with the walk being simple
    
    # Check for 4-cycles
    # s1 + s3 = s2 + s4 (mod N), with s1 != s2, s2 != s3
    sums = {}
    for s1 in S:
        for s3 in S:
            key = (s1 + s3) % N
            if key not in sums:
                sums[key] = []
            sums[key].append((s1, s3))
    
    for key, pairs in sums.items():
        if len(pairs) > 1:
            for i in range(len(pairs)):
                for j in range(i + 1, len(pairs)):
                    s1, s3 = pairs[i]
                    s2, s4 = pairs[j]
                    # Check distinctness for simple 4-cycle
                    if s1 != s2 and s2 != s3 and s3 != s4:
                        # Also need the edges to be distinct
                        # (s1, s3) != (s2, s4) as unordered pairs? 
                        # If (s1,s3) = (s2,s4), then s1=s2 or s1=s4
                        # If s1=s2, fails above
                        # If s1=s4, then s3=s2, fails s2!=s3
                        return False
    
    # Check for 6-cycles
    # s1 + s3 + s5 = s2 + s4 + s6 (mod N)
    sums3 = {}
    for s1 in S:
        for s3 in S:
            for s5 in S:
                key = (s1 + s3 + s5) % N
                if key not in sums3:
                    sums3[key] = []
                sums3[key].append((s1, s3, s5))
    
    for key, triples in sums3.items():
        if len(triples) > 1:
            for i in range(len(triples)):
                for j in range(i + 1, len(triples)):
                    s1, s3, s5 = triples[i]
                    s2, s4, s6 = triples[j]
                    # Check non-backtracking and distinctness
                    if (s1 != s2 and s2 != s3 and s3 != s4 and s4 != s5 and s5 != s6 and
                        (s1 + s4 - s2 - s3) % N != 0 and  # A0 != A_mid
                        (s2 - s3 + s4 - s5) % N != 0 and  # B1 != B_end
                        (s1 - s2 + s3 - s4 + s5) % N != 0):  # B1 != B_end (redundant check)
                        return False
    
    # Check for 8-cycles
    # s1 + s3 + s5 + s7 = s2 + s4 + s6 + s8 (mod N)
    sums4 = {}
    for s1 in S:
        for s3 in S:
            for s5 in S:
                for s7 in S:
                    key = (s1 + s3 + s5 + s7) % N
                    if key not in sums4:
                        sums4[key] = []
                    sums4[key].append((s1, s3, s5, s7))
    
    for key, quads in sums4.items():
        if len(quads) > 1:
            for i in range(len(quads)):
                for j in range(i + 1, len(quads)):
                    s1, s3, s5, s7 = quads[i]
                    s2, s4, s6, s8 = quads[j]
                    # Non-backtracking
                    if (s1 != s2 and s2 != s3 and s3 != s4 and s4 != s5 and
                        s5 != s6 and s6 != s7 and s7 != s8 and s8 != s1):
                        # Check distinctness of A vertices:
                        # A0, A_{s1-s2}, A_{s1-s2+s3-s4}, A_{s1-s2+s3-s4+s5-s6}
                        # All distinct
                        a1 = (s1 - s2) % N
                        a2 = (s1 - s2 + s3 - s4) % N
                        a3 = (s1 - s2 + s3 - s4 + s5 - s6) % N
                        if a1 != 0 and a2 != 0 and a3 != 0 and a1 != a2 and a2 != a3 and a1 != a3:
                            # Check B vertices distinct:
                            # B_s1, B_{s1-s2+s3}, B_{s1-s2+s3-s4+s5}
                            b1 = s1
                            b2 = (s1 - s2 + s3) % N
                            b3 = (s1 - s2 + s3 - s4 + s5) % N
                            if b1 != b2 and b2 != b3 and b1 != b3:
                                return False
    
    # Check for 10-cycles? No, we need girth >= 10, so no cycles of length < 10
    # We've checked 4, 6, 8. If none found, girth >= 10 (since bipartite, no odd cycles)
    return True

def _build_circulant_edges(N, S):
    """Build edge list for bipartite circulant with N vertices per part."""
    edges = []
    for i in range(N):
        for s in S:
            j = (i + s) % N
            # Part A: 0..N-1, Part B: N..2N-1
            a = i
            b = N + j
            if a < b:
                edges.append((a, b))
    return 2 * N, edges

def _construct_luw(k, g):
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
