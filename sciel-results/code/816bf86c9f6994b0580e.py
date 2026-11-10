import time
import itertools

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # If k=4 and g=10, try to find the 320-vertex cage
    if k == 4 and g == 10:
        # Try known or candidate circulant constructions
        candidates = [
            [0, 1, 11, 100],
            [0, 1, 16, 100],
            [0, 1, 21, 100],
            [0, 1, 26, 100],
            [0, 1, 31, 100],
            [0, 1, 36, 100],
            [0, 1, 41, 100],
            [0, 1, 46, 100],
            [0, 1, 51, 100],
            [0, 1, 56, 100],
            [0, 1, 61, 100],
            [0, 1, 66, 100],
            [0, 1, 71, 100],
            [0, 1, 76, 100],
            [0, 1, 81, 100],
            [0, 1, 86, 100],
            [0, 1, 91, 100],
            [0, 1, 96, 100],
        ]
        
        N = 160
        for S in candidates:
            if time.time() > deadline:
                break
            if check_circulant_girth(N, S, g):
                # Construct edges
                edges = []
                for i in range(N):
                    for s in S:
                        j = (i + s) % N
                        if i < j: # Bipartite: i in 0..N-1, j in N..2N-1
                            # We need to map to 0..2N-1
                            # Let's say part A is 0..N-1, part B is N..2N-1
                            # Edge between i and N+j
                            edges.append((i, N + j))
                return 2 * N, edges
        
        # If no candidate worked, fall back to LUW
    return construct_luw(k, g)

def check_circulant_girth(N, S, g):
    # Check if the bipartite circulant graph with parts Z_N and Z_N
    # and connection set S has girth at least g.
    # Girth g means no cycles of length < g.
    # Since it's bipartite, we only need to check even lengths: 4, 6, ..., g-2.
    
    # 1. Check for 4-cycles:
    # A 4-cycle exists if there are s1, s2, s3, s4 in S such that
    # s1 - s2 + s3 - s4 = 0 => s1 + s3 = s2 + s4.
    # And the vertices must be distinct.
    # Vertices: A0, B_s1, A_s1-s2, B_s1-s2+s3, A0.
    # Distinctness:
    # B_s1 != B_s1-s2+s3 => s1 != s1-s2+s3 => s2 != s3.
    # A0 != A_s1-s2 => s1 != s2.
    # Also B_s1 != B_{s1-s2+s3} is covered.
    # And A_s1-s2 != A0 is covered.
    # Also need A0 != A_s1-s2+s3-s4 (which is A0).
    # And B_s1 != B_{s1-s2+s3} is s2 != s3.
    # What if s1=s3 and s2=s4? Then 2s1 = 2s2 => s1=s2 (mod N/2).
    # If s1=s2, backtracking.
    
    # Efficient check for 4-cycles:
    # Count pairs (s1, s3) such that s1+s3 = s2+s4.
    # If any sum appears more than once with distinct pairs, we have a 4-cycle candidate.
    # We need to ensure the cycle is simple.
    
    sums = {}
    for s1 in S:
        for s3 in S:
            key = (s1 + s3) % N
            if key not in sums:
                sums[key] = []
            sums[key].append((s1, s3))
            
    for key, pairs in sums.items():
        if len(pairs) > 1:
            # Check if any two pairs form a simple 4-cycle
            for i in range(len(pairs)):
                for j in range(i+1, len(pairs)):
                    s1, s3 = pairs[i]
                    s2, s4 = pairs[j]
                    # Check distinctness conditions
                    # s1 != s2 (A0 != A_s1-s2)
                    # s2 != s3 (B_s1 != B_s1-s2+s3)
                    # s3 != s4 (A_s1-s2 != A_s1-s2+s3-s4 = A0? No, A_s1-s2+s3-s4 = A0. So A_s1-s2 != A0 => s1!=s2. And A_s1-s2 != A0 is same.)
                    # Wait, the vertices are:
                    # A0
                    # B_{s1}
                    # A_{s1-s2}
                    # B_{s1-s2+s3}
                    # A0
                    # Conditions for simple cycle:
                    # 1. s1 != 0? No, s1 is offset.
                    # 2. A0 != A_{s1-s2} => s1 != s2.
                    # 3. B_{s1} != B_{s1-s2+s3} => s1 != s1-s2+s3 => s2 != s3.
                    # 4. A0 != B_{s1}? Different parts.
                    # 5. A0 != B_{s1-s2+s3}? Different parts.
                    # 6. A_{s1-s2} != B_{s1}? Different parts.
                    # 7. A_{s1-s2} != B_{s1-s2+s3}? Different parts.
                    # So we just need s1 != s2 and s2 != s3.
                    # Also, the edge (A0, B_s1) and (A0, B_{s1-s2+s3}) must be distinct?
                    # If s1 = s1-s2+s3, then B_s1 = B_{s1-s2+s3}. This is covered by s2=s3.
                    # If s1 = s1-s2+s3, then s2=s3.
                    # What if s1 = s1-s2+s3 is false, but the edges are the same?
                    # Edge is defined by offset.
                    # So if s1 != s2 and s2 != s3, it's a simple 4-cycle?
                    # We also need the pair (s1, s3) to be distinct from (s2, s4) in a way that doesn't reuse edges?
                    # If (s1, s3) = (s2, s4), then s1=s2 and s3=s4.
                    # Then s1=s2 fails condition 2.
                    # So if s1!=s2 and s2!=s3, it's a valid 4-cycle.
                    
                    if s1 != s2 and s2 != s3:
                        return False
    # If we reach here, no 4-cycles.
    
    # Check for 6-cycles:
    # s1 - s2 + s3 - s4 + s5 - s6 = 0
    # s1 + s3 + s5 = s2 + s4 + s6
    # We need to check if any sum of 3 elements from S equals another sum of 3 elements from S.
    # And the resulting walk is a simple 6-cycle.
    
    sums3 = {}
    for s1 in S:
        for s3 in S:
            for s5 in S:
                key = (s1 + s3 + s5) % N
                if key not in sums3:
                    sums3[key] = []
                sums3[key].append((s1, s3, s5))
                
    for key, triplets in sums3.items():
        if len(triplets) > 1:
            for i in range(len(triplets)):
                for j in range(i+1, len(triplets)):
                    s1, s3, s5 = triplets[i]
                    s2, s4, s6 = triplets[j]
                    # Check simple cycle conditions
                    # Vertices: A0, B_s1, A_s1-s2, B_s1-s2+s3, A_s1-s2+s3-s4, B_s1-s2+s3-s4+s5, A0
                    # Non-backtracking:
                    # s1 != s2
                    # s2 != s3
                    # s3 != s4
                    # s4 != s5
                    # s5 != s6
                    # s6 != s1 (to avoid closing with backtrack? No, s6 is last edge to A0. A0 is start. Edge is B_{...} to A0. Offset is -s6. Previous edge was to B_{...} with offset s5. So we need s5 != -(-s6)? No.
                    # The edge into A0 is from B_{s1-s2+s3-s4+s5} with offset -s6.
                    # The edge out of A0 was to B_{s1} with offset s1.
                    # These are different edges unless s1 = -s6? No, they are different edges in the graph.
                    # But for the cycle to be simple, we need all 6 vertices distinct.
                    # A0 != A_{s1-s2} => s1 != s2
                    # A0 != A_{s1-s2+s3-s4} => s1-s2 != s3-s4 => s1+s4 != s2+s3
                    # A0 != B_{s1} (diff parts)
                    # A0 != B_{s1-s2+s3} (diff parts)
                    # A0 != B_{s1-s2+s3-s4+s5} (diff parts)
                    # B_{s1} != B_{s1-s2+s3} => s1 != s1-s2+s3 => s2 != s3
                    # B_{s1} != B_{s1-s2+s3-s4+s5} => s1 != s1-s2+s3-s4+s5 => s2-s3+s4-s5 != 0
                    # A_{s1-s2} != A_{s1-s2+s3-s4} => s3 != s4
                    # A_{s1-s2} != B_{s1} (diff parts)
                    # A_{s1-s2} != B_{s1-s2+s3} (diff parts)
                    # A_{s1-s2} != B_{s1-s2+s3-s4+s5} (diff parts)
                    # B_{s1-s2+s3} != B_{s1-s2+s3-s4+s5} => s3-s4 != s3-s4+s5 => s5 != 0? No.
                    # s3-s4 != s3-s4+s5 => s5 != 0.
                    # Wait, B_{x} != B_{y} iff x != y.
                    # So we need all B vertices distinct and all A vertices distinct.
                    
                    if (s1 != s2 and s2 != s3 and s3 != s4 and s4 != s5 and s5 != s6 and s6 != s1 and
                        (s1 + s4 - s2 - s3) % N != 0 and
                        (s2 - s3 + s4 - s5) % N != 0 and
                        (s1 - s2 + s3 - s4 + s5) % N != 0): # Last one is B_s1 != B_end?
                        # B_s1 index: s1
                        # B_end index: s1-s2+s3-s4+s5
                        # Distinct if s1 != s1-s2+s3-s4+s5 => s2-s3+s4-s5 != 0.
                        # This is covered by (s2 - s3 + s4 - s5) % N != 0.
                        return False
    return True

def construct_luw(k, g):
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
