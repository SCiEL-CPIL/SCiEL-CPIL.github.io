import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    # For k=4, g=10, try the 2-vertex voltage graph construction
    if k == 4 and g == 10:
        result = _voltage_lift_search(deadline)
        if result is not None:
            return result

    # Fallback to LUW if voltage search fails or times out
    result = _luw_construct(k, g)
    if result is not None:
        n, edges = result
        if _verify_girth(n, edges, g):
            return n, edges

    return _luw_construct(k, g)


def _check_voltages(v1, v2, v3, v4, M):
    """
    Check that 4 distinct voltages in Z_M satisfy:
    no non-backtracking closed walk of length 4, 6, or 8 has sum 0 mod M.
    
    Uses algebraic identities to avoid brute-force walk enumeration.
    Returns True if all constraints are satisfied.
    """
    vs = (v1, v2, v3, v4)
    vs2 = tuple((v * v) % M for v in vs)
    
    # L=4: non-backtracking walks of length 4.
    # After fixing i1=0, the non-backtracking constraint forces:
    #   Pattern A: 0,1,0,1 -> sum = v1^2 + v2^2
    #   Pattern B: 0,1,2,1 -> sum = v1^2 + v2^2 + v3^2
    #   Pattern C: 0,1,2,3 -> sum = v1^2 + v2^2 + v3^2 + v4^2  (also covers L=6 pattern)
    #   Pattern D: 0,1,2,0 -> sum = v1^2 + v2^2 + v3^2 + v1^2 = 2*v1^2 + v2^2 + v3^2
    # We need ALL these to be nonzero mod M.
    
    # L=4 constraints
    for i in range(4):
        for j in range(i + 1, 4):
            if (vs2[i] + vs2[j]) % M == 0:
                return False
            for k in range(4):
                if k == i or k == j:
                    if (vs2[i] + vs2[j] + vs2[k]) % M == 0:
                        return False
                    # Pattern D type: 2*vs2[i] + vs2[j] + vs2[k]
                    if (2 * vs2[i] + vs2[j] + vs2[k]) % M == 0:
                        return False
                    if (vs2[i] + 2 * vs2[j] + vs2[k]) % M == 0:
                        return False
                    if (vs2[i] + vs2[j] + 2 * vs2[k]) % M == 0:
                        return False
    
    # L=6: additional patterns
    # 0,1,2,3,2,1 -> sum = v1^2+v2^2+v3^2+v4^2+v3^2+v2^2 = v1^2+2*v2^2+2*v3^2+v4^2
    # 0,1,2,3,0,1 -> sum = v1^2+v2^2+v3^2+v4^2+v1^2+v2^2 = 2*v1^2+2*v2^2+v3^2+v4^2
    # 0,1,2,1,0,1 -> sum = v1^2+v2^2+v3^2+v2^2+v1^2+v2^2 = 2*v1^2+3*v2^2+v3^2
    # 0,1,0,1,2,1 -> sum = v1^2+v2^2+v1^2+v2^2+v3^2+v2^2 = 2*v1^2+3*v2^2+v3^2
    # 0,1,2,3,0,2 -> sum = v1^2+v2^2+v3^2+v4^2+v1^2+v3^2 = 2*v1^2+v2^2+2*v3^2+v4^2
    # 0,1,2,3,4,3 -> sum = v1^2+v2^2+v3^2+v4^2+v3^2+v4^2 = v1^2+v2^2+2*v3^2+2*v4^2
    # 0,1,2,0,1,2 -> sum = v1^2+v2^2+v3^2+v1^2+v2^2+v3^2 = 2*v1^2+2*v2^2+2*v3^2
    # 0,1,2,3,2,0 -> sum = v1^2+v2^2+v3^2+v4^2+v3^2+v1^2 = 2*v1^2+v2^2+2*v3^2+v4^2
    # 0,1,2,3,0,3 -> sum = v1^2+v2^2+v3^2+v4^2+v1^2+v4^2 = 2*v1^2+v2^2+v3^2+2*v4^2
    
    for i in range(4):
        for j in range(i + 1, 4):
            for k in range(j + 1, 4):
                # All 4 distinct: v1^2+v2^2+v3^2+v4^2
                if (vs2[i] + vs2[j] + vs2[k] + vs2[(i+j+k) % 4]) % M == 0:
                    return False
                # 2*v1^2+2*v2^2+v3^2+v4^2 type
                for a in range(4):
                    for b in range(4):
                        if a == b:
                            continue
                        # coefficient pattern: 2 on two, 1 on two
                        coeffs = [0, 0, 0, 0]
                        coeffs[a] = 2
                        coeffs[b] = 2
                        rem = [x for x in range(4) if x != a and x != b]
                        coeffs[rem[0]] = 1
                        coeffs[rem[1]] = 1
                        s = sum(coeffs[t] * vs2[t] for t in range(4))
                        if s % M == 0:
                            return False
                # 2*v1^2+3*v2^2+v3^2 type (one coeff=2, one=3, one=1, one=0)
                for a in range(4):
                    for b in range(4):
                        for c in range(4):
                            if a == b or a == c or b == c:
                                continue
                            d = 6 - a - b - c  # remaining index
                            if d < 0 or d > 3:
                                continue
                            coeffs = [0, 0, 0, 0]
                            coeffs[a] = 2
                            coeffs[b] = 3
                            coeffs[c] = 1
                            s = sum(coeffs[t] * vs2[t] for t in range(4))
                            if s % M == 0:
                                return False
    
    # L=8: non-backtracking walks of length 8.
    # Key patterns (after fixing i1=0):
    # 0,1,0,1,0,1,0,1 -> 4*v1^2 + 4*v2^2
    # 0,1,2,1,0,1,2,1 -> 2*v1^2 + 4*v2^2 + 2*v3^2
    # 0,1,2,3,0,1,2,3 -> 2*(v1^2+v2^2+v3^2+v4^2)
    # 0,1,2,3,4,3,2,1 -> 2*v1^2+2*v2^2+2*v3^2+2*v4^2 (same as above)
    # 0,1,2,3,2,1,0,1 -> 3*v1^2+4*v2^2+2*v3^2
    # 0,1,2,3,0,2,1,2 -> 2*v1^2+2*v2^2+4*v3^2
    # 0,1,2,3,2,3,0,2 -> v1^2+v2^2+5*v3^2+v4^2
    # 0,1,2,3,4,0,1,4 -> 3*v1^2+2*v2^2+v3^2+3*v4^2
    # 0,1,2,3,4,3,0,4 -> 2*v1^2+v2^2+3*v3^2+3*v4^2
    # 0,1,2,3,4,0,3,4 -> 2*v1^2+v2^2+v3^2+4*v4^2
    # 0,1,2,3,4,3,2,0 -> 3*v1^2+2*v2^2+3*v3^2+v4^2
    # 0,1,2,0,1,2,3,2 -> 3*v1^2+3*v2^2+3*v3^2
    # 0,1,2,3,0,1,3,2 -> 3*v1^2+2*v2^2+3*v3^2
    # 0,1,2,3,4,0,1,0 -> 5*v1^2+2*v2^2+v3^2
    # 0,1,2,3,4,3,4,0 -> 2*v1^2+v2^2+2*v3^2+4*v4^2
    # 0,1,2,3,4,3,4,2 -> v1^2+v2^2+3*v3^2+3*v4^2
    # 0,1,2,3,4,0,3,0 -> 3*v1^2+v2^2+3*v3^2
    # 0,1,2,3,4,0,2,0 -> 4*v1^2+v2^2+2*v3^2
    # 0,1,2,3,4,3,2,3 -> v1^2+2*v2^2+4*v3^2+v4^2
    # 0,1,2,3,4,3,0,3 -> 3*v1^2+v2^2+2*v3^2+2*v4^2
    # 0,1,2,3,4,0,1,2 -> 3*v1^2+3*v2^2+2*v3^2
    # 0,1,2,3,4,3,4,3 -> 2*v1^2+v2^2+2*v3^2+3*v4^2
    # 0,1,2,3,4,0,2,3 -> 3*v1^2+v2^2+2*v3^2+2*v4^2
    # 0,1,2,3,4,3,2,4 -> v1^2+2*v2^2+3*v3^2+2*v4^2
    # 0,1,2,3,4,0,3,2 -> 3*v1^2+v2^2+2*v3^2+2*v4^2
    # 0,1,2,3,4,3,0,2 -> 3*v1^2+v2^2+2*v3^2+v4^2  (wait, let me be more careful)
    
    # Rather than enumerate all by hand, let's use a smarter approach:
    # Generate all non-backtracking walks of length 8 starting from index 0,
    # but do it analytically. There are at most 3^7 = 2187 such walks.
    # For each walk, compute the sum of v_{i_t}^2 with signs, check mod M.
    # But we can do this with precomputed coefficient vectors.
    
    # Actually, let's just enumerate all non-backtracking walks of length 8
    # from vertex 0 and check the sum. There are at most 3^7 = 2187.
    # We precompute the coefficient pattern for each walk.
    
    # Precompute: for each non-backtracking walk (i1,...,i8) with i1=0,
    # the sum is sum_{t=1}^{8} (-1)^{t+1} * v_{i_t}
    # But for the voltage graph, the condition is that this sum is 0 mod M.
    # However, since we're using a 2-vertex base, the walk alternates between
    # the two "layers", so the voltage sum is actually:
    # v_{i1} - v_{i2} + v_{i3} - v_{i4} + v_{i5} - v_{i6} + v_{i7} - v_{i8}
    # where the i_t are indices into {v1,v2,v3,v4}.
    
    # For a non-backtracking walk, i_t != i_{t+1}.
    # The sum S = v_{i1} - v_{i2} + v_{i3} - v_{i4} + v_{i5} - v_{i6} + v_{i7} - v_{i8}
    # For this to be 0 mod M for all voltages, we need... but we're checking
    # for specific voltages, so we just compute S mod M.
    
    # Let's enumerate all non-backtracking walks of length 8 from index 0.
    # i1 = 0 (fixed)
    # i2 in {1,2,3}
    # i3 in {0,1,2,3} \ {i2}
    # etc.
    
    # For efficiency, precompute the coefficient vector (c1,c2,c3,c4) for each walk,
    # where c_j = sum of signs of positions t where i_t = j.
    
    # Then S = c1*v1 + c2*v2 + c3*v3 + c4*v4, and we check S % M != 0.
    
    # Let's generate all walks and their coefficient vectors.
    walks_8 = []
    def gen_walks(idx, pos, last, coeffs):
        if pos == 8:
            walks_8.append(tuple(coeffs))
            return
        for nxt in range(4):
            if nxt == last:
                continue
            new_coeffs = coeffs[:]
            sign = 1 if (pos + 1) % 2 == 0 else -1  # pos is 0-indexed, t = pos+1
            new_coeffs[nxt] += sign
            gen_walks(nxt, pos + 1, nxt, new_coeffs)
    
    # i1 = 0, sign = +1
    gen_walks(0, 1, 0, [1, 0, 0, 0])
    
    # Check each walk
    for coeffs in walks_8:
        s = (coeffs[0] * v1 + coeffs[1] * v2 + coeffs[2] * v3 + coeffs[3] * v4) % M
        if s == 0:
            return False
    
    return True


def _voltage_lift_search(deadline):
    """
    Search for smallest M and voltages v1..v4 in Z_M such that:
    1. v1, v2, v3, v4 are distinct.
    2. No non-backtracking closed walk of length < 10 has sum 0.
    Uses algebraic constraint checking for speed.
    """
    # Prioritize M=160 as it corresponds to the known 320-vertex cage
    m_candidates = [160] + [m for m in range(4, 300) if m != 160]
    
    for M in m_candidates:
        if time.time() > deadline - 10:
            break
            
        # Fix v1=0, v2=1. Search a, b in 2..M-1 with a < b.
        # To speed up, we can check constraints as we go.
        
        found = False
        for a in range(2, M):
            if found:
                break
            for b in range(a + 1, M):
                if time.time() > deadline - 5:
                    found = True  # break out
                    break
                    
                voltages = (0, 1, a, b)
                
                if _check_voltages(0, 1, a, b, M):
                    # Found valid M and voltages
                    n = 2 * M
                    edges = []
                    for x in range(M):
                        for v in voltages:
                            y = (x + v) % M
                            u = x
                            w = M + y
                            edges.append((u, w))
                    
                    # Verify
                    if _verify_girth(n, edges, 10):
                        return n, edges
                    else:
                        # Verification failed, try next
                        continue
        
    return None


def _verify_girth(n, edges, g):
    """Verify graph is 4-regular and has girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != 4:
            return False
            
    edge_set = set()
    for u, v in edges:
        if (u, v) in edge_set or (v, u) in edge_set:
            return False
        edge_set.add((u, v))
        
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] >= g:
                        break
                    queue.append(v)
                elif dist[v] != dist[u] + 1:
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True


def _luw_construct(k, g):
    """
    LUW construction fallback.
    """
    q = k
    d = (g + 1) // 2
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {}
    for c in range(2, d + 1):
        if c == 2:
            rule[c] = (1, 1)
        elif c == 3:
            rule[c] = (2, 1)
        elif c == 4:
            rule[c] = (1, 2)
        else:
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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
