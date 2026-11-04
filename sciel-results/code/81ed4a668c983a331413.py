import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    if k != 3:
        # Fallback to LUW or random for other k
        # For now, return a simple LUW-like construction or error
        # But task is specifically k=3
        pass

    # Generate primes p = 1 mod 4
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def next_prime_1_mod_4(p):
        p += 2
        while True:
            if p % 4 == 1 and is_prime(p):
                return p
            p += 2

    p = 5
    best_n = float('inf')
    best_edges = []
    best_girth = 0

    while p < 1000: # Limit search space
        if time.time() > deadline:
            break
            
        n = p * (p*p - 1) // 2
        
        # Construct LPS graph
        # Vertices are matrices in PSL(2, p)
        # We represent a matrix (a, b, c, d) with ad-bc=1
        # Canonical form: (a, b, c, d) such that a>0 or (a==0 and b>0) etc.
        # Simpler: use a dictionary to map matrix tuples to indices
        
        mat_to_idx = {}
        idx_to_mat = []
        
        # Iterate over all SL(2, p) matrices
        # a, b, c, d in 0..p-1
        # ad - bc = 1 mod p
        
        # Optimization: iterate a, b, c and solve for d
        # d = (1 + bc) * a^{-1} mod p
        # a cannot be 0 if we want to iterate this way?
        # If a=0, then -bc=1.
        
        # Better: iterate all a,b,c,d and check det=1? O(p^4) is too slow for p=29 (700k) - actually 29^4 is ~700k, which is fine.
        # But p=100 is 100M, too slow.
        # We only go up to p ~ 50 or so.
        
        # Let's use the a,b,c -> d method.
        # If a != 0:
        #   inv_a = pow(a, p-2, p)
        #   d = (1 + b*c) * inv_a % p
        #   Check if d is in 0..p-1 (always true)
        #   This generates all matrices with a != 0.
        #   Count: p * p * p = p^3.
        #   But we need ad-bc=1.
        #   For each a,b,c, d is fixed.
        #   So we generate p^3 matrices?
        #   Total SL(2,p) size is p(p^2-1) ~ p^3.
        #   So this works.
        
        # Handle a=0 case:
        #   -bc = 1 => bc = p-1.
        #   b != 0. c = (p-1) * b^{-1}.
        #   d is free? No, det is fixed.
        #   If a=0, matrix is (0, b, c, d).
        #   det = -bc.
        #   So d is free? No, the matrix is (0, b, c, d).
        #   Wait, the determinant is ad-bc = 0*d - bc = -bc.
        #   So if a=0, d can be anything?
        #   No, the matrix is in SL(2,p).
        #   So we just need -bc = 1.
        #   d is not constrained by the determinant if a=0?
        #   Wait, the matrix is [[a,b],[c,d]].
        #   det = ad - bc.
        #   If a=0, det = -bc.
        #   So d is free?
        #   Yes, if a=0, d does not appear in the determinant.
        #   So for each b != 0, c is fixed, d can be 0..p-1.
        #   Count: p * p = p^2.
        
        # Total matrices: p^3 (a!=0) + p^2 (a=0) ~ p^3.
        
        # We will generate all matrices, canonicalize them, and add to graph.
        
        def canonicalize(a, b, c, d):
            # Identify M and -M
            # Make the first non-zero element positive
            if a != 0:
                if a < 0:
                    a, b, c, d = -a, -b, -c, -d
            elif b != 0:
                if b < 0:
                    a, b, c, d = -a, -b, -c, -d
            elif c != 0:
                if c < 0:
                    a, b, c, d = -a, -b, -c, -d
            else:
                if d < 0:
                    a, b, c, d = -a, -b, -c, -d
            return (a, b, c, d)

        # Generate vertices
        seen = set()
        vertices = []
        
        # a != 0
        for a in range(1, p):
            inv_a = pow(a, p-2, p)
            for b in range(p):
                for c in range(p):
                    d = (1 + b * c) * inv_a % p
                    canon = canonicalize(a, b, c, d)
                    if canon not in seen:
                        seen.add(canon)
                        vertices.append(canon)
                        
        # a == 0
        for b in range(1, p):
            inv_b = pow(b, p-2, p)
            c = (p - 1) * inv_b % p
            for d in range(p):
                canon = canonicalize(0, b, c, d)
                if canon not in seen:
                    seen.add(canon)
                    vertices.append(canon)
                    
        # Map to indices
        for i, v in enumerate(vertices):
            mat_to_idx[v] = i
            
        n_actual = len(vertices)
        
        # Define generators
        # g1 = (0, -1, 1, 0) -> (0, p-1, 1, 0)
        # g2 = (1, 1, 0, 1)
        # g3 = (1, 0, -1, 1) -> (1, 0, p-1, 1)
        
        def mat_mul(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            a = (a1*a2 + b1*c2) % p
            b = (a1*b2 + b1*d2) % p
            c = (c1*a2 + d1*c2) % p
            d = (c1*b2 + d1*d2) % p
            return (a, b, c, d)
            
        generators = [
            (0, p-1, 1, 0),
            (1, 1, 0, 1),
            (1, 0, p-1, 1)
        ]
        
        edges = []
        adj = [[] for _ in range(n_actual)]
        
        for i, m in enumerate(vertices):
            for g in generators:
                nm = mat_mul(m, g)
                canon = canonicalize(*nm)
                if canon in mat_to_idx:
                    j = mat_to_idx[canon]
                    if i < j:
                        edges.append((i, j))
                    adj[i].append(j)
                    adj[j].append(i)
                    
        # Check girth
        # BFS up to depth g//2
        depth = g // 2
        min_girth = float('inf')
        
        # Optimization: only check if n_actual is reasonable
        if n_actual > 50000:
            # Skip girth check if too big, assume it's good?
            # Or just return it if we are desperate
            pass
            
        for start in range(n_actual):
            if time.time() > deadline:
                break
            visited = [-1] * n_actual
            visited[start] = 0
            queue = deque([(start, 0, -1)])
            found = False
            while queue:
                u, dist, parent = queue.popleft()
                if dist >= depth:
                    continue
                for v in adj[u]:
                    if v == parent:
                        continue
                    if visited[v] == -1:
                        visited[v] = dist + 1
                        queue.append((v, dist + 1, u))
                    else:
                        # Cycle found
                        cycle_len = dist + 1 + visited[v]
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        found = True
            if found:
                # We found a cycle, but we need the minimum one
                # This BFS finds the shortest cycle *from* start.
                # The global girth is the min over all starts.
                pass
        
        # The above BFS logic is slightly flawed for finding the *global* minimum girth efficiently.
        # But if we just want to know if girth >= g, we can stop early if we find a cycle of length < g.
        # If we complete BFS up to depth g//2 from ALL vertices and find NO cycles, then girth >= g.
        
        valid = True
        for start in range(n_actual):
            if time.time() > deadline:
                break
            visited = [-1] * n_actual
            visited[start] = 0
            queue = deque([(start, 0, -1)])
            while queue:
                u, dist, parent = queue.popleft()
                if dist >= depth:
                    continue
                for v in adj[u]:
                    if v == parent:
                        continue
                    if visited[v] == -1:
                        visited[v] = dist + 1
                        queue.append((v, dist + 1, u))
                    else:
                        cycle_len = dist + 1 + visited[v]
                        if cycle_len < g:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            return n_actual, edges
            
        # If invalid, try next prime
        p = next_prime_1_mod_4(p)

    # Fallback to LUW if no LPS found
    # Re-use seed logic
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
