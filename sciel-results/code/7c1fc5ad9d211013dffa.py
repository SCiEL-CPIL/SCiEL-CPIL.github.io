import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Moore bound for even girth g=2R
    R = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(R))
    
    # We search for a circulant graph C_n(a, b)
    # n must be at least moore_bound
    # We search n in [moore_bound, moore_bound + 100]
    # If not found, we expand range or fallback
    
    # Precompute non-backtracking walk sums for lengths 3 to g-1
    # A non-backtracking walk is a sequence of steps s_1, ..., s_L 
    # from {a, -a, b, -b} such that s_i != -s_{i-1}.
    # The sum is sum(s_i).
    # We check if sum(s_i) == 0 mod n.
    
    # To be efficient, we can represent the state as (last_step, current_sum)
    # But current_sum depends on a and b.
    # Instead, we collect the set of "coefficient pairs" (c1, c2) such that 
    # c1*a + c2*b is the sum of a non-backtracking walk of length L.
    # If (c1, c2) is in the set, and c1*a + c2*b == 0 mod n, then we have a cycle of length L.
    
    valid_coeffs = set()
    if g > 3:
        # State: (last_step, c1, c2)
        # last_step is in {1, -1, 2, -2} representing a, -a, b, -b
        # c1, c2 are coefficients for a and b
        states = set()
        
        # Length 1
        states.add((1, 1, 0))
        states.add((-1, -1, 0))
        states.add((2, 0, 1))
        states.add((-2, 0, -1))
        
        for L in range(2, g):
            new_states = set()
            for last_step, c1, c2 in states:
                for step in [1, -1, 2, -2]:
                    if step == -last_step:
                        continue
                    nc1 = c1 + (1 if step == 1 else -1 if step == -1 else 0)
                    nc2 = c2 + (1 if step == 2 else -1 if step == -2 else 0)
                    new_states.add((step, nc1, nc2))
            states = new_states
            if L >= 3:
                for _, c1, c2 in states:
                    valid_coeffs.add((c1, c2))
    
    best_n = None
    best_edges = None
    
    # Search range
    # Start from Moore bound
    # Circulant graphs often exist near the Moore bound
    n_max_search = moore_bound + 200
    
    for n in range(moore_bound, n_max_search + 1):
        if time.time() > deadline - 5:
            break
            
        found = False
        
        # Iterate a from 1 to n//2 - 1
        # To avoid duplicates, restrict b > a
        # Also a, b must be coprime to n? No, not required for simple graph, 
        # but if gcd(a,n) > 1, the graph might be disconnected. 
        # The task requires a connected graph? "small 4-regular graph". 
        # Usually cages are connected. 
        # If disconnected, it's not a cage. 
        # We should ensure connectivity or at least that the graph is a single component.
        # For circulant C_n(a,b), it is connected iff gcd(n, a, b) = 1.
        
        for a in range(1, n // 2):
            if time.time() > deadline - 2:
                break
            for b in range(a + 1, n // 2):
                if a + b == n:
                    continue
                
                # Check connectivity: gcd(n, a, b) == 1
                import math
                if math.gcd(math.gcd(n, a), b) != 1:
                    continue
                
                # Check girth
                valid = True
                for c1, c2 in valid_coeffs:
                    if (c1 * a + c2 * b) % n == 0:
                        valid = False
                        break
                
                if valid:
                    # Construct edges
                    edges = set()
                    for i in range(n):
                        j1 = (i + a) % n
                        if i < j1:
                            edges.add((i, j1))
                        else:
                            edges.add((j1, i))
                        
                        j2 = (i + b) % n
                        if i < j2:
                            edges.add((i, j2))
                        else:
                            edges.add((j2, i))
                    
                    best_n = n
                    best_edges = list(edges)
                    found = True
                    break
            
            if found:
                break
        
        if found:
            return best_n, best_edges

    # Fallback: LUW Construction
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q

    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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
                if side == 0:
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    queue = [0]
    head = 0
    
    while head < len(order):
        if time.time() > deadline:
            break
        v_idx = queue[head]
        head += 1
        v = order[v_idx]
        for other in neighbours(v):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            if v[0] == 0:
                edges.append((v_idx, index[other]))
                
    return len(order), edges
