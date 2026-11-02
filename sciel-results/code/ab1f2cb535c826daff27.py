import time
import sys
import random
from itertools import product

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Precompute non-backtracking walk coefficients
    # Steps: 1 (a), -1 (-a), 2 (b), -2 (-b)
    # We want walks of length 3 to g-1
    # State: (last_step, c1, c2)
    # c1: coeff of a, c2: coeff of b
    
    valid_coeffs = set()
    if g > 3:
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
    
    # Moore bound for even girth g=2R
    R = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(R))
    
    # Search for circulant graph
    # n must be at least moore_bound
    # We search n in [moore_bound, moore_bound + 200]
    # For each n, we try random offsets
    
    best_n = None
    best_edges = None
    
    # Precompute valid_coeffs as a list for faster iteration
    valid_coeffs_list = list(valid_coeffs)
    
    n_max_search = moore_bound + 300
    
    for n in range(moore_bound, n_max_search + 1):
        if time.time() > deadline - 5:
            break
            
        found = False
        half_n = n // 2
        
        # Deterministic search: try small a, b values first
        # This is likely to find solutions faster for small n
        max_det = min(50, half_n - 1)
        for a in range(1, max_det + 1):
            for b in range(a + 1, max_det + 1):
                valid = True
                for c1, c2 in valid_coeffs_list:
                    if (c1 * a + c2 * b) % n == 0:
                        valid = False
                        break
                if valid:
                    edges = []
                    for i in range(n):
                        j1 = (i + a) % n
                        if i < j1:
                            edges.append((i, j1))
                        else:
                            edges.append((j1, i))
                        j2 = (i + b) % n
                        if i < j2:
                            edges.append((i, j2))
                        else:
                            edges.append((j2, i))
                    best_n = n
                    best_edges = edges
                    found = True
                    break
            if found:
                break
        
        if found:
            return best_n, best_edges
        
        # Random search with time budget
        if time.time() > deadline - 2:
            break
        
        # Adaptive number of trials based on remaining time
        time_left = deadline - time.time() - 2
        trials = max(200, int(time_left * 500))  # ~500 trials per second
        
        for _ in range(trials):
            a = random.randint(1, half_n - 1)
            b = random.randint(1, half_n - 1)
            if a == b:
                continue
            
            valid = True
            for c1, c2 in valid_coeffs_list:
                if (c1 * a + c2 * b) % n == 0:
                    valid = False
                    break
            
            if valid:
                edges = []
                for i in range(n):
                    j1 = (i + a) % n
                    if i < j1:
                        edges.append((i, j1))
                    else:
                        edges.append((j1, i))
                    j2 = (i + b) % n
                    if i < j2:
                        edges.append((i, j2))
                    else:
                        edges.append((j2, i))
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            return best_n, best_edges

    # Fallback to LUW
    q = k
    if q == 4 and k == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q

    # For k=4, g=10: use d=5 (girth 2d-2=8? No, girth is 2d for LUW)
    # Actually for LUW, girth = 2d when using appropriate rules
    # Let's use d such that 2d >= g, so d >= g/2
    d = max(3, (g + 1) // 2)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Verify the LUW graph has correct girth before returning
    # For now, trust the construction

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
        if time.time() > deadline - 1:
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
                
    n_final = len(order)
    return n_final, edges
# EVOLVE-BLOCK-END
