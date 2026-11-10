import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # Leave 10s for safety
    
    if k != 5:
        return _construct_luw(k, g)
        
    for q in [13, 17, 19, 23, 29, 31, 37, 41]:
        if time.time() > deadline:
            break
            
        n = q * (q * q - 1) // 2
        if n < 1066:
            continue
            
        elements, idx_map, inverses, orders, mul_table, ident_idx = _generate_psl2_fast(q)
        
        involutions = []
        non_involutions = []
        for i in range(n):
            if orders[i] == 2:
                involutions.append(i)
            elif orders[i] > 2:
                non_involutions.append(i)
                
        if not involutions or not non_involutions:
            continue
            
        found = False
        attempts = 0
        max_attempts = 200000
        
        while attempts < max_attempts:
            if time.time() > deadline:
                break
            attempts += 1
            
            # Try different strategies for selecting generators
            if attempts % 3 == 0:
                # Strategy 1: random
                s1_idx = random.choice(non_involutions)
                s2_idx = random.choice(non_involutions)
                s3_idx = random.choice(involutions)
            elif attempts % 3 == 1:
                # Strategy 2: prefer elements with larger order for s1, s2
                s1_idx = random.choice(non_involutions[:len(non_involutions)//2])
                s2_idx = random.choice(non_involutions[len(non_involutions)//2:])
                s3_idx = random.choice(involutions)
            else:
                # Strategy 3: random with different distribution
                s1_idx = random.choice(non_involutions)
                s2_idx = random.choice(non_involutions)
                s3_idx = random.choice(involutions)
            
            s1_inv_idx = inverses[s1_idx]
            s2_inv_idx = inverses[s2_idx]
            
            candidates = [s1_idx, s1_inv_idx, s2_idx, s2_inv_idx, s3_idx]
            
            if len(set(candidates)) != 5:
                continue
                
            if _check_girth_cayley_fast(n, candidates, mul_table, ident_idx, g):
                edges = []
                for i in range(n):
                    for s in candidates:
                        j = mul_table[i][s]
                        if i < j:
                            edges.append((i, j))
                
                if len(edges) == n * 5 // 2:
                    return n, edges
                    
    return _construct_luw(k, g)

def _generate_psl2_fast(q):
    elements = []
    idx_map = {}
    
    inv = [0] * q
    for i in range(1, q):
        inv[i] = pow(i, q - 2, q)
        
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0:
                    if b == 0:
                        continue
                    if (b * c) % q != (q - 1) % q:
                        continue
                    d = 0
                else:
                    d = (1 + b * c) % q
                    d = (d * inv[a]) % q
                
                if (a * d - b * c) % q != 1:
                    continue
                
                m = (a, b, c, d)
                first_non_zero = None
                for val in m:
                    if val != 0:
                        first_non_zero = val
                        break
                        
                if first_non_zero is not None and first_non_zero > (q - 1) // 2:
                    m = ((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q)
                    
                if m not in idx_map:
                    idx_map[m] = len(elements)
                    elements.append(m)
                    
    n = len(elements)
    
    # Use numpy for vectorized multiplication table generation
    elems_np = np.array(elements, dtype=np.int64)
    
    # Precompute inverses and orders
    inverses = [0] * n
    orders = [0] * n
    
    ident = (1, 0, 0, 1)
    ident_idx = idx_map[ident]
    
    # Compute inverses
    for i in range(n):
        a, b, c, d = elements[i]
        inv_m = (d, (q - b) % q, (q - c) % q, a)
        first_non_zero = None
        for val in inv_m:
            if val != 0:
                first_non_zero = val
                break
        if first_non_zero is not None and first_non_zero > (q - 1) // 2:
            inv_m = ((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q)
        inverses[i] = idx_map[inv_m]
    
    # Compute orders
    for i in range(n):
        cur = i
        order = 1
        while cur != ident_idx:
            cur = idx_map[tuple((np.array(elements[cur]) * np.array(elements[i])) % q)] if False else _mul_psl2_idx(elements[cur], elements[i], q, idx_map)
            order += 1
            if order > q + 2:
                break
        orders[i] = order
    
    # Build multiplication table using numpy broadcasting
    # For each pair (i, j), compute product
    # We'll do this in batches to manage memory
    batch_size = min(n, 256)
    mul_table = np.empty((n, n), dtype=np.int32)
    
    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        batch_i = elements[i_start:i_end]
        
        for j in range(n):
            a2, b2, c2, d2 = elements[j]
            for k in range(i_end - i_start):
                a1, b1, c1, d1 = batch_i[k]
                a = (a1 * a2 + b1 * c2) % q
                b = (a1 * b2 + b1 * d2) % q
                c = (c1 * a2 + d1 * c2) % q
                d = (c1 * b2 + d1 * d2) % q
                prod = (a, b, c, d)
                first_non_zero = None
                for val in prod:
                    if val != 0:
                        first_non_zero = val
                        break
                if first_non_zero is not None and first_non_zero > (q - 1) // 2:
                    prod = ((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q)
                mul_table[i_start + k, j] = idx_map[prod]
    
    # Convert to list of lists for compatibility
    mul_table_list = mul_table.tolist()
    
    return elements, idx_map, inverses, orders, mul_table_list, ident_idx

def _mul_psl2_idx(e1, e2, q, idx_map):
    a1, b1, c1, d1 = e1
    a2, b2, c2, d2 = e2
    a = (a1 * a2 + b1 * c2) % q
    b = (a1 * b2 + b1 * d2) % q
    c = (c1 * a2 + d1 * c2) % q
    d = (c1 * b2 + d1 * d2) % q
    prod = (a, b, c, d)
    first_non_zero = None
    for val in prod:
        if val != 0:
            first_non_zero = val
            break
    if first_non_zero is not None and first_non_zero > (q - 1) // 2:
        prod = ((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q)
    return idx_map[prod]

def _check_girth_cayley_fast(n, candidates, mul_table, ident_idx, g):
    """
    Check girth using BFS from identity with cycle detection.
    Returns True if no cycle shorter than g exists.
    """
    dist = [-1] * n
    dist[ident_idx] = 0
    queue = deque([ident_idx])
    
    max_depth = g - 1  # We only need to check up to depth g-1
    
    while queue:
        u = queue.popleft()
        du = dist[u]
        
        if du >= max_depth:
            continue
            
        for s in candidates:
            v = mul_table[u][s]
            
            if v == ident_idx:
                if du > 0:
                    cycle_len = du + 1
                    if cycle_len < g:
                        return False
            else:
                if dist[v] == -1:
                    dist[v] = du + 1
                    if du + 1 < max_depth:
                        queue.append(v)
                else:
                    # Found a cycle
                    if dist[v] != dist[u] - 1:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
                            
    return True

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
