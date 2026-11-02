"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 115  # safety margin before 120s kill
    q = k
    
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)
    
    if q == 4:
        add_table = np.array([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]], dtype=np.uint8)
        mul_table = np.array([[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]], dtype=np.uint8)
        sub_table = add_table
    else:
        add_table = np.array([[(i+j) % q for j in range(q)] for i in range(q)], dtype=np.uint16)
        mul_table = np.array([[(i*j) % q for j in range(q)] for i in range(q)], dtype=np.uint16)
        sub_table = np.array([[(i-j) % q for j in range(q)] for i in range(q)], dtype=np.uint16)
    
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        pairs = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule_a[c], rule_b[c] = pairs[r]
    
    # Encode: vertex_id = side * q^d + x[1]*q^(d-1) + x[2]*q^(d-2) + ... + x[d]*q^0
    q_pow = [1] * (d + 1)
    for i in range(1, d + 1):
        q_pow[i] = q_pow[i-1] * q
    base = q_pow[d]  # q^d
    
    # Precompute digit extraction tables for fast decoding
    # For each position i (1-indexed from left), digit value = (x_idx // q^(d-i)) % q
    digit_tables = []
    for i in range(1, d + 1):
        divisor = q_pow[d - i]
        digit_tables.append((divisor, q))
    
    # Precompute transitions more efficiently
    # For each side, precompute the transition as a 2D array: trans[side][x_idx, first] = y_idx
    trans = [None, None]
    
    for side in range(2):
        # We'll compute transitions on-the-fly during BFS to save memory
        # But precompute the rule indices for speed
        trans[side] = None
    
    # BFS using integer vertex IDs
    start_id = 0  # side=0, x_idx=0
    
    index = {start_id: 0}
    order = [start_id]
    edges = []
    
    queue = deque([0])
    
    # Precompute for each position c, the rule_a[c] and rule_b[c] as numpy arrays for vectorization
    # Actually, let's just optimize the inner loop
    
    while queue:
        if time.time() > deadline - 3:
            break
        
        vi = queue.popleft()
        v = order[vi]
        side = v // base
        x_idx = v % base
        
        # Decode x_idx to get x[1..d]
        x = [0] * (d + 1)
        rem = x_idx
        for i in range(d, 0, -1):
            x[i] = rem % q
            rem //= q
        
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a = rule_a[c]
                b = rule_b[c]
                if side == 0:
                    y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                else:
                    y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
            
            # Encode y[1..d] to y_idx
            y_idx = 0
            for i in range(1, d + 1):
                y_idx = y_idx * q + y[i]
            
            other = (1 - side) * base + y_idx
            
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            
            if side == 0:
                edges.append((vi, index[other]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
