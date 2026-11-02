"""
Construct a small 5-regular graph of girth 9 (cage problem n(5,9)).

Uses the LUW construction with GF(4) arithmetic and optimized vectorized operations.
"""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    q = k  # For k=5, we'll use a different approach
    # Actually, the LUW construction gives degree q+1 where q is a prime power.
    # For k=5, q=4.
    
    q = 4  # GF(4)
    
    # GF(4) tables
    add_table = np.array([
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ], dtype=np.int64)
    
    mul_table = np.array([
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ], dtype=np.int64)
    
    # Choose d (odd, >= 3) such that d + 5 >= g
    d_min = g - 5
    if d_min < 3:
        d_min = 3
    if d_min % 2 == 0:
        d_min += 1
    d = d_min
    
    m = (d + 1) // 2
    qm = q ** m
    n = 2 * qm
    
    # Rule
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule_a = np.array([rule[c][0] for c in range(2, d + 1)], dtype=np.int64)
    rule_b = np.array([rule[c][1] for c in range(2, d + 1)], dtype=np.int64)
    
    # Precompute all vertex coordinates
    # Vertex i (0 <= i < qm) on side 0 has coords x[1..m]
    # Vertex i + qm on side 1 has same coords
    
    # Precompute coords for all qm vertices
    coords = np.zeros((qm, d + 1), dtype=np.int64)
    for idx in range(qm):
        temp = idx
        for i in range(m):
            coords[idx, i + 1] = temp % q
            temp //= q
    
    # For each side-0 vertex, compute neighbors
    edges = []
    edge_set = set()
    
    for idx in range(qm):
        if time.time() > deadline - 5.0:
            break
        
        x = coords[idx]
        
        # q neighbors via parameter a
        for a in range(q):
            y = np.zeros(d + 1, dtype=np.int64)
            y[1] = a
            for c in range(2, d + 1):
                ac = rule_a[c - 2]
                bc = rule_b[c - 2]
                prod = mul_table[y[ac], x[bc]]
                y[c] = add_table[x[c], prod]
            
            # Encode y
            y_idx = 0
            for i in range(m):
                y_idx = y_idx * q + y[i + 1]
            
            neighbor = qm + y_idx
            edge = (min(idx, neighbor), max(idx, neighbor))
            if edge not in edge_set:
                edge_set.add(edge)
                edges.append(edge)
        
        # Diagonal neighbor
        neighbor = qm + idx
        edge = (min(idx, neighbor), max(idx, neighbor))
        if edge not in edge_set:
            edge_set.add(edge)
            edges.append(edge)
    
    return n, edges
# EVOLVE-BLOCK-END
