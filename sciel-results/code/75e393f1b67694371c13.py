import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses maximum packing of 2x2x2 Strassen blocks + schoolbook for rest.
    Total rank = 42 + 57 = 99.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    def add_schoolbook(i, j, k):
        add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    # Maximum packing of 6 disjoint 2x2x2 Strassen blocks.
    # j-pairs: {0,1}, {2,3}  (2 pairs, covering j=0..3)
    # k-pairs: {0,1}, {2,3}, {4,5}  (3 pairs, covering k=0..5)
    # i-pairs: alternate {0,1} and {1,2}
    
    # 6 blocks: 2 j-pairs x 3 k-pairs
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    covered = set()
    block_idx = 0
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            # Alternate i-pair
            if block_idx % 2 == 0:
                i1, i2 = 0, 1
            else:
                i1, i2 = 1, 2
            add_strassen_block(i1, i2, j1, j2, k1, k2)
            for i in (i1, i2):
                for j in (j1, j2):
                    for k in (k1, k2):
                        covered.add((i, j, k))
            block_idx += 1

    # Add schoolbook for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    # If we hit the deadline, add all remaining schoolbook terms
                    # to ensure correctness, then return what we have
                    for i2 in range(i, n):
                        for j2 in range(j if i2 == i else 0, m):
                            for k2 in range(k if i2 == i and j2 == j else 0, p):
                                if (i2, j2, k2) not in covered:
                                    add_schoolbook(i2, j2, k2)
                    rank = len(U_cols)
                    U = [[0] * rank for _ in range(U_rows)]
                    V = [[0] * rank for _ in range(V_rows)]
                    W = [[0] * rank for _ in range(W_rows)]
                    for c in range(rank):
                        for r in range(U_rows):
                            U[r][c] = U_cols[c][r]
                        for r in range(V_rows):
                            V[r][c] = V_cols[c][r]
                        for r in range(W_rows):
                            W[r][c] = W_cols[c][r]
                    return {"U": U, "V": V, "W": W}
                if (i, j, k) not in covered:
                    add_schoolbook(i, j, k)

    rank = len(U_cols)
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
