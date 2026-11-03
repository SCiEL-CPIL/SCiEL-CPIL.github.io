import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    def verify_decomposition(U, V, W, R):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        total = 0.0
                        for r in range(R):
                            total += W[k * n + i][r] * U[i * m + j][r] * V[jp * p + k][r]
                        expected = 1.0 if j == jp else 0.0
                        if abs(total - expected) > 1e-9:
                            return False
        return True
    
    def strassen_block_terms(i1, i2, j1, j2, k1, k2):
        terms = []
        U1 = {(i1, j1): 1, (i2, j2): 1}
        V1 = {(j1, k1): 1, (j2, k2): 1}
        W1 = {(k1, i1): 1, (k2, i2): 1}
        terms.append((U1, V1, W1))
        U2 = {(i2, j1): 1, (i2, j2): 1}
        V2 = {(j1, k1): 1}
        W2 = {(k1, i2): 1, (k2, i2): -1}
        terms.append((U2, V2, W2))
        U3 = {(i1, j1): 1}
        V3 = {(j1, k2): 1, (j2, k2): -1}
        W3 = {(k2, i1): 1, (k2, i2): 1}
        terms.append((U3, V3, W3))
        U4 = {(i2, j2): 1}
        V4 = {(j2, k1): 1, (j1, k1): -1}
        W4 = {(k1, i2): 1, (k1, i1): 1}
        terms.append((U4, V4, W4))
        U5 = {(i1, j1): 1, (i1, j2): 1}
        V5 = {(j2, k2): 1}
        W5 = {(k2, i1): 1, (k1, i1): -1}
        terms.append((U5, V5, W5))
        U6 = {(i2, j1): 1, (i1, j1): -1}
        V6 = {(j1, k1): 1, (j1, k2): 1}
        W6 = {(k2, i2): 1}
        terms.append((U6, V6, W6))
        U7 = {(i1, j2): 1, (i2, j2): -1}
        V7 = {(j2, k1): 1, (j2, k2): 1}
        W7 = {(k1, i1): 1}
        terms.append((U7, V7, W7))
        return terms
    
    def build_4block_decomposition():
        """4 disjoint Strassen blocks + schoolbook for remainder (rank 56)."""
        blocks = [
            (0, 1, 0, 1, 0, 1),
            (0, 1, 2, 3, 0, 1),
            (0, 1, 0, 1, 2, 3),
            (0, 1, 2, 3, 2, 3),
        ]
        all_terms = []
        covered = set()
        for i1, i2, j1, j2, k1, k2 in blocks:
            block_terms = strassen_block_terms(i1, i2, j1, j2, k1, k2)
            all_terms.extend(block_terms)
            for i in [i1, i2]:
                for j in [j1, j2]:
                    for k in [k1, k2]:
                        covered.add((i, j, k))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U_sb = {(i, j): 1}
                        V_sb = {(j, k): 1}
                        W_sb = {(k, i): 1}
                        all_terms.append((U_sb, V_sb, W_sb))
        R = len(all_terms)
        U = [[0] * R for _ in range(dim_u)]
        V = [[0] * R for _ in range(dim_v)]
        W = [[0] * R for _ in range(dim_w)]
        for c, (U_dict, V_dict, W_dict) in enumerate(all_terms):
            for (i, j), coeff in U_dict.items():
                U[i * m + j][c] = coeff
            for (j, k), coeff in V_dict.items():
                V[j * p + k][c] = coeff
            for (k, i), coeff in W_dict.items():
                W[k * n + i][c] = coeff
        if not verify_decomposition(U, V, W, R):
            return None
        return U, V, W, R
    
    # Try 4-block decomposition
    result = build_4block_decomposition()
    if result is not None:
        U, V, W, R = result
        return {"U": U, "V": V, "W": W}
    
    # Final fallback: pure schoolbook
    R = n * m * p
    U = [[0] * R for _ in range(dim_u)]
    V = [[0] * R for _ in range(dim_v)]
    W = [[0] * R for _ in range(dim_w)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
