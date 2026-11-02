"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=4
    
    # We construct a rank 35 decomposition.
    # It consists of:
    # 1. A rank 7 Strassen decomposition for the 2x2x2 sub-block (i,j,k in 0..1)
    # 2. The remaining schoolbook terms for the 3x3x3 block (i,j,k in 0..2) excluding the 2x2x2 sub-block
    # 3. The schoolbook terms for the 3x3x1 block (k=3)
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    # u_indices: list of (idx, coeff) for A (size 9)
    # v_indices: list of (idx, coeff) for B (size 12)
    # w_indices: list of (idx, coeff) for C (size 12)
    def add_term(u_indices, v_indices, w_indices):
        U_row = [0.0] * (n * m)
        V_row = [0.0] * (m * p)
        W_row = [0.0] * (p * n)
        
        for idx, coeff in u_indices:
            U_row[idx] = coeff
        for idx, coeff in v_indices:
            V_row[idx] = coeff
        for idx, coeff in w_indices:
            W_row[idx] = coeff
            
        # Append to lists
        # We will build the matrices as lists of lists of columns?
        # No, the interface requires U (n*m, rank), V (m*p, rank), W (p*n, rank).
        # So U is a list of rows.
        # It is easier to build a list of columns (terms) and then transpose.
        pass

    # Let's build the terms as a list of tuples (u_list, v_list, w_list)
    terms = []
    
    # 1. Strassen terms for i,j,k in 0..1
    # u, v, w are lists of length 9, 12, 12 respectively, but mostly 0.
    # We represent them as dicts or sparse lists for now, then convert.
    
    def make_u(indices):
        u = [0.0] * 9
        for idx, coeff in indices:
            u[idx] = coeff
        return u

    def make_v(indices):
        v = [0.0] * 12
        for idx, coeff in indices:
            v[idx] = coeff
        return v

    def make_w(indices):
        w = [0.0] * 12
        for idx, coeff in indices:
            w[idx] = coeff
        return w

    # Strassen 2x2x2
    # M1
    u1 = make_u([(0, 1), (3, 1)]) # A00, A11
    v1 = make_v([(0, 1), (1, 1)]) # B00, B01? No.
    # B indices: j*p + k.
    # j=0, k=0 -> 0. j=0, k=1 -> 1. j=1, k=0 -> 4. j=1, k=1 -> 5.
    # Wait, p=4.
    # B_{00} is idx 0. B_{01} is idx 1. B_{10} is idx 4. B_{11} is idx 5.
    # My previous Strassen derivation assumed p=2 for the sub-block.
    # Here p=4.
    # B_{jk} index is j*4 + k.
    # For j=0, k=0: 0.
    # For j=0, k=1: 1.
    # For j=1, k=0: 4.
    # For j=1, k=1: 5.
    
    # C indices: k*n + i.
    # C_{00} is idx 0. C_{01} is idx 1? No.
    # C_{ik} index is k*3 + i.
    # C_{00}: k=0, i=0 -> 0.
    # C_{01}: k=1, i=0 -> 3.
    # C_{10}: k=0, i=1 -> 1.
    # C_{11}: k=1, i=1 -> 4.
    
    # Let's re-verify Strassen terms with correct indices.
    
    # M1 = (A00+A11)(B00+B11)
    # A00: idx 0. A11: idx 3.
    # B00: idx 0. B11: idx 5.
    # Contributes to C00 (idx 0) and C11 (idx 4).
    u1 = make_u([(0, 1), (3, 1)])
    v1 = make_v([(0, 1), (5, 1)])
    w1 = make_w([(0, 1), (4, 1)])
    terms.append((u1, v1, w1))
    
    # M2 = (A10+A11)B00
    # A10: idx 2. A11: idx 3.
    # B00: idx 0.
    # Contributes to C10 (idx 1) and C11 (idx 4).
    u2 = make_u([(2, 1), (3, 1)])
    v2 = make_v([(0, 1)])
    w2 = make_w([(1, 1), (4, 1)])
    terms.append((u2, v2, w2))
    
    # M3 = A00(B10-B11)
    # A00: idx 0.
    # B10: idx 4. B11: idx 5.
    # Contributes to C01 (idx 3) and C11 (idx 4).
    u3 = make_u([(0, 1)])
    v3 = make_v([(4, 1), (5, -1)])
    w3 = make_w([(3, 1), (4, 1)])
    terms.append((u3, v3, w3))
    
    # M4 = A11(B01-B00)
    # A11: idx 3.
    # B01: idx 1. B00: idx 0.
    # Contributes to C00 (idx 0) and C10 (idx 1).
    u4 = make_u([(3, 1)])
    v4 = make_v([(1, 1), (0, -1)])
    w4 = make_w([(0, 1), (1, 1)])
    terms.append((u4, v4, w4))
    
    # M5 = (A00+A01)B11
    # A00: idx 0. A01: idx 1.
    # B11: idx 5.
    # Contributes to C00 (idx 0) and C01 (idx 3).
    # Note: In Strassen, C00 = M1 + M4 - M5 + M7.
    # So M5 has coefficient -1 for C00.
    # C01 = M3 + M5.
    # So M5 has coefficient 1 for C01.
    u5 = make_u([(0, 1), (1, 1)])
    v5 = make_v([(5, 1)])
    w5 = make_w([(0, -1), (3, 1)])
    terms.append((u5, v5, w5))
    
    # M6 = (A10-A00)(B00+B01)
    # A10: idx 2. A00: idx 0.
    # B00: idx 0. B01: idx 1.
    # Contributes to C11 (idx 4).
    # C11 = M1 - M2 + M3 + M6.
    u6 = make_u([(2, 1), (0, -1)])
    v6 = make_v([(0, 1), (1, 1)])
    w6 = make_w([(4, 1)])
    terms.append((u6, v6, w6))
    
    # M7 = (A01-A11)(B10+B11)
    # A01: idx 1. A11: idx 3.
    # B10: idx 4. B11: idx 5.
    # Contributes to C00 (idx 0).
    # C00 = M1 + M4 - M5 + M7.
    u7 = make_u([(1, 1), (3, -1)])
    v7 = make_v([(4, 1), (5, 1)])
    w7 = make_w([(0, 1)])
    terms.append((u7, v7, w7))
    
    # 2. Remaining schoolbook terms for 3x3x3
    # Skip i,j,k in 0..1
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue
                u = make_u([(i * 3 + j, 1)])
                v = make_v([(j * 4 + k, 1)])
                w = make_w([(k * 3 + i, 1)])
                terms.append((u, v, w))
                
    # 3. Schoolbook terms for k=3
    for i in range(3):
        for j in range(3):
            u = make_u([(i * 3 + j, 1)])
            v = make_v([(j * 4 + 3, 1)])
            w = make_w([(3 * 3 + i, 1)])
            terms.append((u, v, w))
            
    rank = len(terms)
    
    # Construct final matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u, v, w = terms[c]
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
