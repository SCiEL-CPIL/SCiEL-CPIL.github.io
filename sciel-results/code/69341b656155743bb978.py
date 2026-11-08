import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    This is the best known upper bound for the rank of the 4x4 matrix multiplication tensor.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1
    """
    start_time = time.time()
    DEADLINE = 140  # Safety margin
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def get_U_index(row, col):
        return row * m + col
    
    def get_V_index(row, col):
        return row * p + col
    
    def get_W_index(row, col):
        # Trace convention: W[k*n+i, r] where k=col of C, i=row of C
        # In C[i, j], i is row, j is col.
        # The task says W[k*n+i, r].
        # Usually W is indexed by the output indices.
        # If C[i,j] is the output, the index is j*n + i?
        # Let's check the seed: W[k*n+i, r] where k=col, i=row.
        # So for C[row, col], index is col * n + row.
        return col * n + row

    # Recursive Strassen
    # We define the 7 block products P_t = L_t * R_t
    # Then each P_t is a 2x2 product computed via Strassen (7 scalar mults)
    
    # L_t and R_t are 2x2 matrices of linear forms on A and B respectively.
    # We represent a linear form as a dictionary {(row, col): coeff}
    
    def a_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
        
    def b_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)

    # Define the 7 terms
    # L_terms[t] is 2x2 list of dicts
    # R_terms[t] is 2x2 list of dicts
    # contrib_terms[t] maps (r,c) in 2x2 block to list of (global_row, global_col, coeff) in C
    
    L_terms = []
    R_terms = []
    contrib_terms = []
    
    # Helper to add to a dict
    def add_to_dict(d, key, val):
        d[key] = d.get(key, 0) + val

    # Term 0: P1 = (A11 + A22)(B11 + B22)
    L0 = [[{} for _ in range(2)] for _ in range(2)]
    R0 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            # L0[r,c] = A11[r,c] + A22[r,c]
            add_to_dict(L0[r][c], a_elem(0,0,r,c), 1)
            add_to_dict(L0[r][c], a_elem(1,1,r,c), 1)
            # R0[r,c] = B11[r,c] + B22[r,c]
            add_to_dict(R0[r][c], b_elem(0,0,r,c), 1)
            add_to_dict(R0[r][c], b_elem(1,1,r,c), 1)
    # Contrib: C11 += P1, C22 += P1
    c0 = {}
    for r in range(2):
        for c in range(2):
            # P is 2x2.
            # P[r,c] contributes to C[2*br+r, 2*bc+c]
            # For P1, it contributes to C11 (block 0,0) and C22 (block 1,1)
            # So P1[r,c] -> C[2*0+r, 2*0+c] and C[2*1+r, 2*1+c]
            c0[(r,c)] = [
                (r, c, 1),          # Block (0,0) -> Row r, Col c
                (2+r, 2+c, 1)       # Block (1,1) -> Row 2+r, Col 2+c
            ]
    L_terms.append(L0); R_terms.append(R0); contrib_terms.append(c0)

    # Term 1: P2 = (A21 + A22)B11
    L1 = [[{} for _ in range(2)] for _ in range(2)]
    R1 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            add_to_dict(L1[r][c], a_elem(1,0,r,c), 1)
            add_to_dict(L1[r][c], a_elem(1,1,r,c), 1)
            add_to_dict(R1[r][c], b_elem(0,0,r,c), 1)
    c1 = {}
    for r in range(2):
        for c in range(2):
            # P2 contributes to C21 (+) and C22 (-)
            c1[(r,c)] = [
                (2+r, c, 1),
                (2+r, 2+c, -1)
            ]
    L_terms.append(L1); R_terms.append(R1); contrib_terms.append(c1)

    # Term 2: P3 = A11(B12 - B22)
    L2 = [[{} for _ in range(2)] for _ in range(2)]
    R2 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            add_to_dict(L2[r][c], a_elem(0,0,r,c), 1)
            add_to_dict(R2[r][c], b_elem(0,1,r,c), 1)
            add_to_dict(R2[r][c], b_elem(1,1,r,c), -1)
    c2 = {}
    for r in range(2):
        for c in range(2):
            # P3 contributes to C12 (+) and C22 (+)
            c2[(r,c)] = [
                (r, 2+c, 1),
                (2+r, 2+c, 1)
            ]
    L_terms.append(L2); R_terms.append(R2); contrib_terms.append(c2)

    # Term 3: P4 = A22(B21 - B11)
    L3 = [[{} for _ in range(2)] for _ in range(2)]
    R3 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            add_to_dict(L3[r][c], a_elem(1,1,r,c), 1)
            add_to_dict(R3[r][c], b_elem(1,0,r,c), 1)
            add_to_dict(R3[r][c], b_elem(0,0,r,c), -1)
    c3 = {}
    for r in range(2):
        for c in range(2):
            # P4 contributes to C11 (+) and C21 (+)
            c3[(r,c)] = [
                (r, c, 1),
                (2+r, c, 1)
            ]
    L_terms.append(L3); R_terms.append(R3); contrib_terms.append(c3)

    # Term 4: P5 = (A11 + A12)B22
    L4 = [[{} for _ in range(2)] for _ in range(2)]
    R4 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            add_to_dict(L4[r][c], a_elem(0,0,r,c), 1)
            add_to_dict(L4[r][c], a_elem(0,1,r,c), 1)
            add_to_dict(R4[r][c], b_elem(1,1,r,c), 1)
    c4 = {}
    for r in range(2):
        for c in range(2):
            # P5 contributes to C11 (-) and C12 (+)
            c4[(r,c)] = [
                (r, c, -1),
                (r, 2+c, 1)
            ]
    L_terms.append(L4); R_terms.append(R4); contrib_terms.append(c4)

    # Term 5: P6 = (A21 - A11)(B11 + B12)
    L5 = [[{} for _ in range(2)] for _ in range(2)]
    R5 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            add_to_dict(L5[r][c], a_elem(1,0,r,c), 1)
            add_to_dict(L5[r][c], a_elem(0,0,r,c), -1)
            add_to_dict(R5[r][c], b_elem(0,0,r,c), 1)
            add_to_dict(R5[r][c], b_elem(0,1,r,c), 1)
    c5 = {}
    for r in range(2):
        for c in range(2):
            # P6 contributes to C22 (+)
            c5[(r,c)] = [
                (2+r, 2+c, 1)
            ]
    L_terms.append(L5); R_terms.append(R5); contrib_terms.append(c5)

    # Term 6: P7 = (A12 - A22)(B21 + B22)
    L6 = [[{} for _ in range(2)] for _ in range(2)]
    R6 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            add_to_dict(L6[r][c], a_elem(0,1,r,c), 1)
            add_to_dict(L6[r][c], a_elem(1,1,r,c), -1)
            add_to_dict(R6[r][c], b_elem(1,0,r,c), 1)
            add_to_dict(R6[r][c], b_elem(1,1,r,c), 1)
    c6 = {}
    for r in range(2):
        for c in range(2):
            # P7 contributes to C11 (+)
            c6[(r,c)] = [
                (r, c, 1)
            ]
    L_terms.append(L6); R_terms.append(R6); contrib_terms.append(c6)

    # Now, for each term t, compute the 7 scalar products using Strassen on the 2x2 L_t and R_t
    # Sub-contributions for 2x2 Strassen result P = M1...M7
    # P11 = M1 + M4 - M5 + M7
    # P12 = M3 + M5
    # P21 = M2 + M4
    # P22 = M1 - M2 + M3 + M6
    
    sub_contrib = [
        [(0,0,1), (1,1,1)],    # M1
        [(1,0,1), (1,1,-1)],   # M2
        [(0,1,1), (1,1,1)],    # M3
        [(0,0,1), (1,0,1)],    # M4
        [(0,0,-1), (0,1,1)],   # M5
        [(1,1,1)],             # M6
        [(0,0,1)]              # M7
    ]

    for t in range(7):
        L_t = L_terms[t]
        R_t = R_terms[t]
        contrib_t = contrib_terms[t]
        
        # Extract a,b,c,d from L_t and e,f,g,h from R_t
        a = L_t[0][0]; b = L_t[0][1]; c = L_t[1][0]; d = L_t[1][1]
        e = R_t[0][0]; f = R_t[0][1]; g = R_t[1][0]; h = R_t[1][1]
        
        # Define 7 sub-products X_k * Y_k
        # X_k is dict, Y_k is dict
        sub_products = []
        
        # M1: (a+d)(e+h)
        x1 = {}
        for dd in [a, d]:
            for k, v in dd.items(): x1[k] = x1.get(k, 0) + v
        y1 = {}
        for dd in [e, h]:
            for k, v in dd.items(): y1[k] = y1.get(k, 0) + v
        sub_products.append((x1, y1))
        
        # M2: (c+d)e
        x2 = {}
        for dd in [c, d]:
            for k, v in dd.items(): x2[k] = x2.get(k, 0) + v
        y2 = dict(e)
        sub_products.append((x2, y2))
        
        # M3: a(f-h)
        x3 = dict(a)
        y3 = {}
        for k, v in f.items(): y3[k] = y3.get(k, 0) + v
        for k, v in h.items(): y3[k] = y3.get(k, 0) - v
        sub_products.append((x3, y3))
        
        # M4: d(g-e)
        x4 = dict(d)
        y4 = {}
        for k, v in g.items(): y4[k] = y4.get(k, 0) + v
        for k, v in e.items(): y4[k] = y4.get(k, 0) - v
        sub_products.append((x4, y4))
        
        # M5: (a+b)h
        x5 = {}
        for dd in [a, b]:
            for k, v in dd.items(): x5[k] = x5.get(k, 0) + v
        y5 = dict(h)
        sub_products.append((x5, y5))
        
        # M6: (c-a)(e+f)
        x6 = {}
        for k, v in c.items(): x6[k] = x6.get(k, 0) + v
        for k, v in a.items(): x6[k] = x6.get(k, 0) - v
        y6 = {}
        for dd in [e, f]:
            for k, v in dd.items(): y6[k] = y6.get(k, 0) + v
        sub_products.append((x6, y6))
        
        # M7: (b-d)(g+h)
        x7 = {}
        for k, v in b.items(): x7[k] = x7.get(k, 0) + v
        for k, v in d.items(): x7[k] = x7.get(k, 0) - v
        y7 = {}
        for dd in [g, h]:
            for k, v in dd.items(): y7[k] = y7.get(k, 0) + v
        sub_products.append((x7, y7))
        
        # Add to U, V, W
        for k in range(7):
            x_k, y_k = sub_products[k]
            rank_idx = t * 7 + k
            
            for (row, col), coeff in x_k.items():
                u_idx = get_U_index(row, col)
                U[u_idx, rank_idx] += coeff
            
            for (row, col), coeff in y_k.items():
                v_idx = get_V_index(row, col)
                V[v_idx, rank_idx] += coeff
            
            for (pr, pc, sub_coeff) in sub_contrib[k]:
                if (pr, pc) in contrib_t:
                    for (out_row, out_col, block_coeff) in contrib_t[(pr, pc)]:
                        w_idx = get_W_index(out_row, out_col)
                        W[w_idx, rank_idx] += sub_coeff * block_coeff

    # Check deadline
    if time.time() > DEADLINE:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
