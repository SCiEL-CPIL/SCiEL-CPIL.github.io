import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    def get_U_index(row, col):
        return row * m + col
    
    def get_V_index(row, col):
        return row * p + col
    
    def get_W_index(row, col):
        return col * n + row
    
    # Strassen 2x2:
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
    
    # Block-level Strassen for 4x4:
    # P1=(A11+A22)(B11+B22), P2=(A21+A22)B11, P3=A11(B12-B22), P4=A22(B21-B11)
    # P5=(A11+A12)B22, P6=(A21-A11)(B11+B12), P7=(A12-A22)(B21+B22)
    # C11=P1+P4-P5+P7, C12=P3+P5, C21=P2+P4, C22=P1-P2+P3+P6
    
    L_terms = []
    R_terms = []
    contrib_terms = []
    
    def a_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def b_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def c_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    # Term 0: P1 = (A11+A22)(B11+B22)
    L0 = [[{} for _ in range(2)] for _ in range(2)]
    R0 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L0[r][c] = {a_elem(0,0,r,c): 1, a_elem(1,1,r,c): 1}
            R0[r][c] = {b_elem(0,0,r,c): 1, b_elem(1,1,r,c): 1}
    c0 = {(0,0): [(0,0,1)], (0,1): [(0,0,1)], (1,0): [(1,1,1)], (1,1): [(1,1,1)]}
    L_terms.append(L0); R_terms.append(R0); contrib_terms.append(c0)
    
    # Term 1: P2 = (A21+A22)B11
    L1 = [[{} for _ in range(2)] for _ in range(2)]
    R1 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L1[r][c] = {a_elem(1,0,r,c): 1, a_elem(1,1,r,c): 1}
            R1[r][c] = {b_elem(0,0,r,c): 1}
    c1 = {(0,0): [(1,0,1)], (0,1): [(1,0,1)], (1,0): [(1,1,-1)], (1,1): [(1,1,-1)]}
    L_terms.append(L1); R_terms.append(R1); contrib_terms.append(c1)
    
    # Term 2: P3 = A11(B12-B22)
    L2 = [[{} for _ in range(2)] for _ in range(2)]
    R2 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L2[r][c] = {a_elem(0,0,r,c): 1}
            R2[r][c] = {b_elem(0,1,r,c): 1, b_elem(1,1,r,c): -1}
    c2 = {(0,0): [(0,1,1)], (0,1): [(0,1,1)], (1,0): [(1,1,1)], (1,1): [(1,1,1)]}
    L_terms.append(L2); R_terms.append(R2); contrib_terms.append(c2)
    
    # Term 3: P4 = A22(B21-B11)
    L3 = [[{} for _ in range(2)] for _ in range(2)]
    R3 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L3[r][c] = {a_elem(1,1,r,c): 1}
            R3[r][c] = {b_elem(1,0,r,c): 1, b_elem(0,0,r,c): -1}
    c3 = {(0,0): [(0,0,1)], (0,1): [(0,0,1)], (1,0): [(1,0,1)], (1,1): [(1,0,1)]}
    L_terms.append(L3); R_terms.append(R3); contrib_terms.append(c3)
    
    # Term 4: P5 = (A11+A12)B22
    L4 = [[{} for _ in range(2)] for _ in range(2)]
    R4 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L4[r][c] = {a_elem(0,0,r,c): 1, a_elem(0,1,r,c): 1}
            R4[r][c] = {b_elem(1,1,r,c): 1}
    c4 = {(0,0): [(0,0,-1)], (0,1): [(0,1,1)], (1,0): [(0,0,-1)], (1,1): [(0,1,1)]}
    L_terms.append(L4); R_terms.append(R4); contrib_terms.append(c4)
    
    # Term 5: P6 = (A21-A11)(B11+B12)
    L5 = [[{} for _ in range(2)] for _ in range(2)]
    R5 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L5[r][c] = {a_elem(1,0,r,c): 1, a_elem(0,0,r,c): -1}
            R5[r][c] = {b_elem(0,0,r,c): 1, b_elem(0,1,r,c): 1}
    c5 = {(0,0): [(1,1,1)], (0,1): [(1,1,1)], (1,0): [(1,1,1)], (1,1): [(1,1,1)]}
    L_terms.append(L5); R_terms.append(R5); contrib_terms.append(c5)
    
    # Term 6: P7 = (A12-A22)(B21+B22)
    L6 = [[{} for _ in range(2)] for _ in range(2)]
    R6 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L6[r][c] = {a_elem(0,1,r,c): 1, a_elem(1,1,r,c): -1}
            R6[r][c] = {b_elem(1,0,r,c): 1, b_elem(1,1,r,c): 1}
    c6 = {(0,0): [(0,0,1)], (0,1): [(0,0,1)], (1,0): [(0,0,1)], (1,1): [(0,0,1)]}
    L_terms.append(L6); R_terms.append(R6); contrib_terms.append(c6)
    
    # Strassen sub-product contributions to 2x2 result:
    # M1->C11:+1,C22:+1  M2->C21:+1,C22:-1  M3->C12:+1,C22:+1
    # M4->C11:+1,C21:+1  M5->C11:-1,C12:+1  M6->C22:+1  M7->C11:+1
    sub_contrib = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ]
    
    def add_dict(target, source, sign=1):
        for key, coeff in source.items():
            target[key] = target.get(key, 0) + sign * coeff
    
    for t in range(7):
        a = L_terms[t][0][0]; b = L_terms[t][0][1]
        c = L_terms[t][1][0]; d = L_terms[t][1][1]
        e = R_terms[t][0][0]; f = R_terms[t][0][1]
        g = R_terms[t][1][0]; h = R_terms[t][1][1]
        ct = contrib_terms[t]
        
        sub_products = []
        x = {}; add_dict(x, a); add_dict(x, d); y = {}; add_dict(y, e); add_dict(y, h)
        sub_products.append((x, y))
        x = {}; add_dict(x, c); add_dict(x, d); y = dict(e)
        sub_products.append((x, y))
        x = dict(a); y = {}; add_dict(y, f); add_dict(y, h, -1)
        sub_products.append((x, y))
        x = dict(d); y = {}; add_dict(y, g); add_dict(y, e, -1)
        sub_products.append((x, y))
        x = {}; add_dict(x, a); add_dict(x, b); y = dict(h)
        sub_products.append((x, y))
        x = {}; add_dict(x, c); add_dict(x, a, -1); y = {}; add_dict(y, e); add_dict(y, f)
        sub_products.append((x, y))
        x = {}; add_dict(x, b); add_dict(x, d, -1); y = {}; add_dict(y, g); add_dict(y, h)
        sub_products.append((x, y))
        
        for k in range(7):
            x_k, y_k = sub_products[k]
            rank_idx = t * 7 + k
            
            for (row, col), coeff in x_k.items():
                if coeff != 0:
                    U[get_U_index(row, col)][rank_idx] += coeff
            
            for (row, col), coeff in y_k.items():
                if coeff != 0:
                    V[get_V_index(row, col)][rank_idx] += coeff
            
            for (pr, pc, sub_coeff) in sub_contrib[k]:
                if (pr, pc) in ct:
                    for (out_row, out_col, block_coeff) in ct[(pr, pc)]:
                        w_val = sub_coeff * block_coeff
                        if w_val != 0:
                            W[get_W_index(out_row, out_col)][rank_idx] += w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
