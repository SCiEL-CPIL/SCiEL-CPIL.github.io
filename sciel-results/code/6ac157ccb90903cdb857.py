import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    start_time = time.time()
    deadline = start_time + 145.0

    # Rank 28 decomposition using Strassen on 2x2 blocks
    # A is 2x4, B is 4x4.
    # Split B into 4 blocks of 2x2: B00, B01, B10, B11.
    # Split A into 2 columns of 2x2: A0, A1.
    # C0 = A0 * B00 + A1 * B10
    # C1 = A0 * B01 + A1 * B11
    
    # We use 7 Strassen multiplications for A0*B00, 7 for A1*B10, 
    # 7 for A0*B01, 7 for A1*B11. Total 28.
    
    rank = 28
    U = np.zeros((n * m, rank)) # 8 x 28
    V = np.zeros((m * p, rank)) # 16 x 28
    W = np.zeros((p * n, rank)) # 8 x 28

    # Strassen terms for 2x2 multiplication
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    
    # Indices for A (2x2 block):
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # Indices for B (2x2 block):
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # We need to map these to the global indices.
    # A global indices: i in [0,1], j in [0,3].
    # For A0 (left half of A): j in [0,1].
    # For A1 (right half of A): j in [2,3].
    # B global indices: j in [0,3], k in [0,3].
    # For B00: j in [0,1], k in [0,1].
    # For B01: j in [0,1], k in [2,3].
    # For B10: j in [2,3], k in [0,1].
    # For B11: j in [2,3], k in [2,3].
    
    # C global indices: i in [0,1], k in [0,3].
    # For C0 (left half of C): k in [0,1].
    # For C1 (right half of C): k in [2,3].
    
    # Let's define the 7 Strassen terms for a generic 2x2 * 2x2.
    # Each term is a rank-1 tensor (u, v, w).
    # u is 2x2 (flattened to 4), v is 2x2 (flattened to 4), w is 2x2 (flattened to 4).
    # But our U, V, W are global vectors.
    
    strassen_terms = [
        # M1: (a+d)(e+h) -> contributes to C00, C11
        # u: a=1, d=1. v: e=1, h=1. w: C00=1, C11=1.
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # M2: (c+d)e -> contributes to C10, -C11
        # u: c=1, d=1. v: e=1. w: C10=1, C11=-1.
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        # M3: a(f-h) -> contributes to C01, C11
        # u: a=1. v: f=1, h=-1. w: C01=1, C11=1.
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        # M4: d(g-e) -> contributes to C00, C10
        # u: d=1. v: g=1, e=-1. w: C00=1, C10=1.
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        # M5: (a+b)h -> contributes to -C00, C01
        # u: a=1, b=1. v: h=1. w: C00=-1, C01=1.
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        # M6: (c-a)(e+f) -> contributes to C11
        # u: c=1, a=-1. v: e=1, f=1. w: C11=1.
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # M7: (b-d)(g+h) -> contributes to C00
        # u: b=1, d=-1. v: g=1, h=1. w: C00=1.
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    # We have 4 products to compute:
    # P1 = A0 * B00 -> contributes to C0
    # P2 = A1 * B10 -> contributes to C0
    # P3 = A0 * B01 -> contributes to C1
    # P4 = A1 * B11 -> contributes to C1
    
    # For each product, we apply the 7 Strassen terms.
    # Total 4 * 7 = 28 terms.
    
    col = 0
    for prod_idx in range(4):
        if time.time() > deadline:
            break
            
        # Determine A block, B block, C block offsets
        if prod_idx == 0:
            # A0 (j=0,1), B00 (j=0,1, k=0,1), C0 (k=0,1)
            a_j_off = 0
            b_j_off = 0
            b_k_off = 0
            c_k_off = 0
        elif prod_idx == 1:
            # A1 (j=2,3), B10 (j=2,3, k=0,1), C0 (k=0,1)
            a_j_off = 2
            b_j_off = 2
            b_k_off = 0
            c_k_off = 0
        elif prod_idx == 2:
            # A0 (j=0,1), B01 (j=0,1, k=2,3), C1 (k=2,3)
            a_j_off = 0
            b_j_off = 0
            b_k_off = 2
            c_k_off = 2
        else:
            # A1 (j=2,3), B11 (j=2,3, k=2,3), C1 (k=2,3)
            a_j_off = 2
            b_j_off = 2
            b_k_off = 2
            c_k_off = 2
            
        for term in strassen_terms:
            if time.time() > deadline:
                break
            
            # Fill U
            for (local_idx, coeff) in term['U']:
                i = local_idx // 2
                j = local_idx % 2
                global_i = i
                global_j = j + a_j_off
                U[global_i * m + global_j, col] += coeff
                
            # Fill V
            for (local_idx, coeff) in term['V']:
                j = local_idx // 2
                k = local_idx % 2
                global_j = j + b_j_off
                global_k = k + b_k_off
                V[global_j * p + global_k, col] += coeff
                
            # Fill W
            for (local_idx, coeff) in term['W']:
                k = local_idx // 2
                i = local_idx % 2
                global_k = k + c_k_off
                global_i = i
                W[global_k * n + global_i, col] += coeff
                
            col += 1
            
    # Verify
    def verify(U_mat, V_mat, W_mat, rank_val):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank_val):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
        
    if not verify(U, V, W, rank):
        # Fallback to naive rank 32 if something went wrong
        rank = 32
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        if not verify(U, V, W, rank):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
