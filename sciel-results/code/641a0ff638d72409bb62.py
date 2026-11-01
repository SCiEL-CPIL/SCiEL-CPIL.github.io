"""
Rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.

Strategy:
1. Build the 53-rank Strassen-block construction (7 blocks of rank 7 + 1 schoolbook block of rank 4).
2. Try to find a rank-9 decomposition for 2x2*2x3 and use 5 such blocks for rank 45.
3. Verify each candidate; return the best valid one.
"""
import numpy as np
import time

def construct():
    n, m, p = 2, 2, 15
    start_time = time.time()
    deadline = start_time + 120

    # Nonzero entries of the multiplication tensor
    nonzeros = []
    for i in range(2):
        for j in range(2):
            for k in range(15):
                a = i * 2 + j
                b = j * 15 + k
                c = 2 * k + i
                nonzeros.append((a, b, c))

    def verify(U, V, W):
        """Verify exact decomposition using numpy einsum."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        for (a, b, c) in nonzeros:
            if abs(T[a, b, c] - 1.0) > 1e-9:
                return False
        for a in range(4):
            i1, j1 = a // 2, a % 2
            for b in range(30):
                j2, k1 = b // 15, b % 15
                for c in range(30):
                    k2, i2 = c // 2, c % 2
                    if not (i1 == i2 and j1 == j2):
                        if abs(T[a, b, c]) > 1e-9:
                            return False
        return True

    # --- Build 53-rank Strassen construction ---
    def make_strassen_block(k0, k1, col_start, U, V, W):
        col = col_start
        # P1: (A00+A11)(B00+B11)
        U[0, col] = 1; U[3, col] = 1
        V[k0, col] = 1; V[15+k1, col] = 1
        W[2*k0, col] = 1; W[2*k1+1, col] = 1
        col += 1
        # P2: (A10+A11)B00
        U[2, col] = 1; U[3, col] = 1
        V[k0, col] = 1
        W[2*k0, col] = 1; W[2*k1, col] = 1
        col += 1
        # P3: A00(B11-B00)
        U[0, col] = 1
        V[15+k1, col] = 1; V[k0, col] = -1
        W[2*k1+1, col] = 1; W[2*k1, col] = 1
        col += 1
        # P4: A11(B10-B01)
        U[3, col] = 1
        V[15+k0, col] = 1; V[15+k1, col] = -1
        W[2*k0, col] = 1; W[2*k1+1, col] = 1
        col += 1
        # P5: (A00+A01)B11
        U[0, col] = 1; U[1, col] = 1
        V[15+k1, col] = 1
        W[2*k0, col] = -1; W[2*k1+1, col] = 1
        col += 1
        # P6: (A10-A00)(B00+B11)
        U[2, col] = 1; U[0, col] = -1
        V[k0, col] = 1; V[15+k1, col] = 1
        W[2*k1+1, col] = 1
        col += 1
        # P7: (A01-A11)(B10+B01)
        U[1, col] = 1; U[3, col] = -1
        V[15+k0, col] = 1; V[15+k1, col] = 1
        W[2*k0, col] = 1
        col += 1
        return col

    def make_schoolbook_block(k, col_start, U, V, W):
        col = col_start
        U[0, col] = 1; V[k, col] = 1; W[2*k, col] = 1; col += 1
        U[1, col] = 1; V[15+k, col] = 1; W[2*k, col] = 1; col += 1
        U[2, col] = 1; V[k, col] = 1; W[2*k+1, col] = 1; col += 1
        U[3, col] = 1; V[15+k, col] = 1; W[2*k+1, col] = 1
        return col

    rank_53 = 7 * 7 + 4
    U53 = np.zeros((4, rank_53))
    V53 = np.zeros((30, rank_53))
    W53 = np.zeros((30, rank_53))

    col = 0
    for block_idx in range(7):
        k0 = 2 * block_idx
        k1 = 2 * block_idx + 1
        col = make_strassen_block(k0, k1, col, U53, V53, W53)
    col = make_schoolbook_block(14, col, U53, V53, W53)

    # Verify 53-rank construction
    if verify(U53, V53, W53):
        best_rank = rank_53
        best_result = {
            "U": U53.tolist(),
            "V": V53.tolist(),
            "W": W53.tolist()
        }
    else:
        # Fallback: pure schoolbook
        best_rank = 60
        U_sb = np.zeros((4, 60))
        V_sb = np.zeros((30, 60))
        W_sb = np.zeros((30, 60))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(15):
                    U_sb[i*2+j, col] = 1
                    V_sb[j*15+k, col] = 1
                    W_sb[k*2+i, col] = 1
                    col += 1
        best_result = {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }

    # --- Try to find a rank-9 decomposition for 2x2*2x3 ---
    # If found, we can build a rank-45 decomposition for 2x2*2x15.
    def try_rank_9_2x2_2x3():
        """
        Construct a rank-9 decomposition for 2x2 * 2x3.
        U: 4x9, V: 6x9, W: 6x9
        """
        U = np.zeros((4, 9))
        V = np.zeros((6, 9))
        W = np.zeros((6, 9))
        
        # Strassen terms for k=0,1
        # P1: (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[4, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # P2: (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[0, 1] = 1; W[3, 1] = 1
        # P3: A00(B11-B00)
        U[0, 2] = 1
        V[4, 2] = 1; V[0, 2] = -1
        W[3, 2] = 1; W[0, 2] = 1
        # P4: A11(B10-B01)
        U[3, 3] = 1
        V[3, 3] = 1; V[1, 3] = -1
        W[0, 3] = 1; W[3, 3] = 1
        # P5: (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[4, 4] = 1
        W[0, 4] = -1; W[3, 4] = 1
        # P6: (A10-A00)(B00+B11)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[4, 5] = 1
        W[3, 5] = 1
        # P7: (A01-A11)(B10+B01)
        U[1, 6] = 1; U[3, 6] = -1
        V[3, 6] = 1; V[1, 6] = 1
        W[0, 6] = 1
        
        # Terms for k=2:
        # C02 = A00*B02 + A01*B12
        # C12 = A10*B02 + A11*B12
        # P8: (A00+A10)*B02
        U[0, 7] = 1; U[2, 7] = 1
        V[2, 7] = 1
        W[4, 7] = 1; W[5, 7] = 1
        # P9: (A01+A11)*B12
        U[1, 8] = 1; U[3, 8] = 1
        V[5, 8] = 1
        W[4, 8] = 1; W[5, 8] = 1
        
        # Verify
        n2, m2, p2 = 2, 2, 3
        nonzeros2 = []
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    a = i * m2 + j
                    b = j * p2 + k
                    c = k * n2 + i
                    nonzeros2.append((a, b, c))
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        for (a, b, c) in nonzeros2:
            if abs(T[a, b, c] - 1.0) > 1e-9:
                return None
        for a in range(4):
            i1, j1 = a // 2, a % 2
            for b in range(6):
                j2, k1 = b // 3, b % 3
                for c in range(6):
                    k2, i2 = c // 2, c % 2
                    if not (i1 == i2 and j1 == j2):
                        if abs(T[a, b, c]) > 1e-9:
                            return None
        return U, V, W

    if time.time() < deadline - 30:
        r9 = try_rank_9_2x2_2x3()
        if r9 is not None:
            U3, V3, W3 = r9
            rank_total = 45
            U_full = np.zeros((4, rank_total))
            V_full = np.zeros((30, rank_total))
            W_full = np.zeros((30, rank_total))
            
            for block in range(5):
                k_offset = block * 3
                for r in range(9):
                    col = block * 9 + r
                    U_full[:, col] = U3[:, r]
                    for j in range(2):
                        for k_local in range(3):
                            V_full[j * 15 + (k_offset + k_local), col] = V3[j * 3 + k_local, r]
                    for k_local in range(3):
                        for i in range(2):
                            W_full[(k_offset + k_local) * 2 + i, col] = W3[k_local * 2 + i, r]
            
            if verify(U_full, V_full, W_full):
                return {
                    "U": U_full.tolist(),
                    "V": V_full.tolist(),
                    "W": W_full.tolist()
                }

    return best_result
