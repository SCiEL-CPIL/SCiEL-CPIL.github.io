import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def _strassen_3x3():
    # 23-term rank-23 decomposition of 3x3 matrix multiplication.
    # Coefficients from Strassen 1969 / standard references.
    # Each term t: U[:,t] (3), V[:,t] (3), W[:,t] (3)
    # C = A*B, C[i,k] = sum_t U[i,t]*V[k,t]*W[i+k,t]  (with appropriate indexing)
    # Actually for tensor: T[i,j,k] = sum_t U[i,t] V[k,t] W[i+k mod ?,t]
    # Standard form: T_{ijk} = sum_t a_{it} b_{kt} c_{(i+k) mod 3, t}? No.
    # The tensor is T_{i,j,k} where A is i x j, B is j x k, C is i x k.
    # T[i,j,k] = 1 iff we multiply row i of A by column k of B, summing over j.
    # So T[i,j,k] = 1 for all valid (i,j,k).
    # Decomposition: T[i,j,k] = sum_t U[i,t] V[k,t] W[t_idx]
    # Wait, the convention is:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For 3x3x3: U is 9xR, V is 9xR, W is 9xR
    # T[i,j,k] = sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]

    # The 23 terms for 3x3 Strassen:
    # I'll use the well-known decomposition.
    # Reference: https://en.wikipedia.org/wiki/Strassen_algorithm#Extension_to_arbitrary_dimensions
    # Or the original paper. Let me use the standard 23-term set.
    
    # Actually, let me use a verified set. The rank of 3x3 is 23.
    # Here are the 23 terms (U, V, W) where each is a vector of length 9
    # for U[i*3+j], V[j*3+k], W[k*3+i]
    
    # This is complex to hardcode correctly from memory. Let me instead
    # use a constructive approach or fall back to a known-good smaller rank.
    
    # Actually, for safety and correctness, let me use a different strategy.
    # The rank of 3x4x7 matrix multiplication.
    # Known results: rank(M_{3,4,7}) <= 3*4*7 = 84 (schoolbook)
    # Using block decomposition: 3x4 = [3x3 | 3x1], 4x7 = [3x7 | 1x7; ...]
    # This is getting complicated. Let me use a simpler approach.
    
    # For 3x4x7:
    # Split 3x4 into 3x3 + 3x1 (column split)
    # Split 4x7 into [3x7; 1x7] (row split)
    # C = A_3x3 * B_3x7 + A_3x1 * B_1x7
    # Rank(M_{3,3,7}) + Rank(M_{3,1,7})
    # M_{3,1,7}: rank 3*1*7 = 21 (schoolbook, but actually rank = 3*7 = 21 since it's just outer products)
    #   Actually M_{3,1,7}: A is 3x1, B is 1x7, C is 3x7. C[i,k] = A[i,0]*B[0,k].
    #   This is rank 3*7 = 21? No, the tensor rank. T[i,j,k] with j fixed=0.
    #   T[i,0,k] = 1. Rank = 3*7 = 21. Yes.
    # M_{3,3,7}: Split 7 into 3+3+1? Or use known bounds.
    #   M_{3,3,3} has rank 23.
    #   M_{3,3,7} <= M_{3,3,3} + M_{3,3,3} + M_{3,3,1}
    #   M_{3,3,1}: A 3x3, B 3x1, C 3x1. C[i,0] = sum_j A[i,j]*B[j,0].
    #   Rank of M_{3,3,1} = 3*3 = 9? T[i,j,0] = 1. Rank = 9.
    #   So M_{3,3,7} <= 23 + 23 + 9 = 55.
    # Total: 55 + 21 = 76.
    
    # So rank(M_{3,4,7}) <= 76.
    
    # Now I need to construct this 76-rank decomposition.
    # Let me build it piece by piece.
    
    # Part 1: M_{3,3,7} using 3x3 blocks of the 7 columns
    # B is 3x7, split columns into 3+3+1
    # C1 = A * B_1 (3x3), C2 = A * B_2 (3x3), C3 = A * B_3 (3x1)
    
    # For M_{3,3,3}: rank 23
    # I need the 23-term decomposition. Let me use a known one.
    
    # The following is the 23-term decomposition for 3x3 from the literature.
    # Each term has (U, V, W) triples where U, V, W are in {0, 1, -1}
    # indexed by (row, col) pairs.
    
    # U index: i*3 + j (i = output row, j = input col of A)
    # V index: j*3 + k (j = input col of A / row of B, k = output col of B)  
    # W index: k*3 + i (k = output col, i = output row)
    
    # Actually wait, for the sub-problem M_{3,3,3}:
    # A is 3x3, B is 3x3, C is 3x3
    # U[i*3+j, t], V[j*3+k, t], W[k*3+i, t]
    
    # The 23 terms (from Strassen 1969, verified against known sources):
    terms_3x3 = [
        # (U_entries, V_entries, W_entries)
        # Each is a dict mapping index to value, or a list of (index, value)
        # Let me use the standard 23-term set.
        # I'll encode as: for each term, lists of (u_idx, v_idx, w_idx, coeff)
        # where T += coeff * e_{u_idx} ⊗ e_{v_idx} ⊗ e_{w_idx}
    ]
    
    # This is error-prone from memory. Let me use a different approach.
    # I'll use the fact that for small matrices, I can use known decompositions.
    # 
    # Alternative: use rank 76 construction but verify it works.
    # If I can't get the exact 23 terms right, I'll use a slightly higher rank
    # that I can construct more reliably.
    
    # Let me try a different split that avoids needing 3x3:
    # 3x4x7: split 3x4 as 2x4 + 1x4 (row split)
    # C = A_1 (2x4) * B (4x7) + A_2 (1x4) * B (4x7)
    # M_{2,4,7} + M_{1,4,7}
    # M_{1,4,7}: A 1x4, B 4x7, C 1x7. Rank = 4*7 = 28.
    # M_{2,4,7}: split 4 into 2+2: M_{2,2,7} + M_{2,2,7}
    #   M_{2,2,7}: split 7 into 2+2+2+1: 3*M_{2,2,2} + M_{2,2,1}
    #   M_{2,2,2}: rank 7 (Strassen)
    #   M_{2,2,1}: rank 4
    #   M_{2,2,7} <= 3*7 + 4 = 25
    # M_{2,4,7} <= 25 + 25 = 50
    # Total: 50 + 28 = 78.
    
    # 78 > 76, so the 3x3 approach is better. Let me try to get the 3x3 terms right.
    
    # Actually, let me just use rank 84 (schoolbook) but with a better optimization
    # that might find a lower rank. Or use a known correct lower bound construction.
    
    # Let me try yet another approach: use the 2x2 Strassen (rank 7) which I'm
    # more confident about, and build up.
    
    # 3x4x7:
    # Split rows of 3 into 2+1
    # C = [A_1 (2x4) * B; A_2 (1x4) * B]
    # Rank(M_{2,4,7}) + Rank(M_{1,4,7})
    # M_{1,4,7} = 28 (trivial)
    # M_{2,4,7}: split cols of 4 into 2+2
    #   = M_{2,2,7} + M_{2,2,7}
    #   M_{2,2,7}: split 7 into 2+2+2+1
    #     = 3*M_{2,2,2} + M_{2,2,1}
    #     M_{2,2,2} = 7 (Strassen 2x2)
    #     M_{2,2,1} = 4
    #     M_{2,2,7} = 21 + 4 = 25
    #   M_{2,4,7} = 50
    # Total = 50 + 28 = 78.
    
    # Let me go with 78 using well-known 2x2 Strassen.
    
    def _strassen_2x2():
        # Rank-7 decomposition of 2x2 matrix multiplication
        # A is 2x2, B is 2x2, C is 2x2
        # U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
        # 7 terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01+A11)(B01+B11)
        #
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M6
        # C11 = M1 - M3 + M4 - M6
        #
        # U[i*2+j]: i=0,1 row; j=0,1 col
        # V[j*2+k]: j=0,1 row of B (=col of A); k=0,1 col of B
        # W[k*2+i]: k=0,1 col of C; i=0,1 row of C
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00 -> idx 0, A11 -> idx 3
        U[0, 0] = 1; U[3, 0] = 1
        # V: B00 -> idx 0, B11 -> idx 3
        V[0, 0] = 1; V[3, 0] = 1
        # W: contributes to C00 (+1), C11 (+1)
        # C00: k=0,i=0 -> idx 0; C11: k=1,i=1 -> idx 3
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 2: (A10+A11)B00
        # U: A10 -> idx 2, A11 -> idx 3
        U[2, 1] = 1; U[3, 1] = 1
        # V: B00 -> idx 0
        V[0, 1] = 1
        # W: C10: k=0,i=1 -> idx 1
        W[1, 1] = 1
        
        # Term 3: A00(B10-B11)
        # U: A00 -> idx 0
        U[0, 2] = 1
        # V: B10 -> idx 2, B11 -> idx 3
        V[2, 2] = 1; V[3, 2] = -1
        # W: C01: k=1,i=0 -> idx 2
        W[2, 2] = 1
        
        # Term 4: A11(B01-B00)
        # U: A11 -> idx 3
        U[3, 3] = 1
        # V: B01 -> idx 1, B00 -> idx 0
        V[1, 3] = 1; V[0, 3] = -1
        # W: C00: k=0,i=0 -> idx 0 (-1), C11: k=1,i=1 -> idx 3 (+1)
        W[0, 3] = -1; W[3, 3] = 1
        
        # Term 5: (A00+A01)B11
        # U: A00 -> idx 0, A01 -> idx 1
        U[0, 4] = 1; U[1, 4] = 1
        # V: B11 -> idx 3
        V[3, 4] = 1
        # W: C00: k=0,i=0 -> idx 0 (+1)
        W[0, 4] = 1
        
        # Term 6: (A10-A00)(B00+B10)
        # U: A10 -> idx 2, A00 -> idx 0
        U[2, 5] = 1; U[0, 5] = -1
        # V: B00 -> idx 0, B10 -> idx 2
        V[0, 5] = 1; V[2, 5] = 1
        # W: C10: k=0,i=1 -> idx 1 (+1)
        W[1, 5] = 1
        
        # Term 7: (A01+A11)(B01+B11)
        # U: A01 -> idx 1, A11 -> idx 3
        U[1, 6] = 1; U[3, 6] = 1
        # V: B01 -> idx 1, B11 -> idx 3
        V[1, 6] = 1; V[3, 6] = 1
        # W: C01: k=1,i=0 -> idx 2 (+1), C11: k=1,i=1 -> idx 3 (-1)
        W[2, 6] = 1; W[3, 6] = -1
        
        return U, V, W
    
    def _trivial_1xp(p):
        # M_{1,1,p}: A 1x1, B 1xp, C 1xp. Rank p.
        # U[0, t] = 1, V[t, t] = 1, W[t, t] = 1 for t=0..p-1
        # Wait: U shape (1*1, p) = (1, p), V shape (1*p, p) = (p, p), W shape (p*1, p) = (p, p)
        U = np.zeros((1, p))
        V = np.zeros((p, p))
        W = np.zeros((p, p))
        for t in range(p):
            U[0, t] = 1
            V[t, t] = 1
            W[t, t] = 1
        return U, V, W
    
    def _trivial_n1p(n, p):
        # M_{n,1,p}: A nx1, B 1xp, C nxp. Rank n*p.
        U = np.zeros((n*1, n*p))
        V = np.zeros((1*p, n*p))
        W = np.zeros((p*n, n*p))
        t = 0
        for i in range(n):
            for k in range(p):
                U[i, t] = 1
                V[k, t] = 1
                W[k*n+i, t] = 1
                t += 1
        return U, V, W
    
    def _trivial_n1(n):
        # M_{n,1,1}: A nx1, B 1x1, C nx1. Rank n.
        U = np.zeros((n*1, n))
        V = np.zeros((1*1, n))
        W = np.zeros((1*n, n))
        for i in range(n):
            U[i, i] = 1
            V[0, i] = 1
            W[i, i] = 1
        return U, V, W
    
    # Build M_{2,2,7} = 3*M_{2,2,2} + M_{2,2,1}
    def _m_22_7():
        U = np.zeros((4, 25))
        V = np.zeros((14, 25))
        W = np.zeros((14, 25))
        
        # Three 2x2 blocks for columns 0-1, 2-3, 4-5
        sU, sV, sW = _strassen_2x2()
        for block in range(3):
            offset = block * 7
            U[:, offset:offset+7] = sU
            # V: rows for block: j*7 + k where j=0,1 and k=0,1 (local)
            # Global: j*7 + block*2 + k
            for j in range(2):
                for k in range(2):
                    local_v_idx = j * 2 + k
                    global_v_idx = j * 7 + block * 2 + k
                    V[global_v_idx, offset:offset+7] = sV[local_v_idx]
            # W: cols for block: k*2 + i where k=0,1 (local), i=0,1
            # Global: (block*2 + k) * 2 + i
            for k in range(2):
                for i in range(2):
                    local_w_idx = k * 2 + i
                    global_w_idx = (block * 2 + k) * 2 + i
                    W[global_w_idx, offset:offset+7] = sW[local_w_idx]
        
        # M_{2,2,1} for column 6: A 2x2, B 2x1, C 2x1. Rank 4.
        # T[i,j,0] = 1 for i=0,1, j=0,1
        for i in range(2):
            for j in range(2):
                t = 21 + i * 2 + j
                U[i * 2 + j, t] = 1
                V[j * 7 + 6, t] = 1
                W[6 * 2 + i, t] = 1  # k=6, i
        
        return U, V, W
    
    # Build M_{2,4,7} = 2 * M_{2,2,7} (split cols of A into 2+2)
    # Wait, M_{2,4,7}: A is 2x4, B is 4x7, C is 2x7
    # Split A's columns into 2+2: A = [A1 (2x2) | A2 (2x2)]
    # Split B's rows into 2+2: B = [B1 (2x7); B2 (2x7)]
    # C = A1*B1 + A2*B2
    # Each M_{2,2,7} has rank 25, total rank 50.
    
    def _m_24_7():
        U1, V1, W1 = _m_22_7()
        U2, V2, W2 = _m_22_7()
        
        U = np.zeros((2*4, 50))
        V = np.zeros((4*7, 50))
        W = np.zeros((7*2, 50))
        
        # Part 1: A1 (cols 0-1) * B1 (rows 0-1)
        # U: rows 0-1, cols 0-1 of A -> indices 0*4+0, 0*4+1, 1*4+0, 1*4+1
        U[0:4, 0:25] = U1
        # V: rows 0-1 of B -> indices 0*7+0 through 0*7+6, 1*7+0 through 1*7+6
        for j in range(2):
            for k in range(7):
                local_v = j * 7 + k
                global_v = j * 7 + k
                V[global_v, 0:25] = V1[local_v]
        # W: same
        W[:, 0:25] = W1
        
        # Part 2: A2 (cols 2-3) * B2 (rows 2-3)
        # U: rows 0-1, cols 2-3 of A -> indices 0*4+2, 0*4+3, 1*4+2, 1*4+3
        for i in range(2):
            for j_local in range(2):
                global_u = i * 4 + 2 + j_local
                local_u = i * 2 + j_local
                U[global_u, 25:50] = U1[local_u]
        # V: rows 2-3 of B -> indices 2*7+0 through 2*7+6, 3*7+0 through 3*7+6
        for j in range(2):
            for k in range(7):
                local_v = j * 7 + k
                global_v = (2 + j) * 7 + k
                V[global_v, 25:50] = V1[local_v]
        # W: same
        W[:, 25:50] = W1
        
        return U, V, W
    
    # M_{1,4,7}: A 1x4, B 4x7, C 1x7. Rank 28.
    def _m_14_7():
        U = np.zeros((1*4, 28))
        V = np.zeros((4*7, 28))
        W = np.zeros((7*1, 28))
        t = 0
        for j in range(4):
            for k in range(7):
                U[j, t] = 1
                V[j * 7 + k, t] = 1
                W[k, t] = 1
                t += 1
        return U, V, W
    
    # Now assemble M_{3,4,7} = M_{2,4,7} + M_{1,4,7}
    # Split rows of A into 2+1: A = [A1 (2x4); A2 (1x4)]
    # C = [A1 * B; A2 * B]
    # Total rank: 50 + 28 = 78
    
    U_24, V_24, W_24 = _m_24_7()
    U_14, V_14, W_14 = _m_14_7()
    
    rank = 78
    U = np.zeros((3*4, rank))
    V = np.zeros((4*7, rank))
    W = np.zeros((7*3, rank))
    
    # Part 1: rows 0-1 of A (2x4) * B (4x7) -> rows 0-1 of C
    U[0:8, 0:50] = U_24
    V[:, 0:50] = V_24
    # W: rows 0-1 of C (2x7) -> indices k*2+i for k=0..6, i=0..1
    for k in range(7):
        for i in range(2):
            W[k * 2 + i, 0:50] = W_24[k * 2 + i]
    
    # Part 2: row 2 of A (1x4) * B (4x7) -> row 2 of C
    for j in range(4):
        U[2 * 4 + j, 50:78] = U_14[j]
    V[:, 50:78] = V_14
    for k in range(7):
        W[k * 3 + 2, 50:78] = W_14[k]
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try the rank-78 construction
    try:
        U78, V78, W78 = _strassen_3x3()
        if U78.shape == (nm, 78) and V78.shape == (mp, 78) and W78.shape == (pn, 78):
            # Verify
            max_err = 0.0
            for i in range(nm):
                a, b = divmod(i, m)
                for j in range(mp):
                    c, d = divmod(j, p)
                    k = d * n + a
                    val = np.dot(U78[i, :], V78[j, :] * W78[k, :])
                    if b == c:
                        err = abs(val - 1.0)
                    else:
                        err = abs(val)
                    max_err = max(max_err, err)
            if max_err < 1e-8:
                best_rank = 78
                best_U, best_V, best_W = U78, V78, W78
    except Exception:
        pass
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
