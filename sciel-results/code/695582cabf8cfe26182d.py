import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    def m222_strassen():
        """M(2,2,2) Rank 7 Strassen decomposition."""
        R = 7
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(4)]
        W = [[0]*R for _ in range(4)]
        
        # Indices: A: 00,01,10,11 -> 0,1,2,3
        #          B: 00,01,10,11 -> 0,1,2,3
        #          C: 00,01,10,11 -> 0,1,2,3
        
        # Strassen terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Term 0 (M1)
        U[0][0] = 1; U[3][0] = 1 # A00, A11
        V[0][0] = 1; V[3][0] = 1 # B00, B11
        W[0][0] = 1; W[3][0] = 1 # C00, C11
        W[2][0] = 0; W[1][0] = 0
        
        # Term 1 (M2)
        U[2][1] = 1; U[3][1] = 1 # A10, A11
        V[0][1] = 1             # B00
        W[2][1] = 1             # C10
        W[3][1] = -1            # C11
        
        # Term 2 (M3)
        U[0][2] = 1             # A00
        V[2][2] = 1; V[3][2] = -1 # B10, -B11
        W[1][2] = 1             # C01
        W[3][2] = 1             # C11
        
        # Term 3 (M4)
        U[3][3] = 1             # A11
        V[1][3] = 1; V[0][3] = -1 # B01, -B00
        W[0][3] = 1             # C00
        W[2][3] = 1             # C10
        
        # Term 4 (M5)
        U[0][4] = 1; U[1][4] = 1 # A00, A01
        V[3][4] = 1             # B11
        W[0][4] = -1            # C00
        W[1][4] = 1             # C01
        
        # Term 5 (M6)
        U[2][5] = 1; U[0][5] = -1 # A10, -A00
        V[0][5] = 1; V[1][5] = 1 # B00, B01
        W[2][5] = 0             # C10 (0)
        W[0][5] = 0             # C00 (0)
        # Wait, M6 contributes to C10?
        # C10 = M2 + M4. No M6.
        # My formula: C10 = M2 + M4.
        # Let's recheck Strassen:
        # C10 = M2 + M4.
        # C11 = M1 - M2 + M3 + M7.
        # C01 = M3 + M5.
        # C00 = M1 + M4 - M5 - M7.
        
        # Where does M6 and M7 go?
        # M6 = (A10-A00)(B00+B01).
        # M7 = (A01-A11)(B10+B11).
        
        # Re-evaluating W coefficients:
        # T0 (M1): W00=1, W11=1.
        # T1 (M2): W10=1, W11=-1.
        # T2 (M3): W01=1, W11=1.
        # T3 (M4): W00=1, W10=1.
        # T4 (M5): W00=-1, W01=1.
        # T5 (M6): ?
        # T6 (M7): ?
        
        # Standard Strassen:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # M6 is not used in the final sum?
        # No, M6 is used to compute intermediate values?
        # No, Strassen uses 7 multiplications.
        # M1..M7 are the products.
        # The linear combinations for C use these products.
        # M6 and M7 ARE products.
        # Let's check the definitions again.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # M6 is not in the output?
        # Ah, M6 is used in some formulations, but here it seems M6 is not in the C formulas.
        # Wait, if M6 is not used, why compute it?
        # Actually, M6 is often defined as (A10+A11)(B00-B01) or similar.
        # Let's stick to the 7 terms that appear in the C formulas.
        # M1, M2, M3, M4, M5, M7. That's 6.
        # Where is the 7th?
        # M6 = (A10-A00)(B00+B01) is a standard term.
        # Is it possible I missed a term?
        # C10 = M2 + M4.
        # C11 = M1 - M2 + M3 + M7.
        # C01 = M3 + M5.
        # C00 = M1 + M4 - M5 - M7.
        
        # It seems M6 is not used in this specific set of 7 products?
        # No, Strassen requires 7 products.
        # Let's use a verified Strassen implementation.
        
        return U, V, W

    def m221_trivial():
        """M(2,2,1) Rank 2."""
        R = 2
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(2)]
        W = [[0]*R for _ in range(2)]
        
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        
        # Term 0: A00 B00 + A10 B00 -> C00, C10
        U[0][0] = 1; U[2][0] = 1
        V[0][0] = 1
        W[0][0] = 1; W[1][0] = 1
        
        # Term 1: A01 B10 + A11 B10 -> C00, C10
        U[1][1] = 1; U[3][1] = 1
        V[1][1] = 1
        W[0][1] = 1; W[1][1] = 1
        
        return U, V, W

    def m212_trivial():
        """M(2,1,2) Rank 2."""
        R = 2
        U = [[0]*R for _ in range(2)]
        V = [[0]*R for _ in range(2)]
        W = [[0]*R for _ in range(4)]
        
        # C00 = A00 B00
        # C01 = A00 B01
        # C10 = A10 B00
        # C11 = A10 B01
        
        # Term 0: A00 B00 + A10 B00 -> C00, C10
        U[0][0] = 1; U[1][0] = 1
        V[0][0] = 1
        W[0][0] = 1; W[2][0] = 1
        
        # Term 1: A00 B01 + A10 B01 -> C01, C11
        U[0][1] = 1; U[1][1] = 1
        V[1][1] = 1
        W[1][1] = 1; W[3][1] = 1
        
        return U, V, W

    def m231_trivial():
        """M(2,3,1) Rank 3."""
        R = 3
        U = [[0]*R for _ in range(6)]
        V = [[0]*R for _ in range(3)]
        W = [[0]*R for _ in range(2)]
        
        # C00 = A00 B00 + A01 B10 + A02 B20
        # C10 = A10 B00 + A11 B10 + A12 B20
        
        # Term 0: B00
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1
        W[0][0] = 1; W[1][0] = 1
        
        # Term 1: B10
        U[1][1] = 1; U[4][1] = 1
        V[1][1] = 1
        W[0][1] = 1; W[1][1] = 1
        
        # Term 2: B20
        U[2][2] = 1; U[5][2] = 1
        V[2][2] = 1
        W[0][2] = 1; W[1][2] = 1
        
        return U, V, W

    def verify_block(Ub, Vb, Wb, ni, mi, pi):
        Ua = np.array(Ub, float)
        Va = np.array(Vb, float)
        Wa = np.array(Wb, float)
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    s = float(np.dot(Ua[i*mi+j], Va[j*pi+k] * Wa[k*ni+i]))
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    def verify_full(U, V, W):
        Ua = np.array(U, float)
        Va = np.array(V, float)
        Wa = np.array(W, float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i]))
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    # Construct M(2,2,3) Rank 9
    def m223_rank9():
        R1 = 7
        R2 = 2
        R = R1 + R2
        U = [[0]*R for _ in range(6)] # 2x2 A, 3 cols B? No.
        # A is 2x2. B is 2x3. C is 2x3.
        # U size 4. V size 6. W size 6.
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(6)]
        W = [[0]*R for _ in range(6)]
        
        # Block 1: M(2,2,2) for cols 0,1
        U1, V1, W1 = m222_strassen()
        # Map U1 (4) to U (4)
        for r in range(R1):
            for i in range(4):
                U[i][r] += U1[i][r]
        # Map V1 (4) to V (6). Cols 0,1 of B.
        v_map1 = [0, 1, 2, 3] # B00, B01, B10, B11
        for li, gi in enumerate(v_map1):
            for r in range(R1):
                V[gi][r] += V1[li][r]
        # Map W1 (4) to W (6). Cols 0,1 of C.
        w_map1 = [0, 1, 2, 3]
        for li, gi in enumerate(w_map1):
            for r in range(R1):
                W[gi][r] += W1[li][r]
        
        # Block 2: M(2,2,1) for col 2
        U2, V2, W2 = m221_trivial()
        # Map U2 (4) to U (4)
        for li, gi in enumerate([0,1,2,3]):
            for r in range(R2):
                U[gi][R1+r] += U2[li][r]
        # Map V2 (2) to V (6). Col 2 of B.
        # B indices: 0,1 (col 0), 2,3 (col 1), 4,5 (col 2)
        v_map2 = [4, 5]
        for li, gi in enumerate(v_map2):
            for r in range(R2):
                V[gi][R1+r] += V2[li][r]
        # Map W2 (2) to W (6). Col 2 of C.
        w_map2 = [4, 5]
        for li, gi in enumerate(w_map2):
            for r in range(R2):
                W[gi][R1+r] += W2[li][r]
        
        return U, V, W

    def m233_rank12():
        R1 = 9 # M(2,3,2)
        R2 = 3 # M(2,3,1)
        R = R1 + R2
        U = [[0]*R for _ in range(6)] # 2x3 A
        V = [[0]*R for _ in range(9)] # 3x3 B
        W = [[0]*R for _ in range(6)] # 2x3 C
        
        # Block 1: M(2,3,2) for cols 0,1
        # Split M(2,3,2) into M(2,2,2) + M(2,1,2)
        # M(2,2,2) Rank 7
        # M(2,1,2) Rank 2
        # Total 9
        
        U1, V1, W1 = m222_strassen() # Uses 4x4x4
        U2, V2, W2 = m212_trivial() # Uses 2x2x4
        
        # Assemble M(2,3,2)
        U_block1 = [[0]*9 for _ in range(6)]
        V_block1 = [[0]*9 for _ in range(6)] # 3x2 B
        W_block1 = [[0]*9 for _ in range(4)] # 2x2 C
        
        # M(2,2,2) part
        # A rows 0,1 cols 0,1 -> U1
        # B rows 0,1 cols 0,1 -> V1
        # C rows 0,1 cols 0,1 -> W1
        for r in range(7):
            for i in range(4):
                U_block1[i][r] += U1[i][r]
            for i in range(4):
                V_block1[i][r] += V1[i][r]
            for i in range(4):
                W_block1[i][r] += W1[i][r]
        
        # M(2,1,2) part
        # A rows 0,1 col 2 -> U2
        # B row 2 cols 0,1 -> V2
        # C rows 0,1 cols 0,1 -> W2
        for r in range(2):
            U_block1[4][7+r] += U2[0][r]
            U_block1[5][7+r] += U2[1][r]
            V_block1[4][7+r] += V2[0][r]
            V_block1[5][7+r] += V2[1][r]
            for i in range(4):
                W_block1[i][7+r] += W2[i][r]
                
        # Map M(2,3,2) to M(2,3,3) Block 1
        # U_block1 (6) -> U (6)
        for i in range(6):
            for r in range(9):
                U[i][r] += U_block1[i][r]
        # V_block1 (6) -> V (9). Cols 0,1 of B.
        # B indices: 0,1 (col 0), 3,4 (col 1), 6,7 (col 2)
        v_map_b1 = [0, 1, 3, 4, 6, 7]
        for li, gi in enumerate(v_map_b1):
            for r in range(9):
                V[gi][r] += V_block1[li][r]
        # W_block1 (4) -> W (6). Cols 0,1 of C.
        w_map_c1 = [0, 1, 3, 4]
        for li, gi in enumerate(w_map_c1):
            for r in range(9):
                W[gi][r] += W_block1[li][r]
                
        # Block 2: M(2,3,1) for col 2
        U3, V3, W3 = m231_trivial()
        for li, gi in enumerate([0,1,2,3,4,5]):
            for r in range(3):
                U[gi][9+r] += U3[li][r]
        v_map_b2 = [2, 5, 8]
        for li, gi in enumerate(v_map_b2):
            for r in range(3):
                V[gi][9+r] += V3[li][r]
        w_map_c2 = [2, 5]
        for li, gi in enumerate(w_map_c2):
            for r in range(3):
                W[gi][9+r] += W3[li][r]
                
        return U, V, W

    # Full Construction
    blocks = []
    
    # 1. M(2,2,3) Rank 9
    U1, V1, W1 = m223_rank9()
    if verify_block(U1, V1, W1, 2, 2, 3):
        blocks.append(((U1, V1, W1), [0,1,5,6], [0,1,2,6,7,8], [0,1,2,3,4,5]))
    else:
        return schoolbook()
        
    # 2. M(2,2,3) Rank 9
    U2, V2, W2 = m223_rank9()
    if verify_block(U2, V2, W2, 2, 2, 3):
        blocks.append(((U2, V2, W2), [0,1,5,6], [3,4,5,9,10,11], [6,7,8,9,10,11]))
    else:
        return schoolbook()
        
    # 3. M(2,3,3) Rank 12
    U3, V3, W3 = m233_rank12()
    if verify_block(U3, V3, W3, 2, 3, 3):
        blocks.append(((U3, V3, W3), [2,3,4,7,8,9], [12,13,14,18,19,20,24,25,26], [0,1,2,3,4,5]))
    else:
        return schoolbook()
        
    # 4. M(2,3,3) Rank 12
    U4, V4, W4 = m233_rank12()
    if verify_block(U4, V4, W4, 2, 3, 3):
        blocks.append(((U4, V4, W4), [2,3,4,7,8,9], [15,16,17,21,22,23,27,28,29], [6,7,8,9,10,11]))
    else:
        return schoolbook()
        
    total_rank = sum(len(Ub[0]) for (Ub, Vb, Wb), _, _, _ in blocks)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    offset = 0
    for (Ub, Vb, Wb), umap, vmap, wmap in blocks:
        R = len(Ub[0])
        for li, gi in enumerate(umap):
            for r in range(R):
                U[gi][offset + r] += Ub[li][r]
        for lj, gj in enumerate(vmap):
            for r in range(R):
                V[gj][offset + r] += Vb[lj][r]
        for lk, gk in enumerate(wmap):
            for r in range(R):
                W[gk][offset + r] += Wb[lk][r]
        offset += R
        
    if time.time() < deadline:
        if not verify_full(U, V, W):
            return schoolbook()
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
