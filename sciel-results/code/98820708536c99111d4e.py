DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition coefficients
    # Terms:
    # 0: (A00+A11)(B00+B11)
    # 1: (A10+A11)B00
    # 2: A00(B01-B11)
    # 3: A11(B10-B00)
    # 4: (A00+A01)B11
    # 5: (A10-A00)(B00+B01)
    # 6: (A01-A11)(B10+B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    def add_strassen(offset, u_idx, v_idx, w_idx):
        # u_idx: [u00, u01, u10, u11]
        # v_idx: [v00, v01, v10, v11]
        # w_idx: [w00, w01, w10, w11]
        
        # Term 0
        U[u_idx[0]][offset+0] += 1
        U[u_idx[3]][offset+0] += 1
        V[v_idx[0]][offset+0] += 1
        V[v_idx[3]][offset+0] += 1
        W[w_idx[0]][offset+0] += 1
        W[w_idx[1]][offset+0] += -1
        W[w_idx[3]][offset+0] += 1
        
        # Term 1
        U[u_idx[2]][offset+1] += 1
        U[u_idx[3]][offset+1] += 1
        V[v_idx[0]][offset+1] += 1
        W[w_idx[2]][offset+1] += 1
        W[w_idx[3]][offset+1] += -1
        
        # Term 2
        U[u_idx[0]][offset+2] += 1
        V[v_idx[1]][offset+2] += 1
        V[v_idx[3]][offset+2] += -1
        W[w_idx[1]][offset+2] += 1
        W[w_idx[3]][offset+2] += 1
        
        # Term 3
        U[u_idx[3]][offset+3] += 1
        V[v_idx[2]][offset+3] += 1
        V[v_idx[0]][offset+3] += -1
        W[w_idx[2]][offset+3] += 1
        W[w_idx[3]][offset+3] += -1
        
        # Term 4
        U[u_idx[0]][offset+4] += 1
        U[u_idx[1]][offset+4] += 1
        V[v_idx[3]][offset+4] += 1
        W[w_idx[1]][offset+4] += 1
        W[w_idx[3]][offset+4] += -1
        
        # Term 5
        U[u_idx[2]][offset+5] += 1
        U[u_idx[0]][offset+5] += -1
        V[v_idx[0]][offset+5] += 1
        V[v_idx[1]][offset+5] += 1
        W[w_idx[3]][offset+5] += 1
        
        # Term 6
        U[u_idx[1]][offset+6] += 1
        U[u_idx[3]][offset+6] += -1
        V[v_idx[2]][offset+6] += 1
        V[v_idx[3]][offset+6] += 1
        W[w_idx[0]][offset+6] += 1
        W[w_idx[2]][offset+6] += 1

    # Block 1: A1 * B11
    # A1: u00=0, u01=1, u10=4, u11=5
    # B11: v00=0, v01=1, v10=6, v11=7
    # C11: w00=0, w01=1, w10=2, w11=3
    add_strassen(0, [0,1,4,5], [0,1,6,7], [0,1,2,3])
    
    # Block 2: A1 * B12
    # A1: [0,1,4,5]
    # B12: B[0:2, 2:4]. v00=2, v01=3, v10=8, v11=9
    # C12: C[0:2, 2:4]. w00=4, w01=5, w10=6, w11=7
    add_strassen(7, [0,1,4,5], [2,3,8,9], [4,5,6,7])
    
    # Block 3: A2 * B21
    # A2: u00=2, u01=3, u10=6, u11=7
    # B21: B[2:4, 0:2]. v00=12, v01=13, v10=18, v11=19
    # C21: C[2:4, 0:2]? No, C is 2x6.
    # C = [C1, C2]. C1 is 2x4.
    # C11 is C[0:2, 0:2]. C12 is C[0:2, 2:4].
    # C21 is C[0:2, 0:2] ? No.
    # A2 is the second half of A.
    # A B1 = A1 B11 + A2 B21 (first 2 cols of C1)
    # So the output of A2 B21 adds to C[0:2, 0:2].
    # So w indices are same as Block 1.
    add_strassen(14, [2,3,6,7], [12,13,18,19], [0,1,2,3])
    
    # Block 4: A2 * B22
    # A2: [2,3,6,7]
    # B22: B[2:4, 2:4]. v00=14, v01=15, v10=20, v11=21
    # C22 adds to C[0:2, 2:4]. w indices same as Block 2.
    add_strassen(21, [2,3,6,7], [14,15,20,21], [4,5,6,7])
    
    # Block 5: A1 * B31
    # B31: B[0:2, 4:6]. v00=4, v01=5, v10=10, v11=11
    # C31: C[0:2, 4:6]. w00=8, w01=9, w10=10, w11=11
    add_strassen(28, [0,1,4,5], [4,5,10,11], [8,9,10,11])
    
    # Block 6: A2 * B32
    # A2: [2,3,6,7]
    # B32: B[2:4, 4:6]. v00=16, v01=17, v10=22, v11=23
    # C32 adds to C[0:2, 4:6]. w indices same as Block 5.
    add_strassen(35, [2,3,6,7], [16,17,22,23], [8,9,10,11])
    
    return {"U": U, "V": V, "W": W}
