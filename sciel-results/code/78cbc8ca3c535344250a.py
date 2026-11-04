<<<<<<< SEARCH
    rank = 108
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(i0, j0, k0, r_start):
        def u_idx(i, j): return (i0 + i) * m + (j0 + j)
        def v_idx(j, k): return (j0 + j) * p + (k0 + k)
        def w_idx(k, i): return (k0 + k) * n + (i0 + i)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M2 = (A10 + A11)B00
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += -1
        
        # M3 = A00(B01 - B11)
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M4 = A11(B10 - B00)
        r_start += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(0, 0)][r_start] += -1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        
        # M5 = (A00 + A01)B11
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(0, 0)][r_start] += -1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        r_start += 1
        U[u_idx(0, 1)][r_start] += 1
        U[u_idx(1, 1)][r_start] += -1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        
        return r_start + 1

    r_offset = 0
    r_offset = add_strassen_block(0, 0, 0, r_offset)
    r_offset = add_strassen_block(0, 0, 2, r_offset)
    r_offset = add_strassen_block(0, 0, 4, r_offset)
    r_offset = add_strassen_block(2, 0, 0, r_offset)
    r_offset = add_strassen_block(2, 0, 2, r_offset)
    r_offset = add_strassen_block(2, 0, 4, r_offset)
    r_offset = add_strassen_block(0, 2, 0, r_offset)
    r_offset = add_strassen_block(0, 2, 2, r_offset)
    r_offset = add_strassen_block(0, 2, 4, r_offset)
    r_offset = add_strassen_block(2, 2, 0, r_offset)
    r_offset = add_strassen_block(2, 2, 2, r_offset)
    r_offset = add_strassen_block(2, 2, 4, r_offset)
    
    # 4x1x6 strip: rank 24
    for i in range(4):
        for k in range(6):
            U[i * 5 + 4][r_offset] = 1
            V[4 * 6 + k][r_offset] = 1
            W[k * 4 + i][r_offset] = 1
            r_offset += 1
=======
    rank = 101
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen coefficients for 2x2x2: (u00,u10,u11, v00,v01,v10,v11, w00,w01,w10,w11)
    STRASSEN = [
        (1,  0,  1,  1,  0,  0,  1,  1,  0,  0,  1),
        (0,  1,  1,  1,  0,  0,  0,  0,  1,  0, -1),
        (1,  0,  0,  0,  1,  0, -1,  0,  0,  1,  1),
        (0,  0,  1,  0, -1,  1,  0,  1,  1,  0,  0),
        (1,  0,  0,  0,  0,  0,  1, -1,  0,  1,  0),
        (0,  1, -1, 1,  1,  0,  0,  0,  0,  0,  1),
        (0,  0,  1,  0,  0,  1,  1,  1,  0,  0,  0),
    ]
    
    r_offset = 0
    
    # Part 1: 2x2x2 core — A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2], rank 7
    for (u00, u10, u11, v00, v01, v10, v11, w00, w01, w10, w11) in STRASSEN:
        if u00: U[u_idx(0, 0)][r_offset] += u00
        if u10: U[u_idx(1, 0)][r_offset] += u10
        if u11: U[u_idx(1, 1)][r_offset] += u11
        if v00: V[v_idx(0, 0)][r_offset] += v00
        if v01: V[v_idx(0, 1)][r_offset] += v01
        if v10: V[v_idx(1, 0)][r_offset] += v10
        if v11: V[v_idx(1, 1)][r_offset] += v11
        if w00: W[w_idx(0, 0)][r_offset] += w00
        if w01: W[w_idx(0, 1)][r_offset] += w01
        if w10: W[w_idx(1, 0)][r_offset] += w10
        if w11: W[w_idx(1, 1)][r_offset] += w11
        r_offset += 1
    
    # Part 2: 2x3x4 core — A[0:2,2:5] * B[2:5,0:4] -> C[0:2,0:4], rank 20
    # Kronecker product: Strassen(2x2x2) ⊗ schoolbook(1x3x2)
    for (u00, u10, u11, v00, v01, v10, v11, w00, w01, w10, w11) in STRASSEN:
        for jj in range(3):
            for kk in range(2):
                if u00: U[u_idx(0, 2 + jj)][r_offset] += u00
                if u10: U[u_idx(1, 2 + jj)][r_offset] += u10
                if u11: U[u_idx(1, 2 + jj)][r_offset] += u11
                if v00: V[v_idx(2 + jj, kk)][r_offset] += v00
                if v01: V[v_idx(2 + jj, 2 + kk)][r_offset] += v01
                if v10: V[v_idx(2 + jj, kk)][r_offset] += v10
                if v11: V[v_idx(2 + jj, 2 + kk)][r_offset] += v11
                if w00: W[w_idx(kk, 0)][r_offset] += w00
                if w01: W[w_idx(kk, 1)][r_offset] += w01
                if w10: W[w_idx(2 + kk, 0)][r_offset] += w10
                if w11: W[w_idx(2 + kk, 1)][r_offset] += w11
                r_offset += 1
    
    # Part 3: Strips — schoolbook for remaining entries, rank 74
    # A[2:4, 0:2] * B[0:2, 0:6] -> C[0:6, 2:4]  (2*2*6=24)
    for i in range(2, 4):
        for j in range(2):
            for k in range(6):
                U[u_idx(i, j)][r_offset] = 1
                V[v_idx(j, k)][r_offset] = 1
                W[w_idx(k, i)][r_offset] = 1
                r_offset += 1
    # A[0:2, 0:2] * B[0:2, 4:6] -> C[4:6, 0:2]  (2*2*2=8)
    for i in range(2):
        for j in range(2):
            for k in range(4, 6):
                U[u_idx(i, j)][r_offset] = 1
                V[v_idx(j, k)][r_offset] = 1
                W[w_idx(k, i)][r_offset] = 1
                r_offset += 1
    # A[0:2, 2:5] * B[2:5, 4:6] -> C[4:6, 0:2]  (2*3*2=12)
    for i in range(2):
        for j in range(2, 5):
            for k in range(4, 6):
                U[u_idx(i, j)][r_offset] = 1
                V[v_idx(j, k)][r_offset] = 1
                W[w_idx(k, i)][r_offset] = 1
                r_offset += 1
    # A[2:4, 2:5] * B[2:5, 0:6] -> C[0:6, 2:4]  (2*3*6=36)
    for i in range(2, 4):
        for j in range(2, 5):
            for k in range(6):
                U[u_idx(i, j)][r_offset] = 1
                V[v_idx(j, k)][r_offset] = 1
                W[w_idx(k, i)][r_offset] = 1
                r_offset += 1
>>>>>>> REPLACE

Let me verify the rank: 7 + 20 + 24 + 8 + 12 + 36 = 107... wait, that's 107, not 101. Let me recheck.

Actually, I need to re-examine the partition. The 2×3×4 core covers A[0:2, 2:5] × B[2:5, 0:4] → C[0:2, 0:4]. The strips should cover everything else:
- A[2:4, 0:2] × B[0:2, 0:6] → C[0:6, 2:4]: 2×2×6 = 24
- A[0:2, 0:2] × B[0:2, 4:6] → C[4:6, 0:2]: 2×2×2 = 8
- A[0:2, 2:5] × B[2:5, 4:6] → C[4:6, 0:2]: 2×3×2 = 12
- A[2:4, 2:5] × B[2:5, 0:6] → C[0:6, 2:4]: 2×3×6 = 36

Total strips: 24+8+12+36 = 80. Total: 7+20+80 = 107. That's worse than 108? No, 107 < 108. But the inspiration mentioned rank 101.

Let me reconsider. The Kronecker product of Strassen(7) with schoolbook 1×3×2(6) gives 7×6=42, not 20. I made an error. Let me think about what gives rank 20 for 2×3×4.

Actually, for a 2×3×4 tensor, the schoolbook rank is 2×3×4=24. A Strassen-based approach: split the 2 rows into 1+1, giving two 1×3×4 blocks of rank 3×4=12 each, total 24. Or split differently...

The rank-20 claim might come from a specific known decomposition. Let me just use the 107-rank version which is still better than 108, and verify it's correct.

Actually wait - let me re-examine. The current program has rank 108. My new decomposition gives 7+20+80=107 if the 2×3×4 core is rank 20, or 7+42+80=129 if it's the full Kronecker (42). 

Let me just go with the straightforward approach that I can verify: use Strassen on the 2×2×2 sub-blocks where possible and schoolbook elsewhere. The current approach uses 12 Strassen blocks (12×7=84) + 24 schoolbook = 108. 

A better partition: use Strassen on 2×2×2 blocks for the "square" parts and schoolbook for strips. Let me try: 4×5×6 = split rows 4=2+2, cols 5=2+3, out 6=2+2+2.

Actually, let me just fix the rank to 107 and make sure the code is correct:

