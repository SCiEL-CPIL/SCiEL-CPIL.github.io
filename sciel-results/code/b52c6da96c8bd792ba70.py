import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _validate(U, V, W):
    n, m, p = DIMENSIONS
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    if len(U) != n*m: return False
    if len(V) != m*p: return False
    if len(W) != p*n: return False
    r = len(U[0]) if n*m else 0
    for row in U:
        if len(row) != r: return False
    for row in V:
        if len(row) != r: return False
    for row in W:
        if len(row) != r: return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1: return False
    return True

def construct():
    t_start = time.time()
    deadline = t_start + 120.0
    
    n, m, p = DIMENSIONS
    if n != 6 or m != 6 or p != 6:
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def mat_mult(A, B):
        r = len(A); c = len(B[0]); k = len(B)
        res = [[0]*c for _ in range(r)]
        for i in range(r):
            for j in range(c):
                s = 0
                for l in range(k):
                    s += A[i][l]*B[l][j]
                res[i][j] = s
        return res

    def trans(M):
        return [list(row) for row in zip(*M)]

    def block_mult(A, B, nA, nB, nC):
        # A is nA x nA, B is nB x nB, C is nC x nC
        # We treat A, B, C as block matrices and multiply them
        # Result is nA*nB x nB*nC ? No, dimensions must match.
        # Actually, for tensor product:
        # U_total shape (nA*nB, rA*rB)
        # V_total shape (nB*nC, rA*rB)
        # W_total shape (nC*nA, rA*rB)
        #
        # U_total = A \otimes U_A
        # V_total = B \otimes U_B
        # W_total = C \otimes U_C
        #
        # Wait, indices:
        # U_A is (nA^2, rA). U_B is (nB^2, rB).
        # U_total[i*nB + j, a*rB + b] = A[i,j] * U_A[a,b]
        # This is Kronecker product A \otimes U_A.
        #
        # V_total[j*nC + k, a*rB + b] = B[j,k] * V_B[a,b]
        # Kronecker product B \otimes V_B.
        #
        # W_total[k*nA + i, a*rB + b] = C[k,i] * W_C[a,b]
        # Kronecker product C \otimes W_C.
        
        rA = len(A[0]) if nA else 0 # No, rA is rank of decomposition A, not size.
        # Let's pass ranks explicitly to avoid confusion.
        pass # This helper is tricky to generalize without passing ranks. I will write specific combination logic below.

    # Schoolbook 2x2
    U2_school = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    V2_school = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    W2_school = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    
    # Strassen 2x2
    # Rows of U: a, b, c, d
    # Cols: m1..m7
    U2_str = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, -1, 0,  1, 0],
        [1, 1, 0,  1, 0,  0, -1]
    ]
    V2_str = [
        [1, 1, 0, 0, 0, 1, 0], # e
        [0, 0, 1, 0, 0, 1, 0], # f
        [0, 0, 0, 1, 0, 0, 1], # g
        [1, 0, -1, 0, 1, 0, 1]  # h
    ]
    W2_str = [
        [1, 0, 0, 1, -1, 0, 1], # C11
        [0, 0, 1, 0,  1, 0, 0], # C12
        [0, 1, 0, 1,  0, 0, 0], # C21
        [1, -1, 1, 0,  0, 1, 0]  # C22
    ]

    if not _validate(U2_str, V2_str, W2_str):
        print("Strassen 2x2 validation failed, falling back to schoolbook")
        U2 = U2_school; V2 = V2_school; W2 = W2_school
    else:
        U2 = U2_str; V2 = V2_str; W2 = W2_str

    # Schoolbook 3x3
    U3_school = []
    V3_school = []
    W3_school = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                rowU = [0]*27; rowV = [0]*27; rowW = [0]*27
                idx = i*3 + j
                rowU[idx] = 1
                rowV[j*3+k] = 1
                rowW[k*3+i] = 1
                U3_school.append(rowU)
                V3_school.append(rowV)
                W3_school.append(rowW)

    if not _validate(U3_school, V3_school, W3_school):
        print("Schoolbook 3x3 validation failed")
        return {"U": [[0]*216 for _ in range(36)], "V": [[0]*216 for _ in range(36)], "W": [[0]*216 for _ in range(36)]}

    # Combination 1: 6 = 2 x 3
    # Blocks: 2x2 grid. Each block is 3x3.
    # A is 6x6, viewed as 2x2 blocks of 3x3.
    # U_block for A (size 4x27): Identity (since we just pass blocks through)
    U_block_A = [[1 if i==j else 0 for j in range(27)] for i in range(27)] # Wait.
    # U_block_A should be identity of size 27? No.
    # U_total = U_A \otimes U2.
    # U_A is (nA^2, rA). Here nA=3, rA=27.
    # U2 is (4, r2).
    # So U_total is (9*4, 27*r2) = (36, 27*r2).
    # U_total[i*4+j, a*r2+b] = U_A[i,a] * U2[j,b].
    # Here U_A is Identity 27x27.
    # So U_total[i*4+i, a*r2+b] = delta(i,a) * U2[i,b]? No.
    # i is 0..8 (index in U_A rows). j is 0..3 (index in U2 rows).
    # a is 0..26 (col in U_A). b is 0..r2-1 (col in U2).
    # If U_A = I, then U_A[i,a] = 1 if i==a else 0.
    # So U_total[i*4+j, i*r2+b] = U2[j,b].
    
    U_A_id = [[1 if i==j else 0 for j in range(27)] for i in range(27)]
    V_B_id = U_A_id
    W_C_id = U_A_id
    
    # Helper to combine
    def combine(U1, V1, W1, U2, V2, W2):
        r1 = len(U1[0])
        r2 = len(U2[0])
        n1 = int(len(U1)**0.5)
        n2 = int(len(U2)**0.5)
        
        U_res = [[0]*(r1*r2) for _ in range(n1*n1*n2*n2)]
        V_res = [[0]*(r1*r2) for _ in range(n1*n1*n2*n2)]
        W_res = [[0]*(r1*r2) for _ in range(n1*n1*n2*n2)]
        
        # U_total[i*n2 + j, a*r2 + b] = U1[i,a] * U2[j,b]
        for i in range(n1*n1):
            for a in range(r1):
                val1 = U1[i][a]
                if val1 == 0: continue
                for j in range(n2*n2):
                    for b in range(r2):
                        val2 = U2[j][b]
                        if val2 == 0: continue
                        U_res[i*n2 + j][a*r2 + b] += val1 * val2
                        
        # V_total[i*n2 + j, a*r2 + b] = V1[i,a] * V2[j,b]
        for i in range(n1*n1):
            for a in range(r1):
                val1 = V1[i][a]
                if val1 == 0: continue
                for j in range(n2*n2):
                    for b in range(r2):
                        val2 = V2[j][b]
                        if val2 == 0: continue
                        V_res[i*n2 + j][a*r2 + b] += val1 * val2
                        
        # W_total[i*n2 + j, a*r2 + b] = W1[i,a] * W2[j,b]
        for i in range(n1*n1):
            for a in range(r1):
                val1 = W1[i][a]
                if val1 == 0: continue
                for j in range(n2*n2):
                    for b in range(r2):
                        val2 = W2[j][b]
                        if val2 == 0: continue
                        W_res[i*n2 + j][a*r2 + b] += val1 * val2
                        
        return U_res, V_res, W_res

    candidates = []
    
    # Strategy 1: 6 = 2 x 3. Use Strassen(2x2) x Schoolbook(3x3).
    # U1 = U3 (27x27), V1 = V3 (27x27), W1 = W3 (27x27)
    # U2 = U2_str (4x7), V2 = V2_str (4x7), W2 = W2_str (4x7)
    # Result Rank = 27 * 7 = 189.
    U_c1, V_c1, W_c1 = combine(U3_school, V3_school, W3_school, U2_str, V2_str, W2_str)
    if _validate(U_c1, V_c1, W_c1):
        candidates.append((189, U_c1, V_c1, W_c1))
    else:
        print("Candidate 1 (189) invalid")

    # Strategy 2: 6 = 3 x 2. Use Schoolbook(3x3) x Strassen(2x2).
    # This is mathematically equivalent in rank but different structure?
    # U_total = U2 \otimes U3 ?
    # If we swap the block structure:
    # 6 = 3 (blocks) x 2 (inside).
    # U1 = U2_str (4x7), V1 = V2_str (4x7), W1 = W2_str (4x7)
    # U2 = U3_school (27x27), V2 = V3_school (27x27), W2 = W3_school (27x27)
    # Result Rank = 7 * 27 = 189.
    U_c2, V_c2, W_c2 = combine(U2_str, V2_str, W2_str, U3_school, V3_school, W3_school)
    if _validate(U_c2, V_c2, W_c2):
        candidates.append((189, U_c2, V_c2, W_c2))
    else:
        print("Candidate 2 (189) invalid")

    # Strategy 3: Use Strassen for all? 6 is not a power of 2.
    # 6 = 2 x 3. We need a 3x3 decomposition. We only have schoolbook.
    # Is there a better 3x3? Winograd is rank 23.
    # If we had Winograd 3x3, rank would be 7 * 23 = 161.
    # I don't have Winograd factors memorized with confidence.
    # Stick to 189.
    
    # Strategy 4: Check if we can just use schoolbook 6x6? Rank 216.
    # 189 < 216, so 189 is better.
    
    if not candidates:
        # Fallback to Schoolbook 6x6
        U = [[0]*216 for _ in range(36)]
        V = [[0]*216 for _ in range(36)]
        W = [[0]*216 for _ in range(36)]
        col = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U[i*6+j][col] = 1
                    V[j*6+k][col] = 1
                    W[k*6+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Pick the best candidate (lowest rank, then first valid)
    candidates.sort(key=lambda x: x[0])
    best_rank, best_U, best_V, best_W = candidates[0]
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
