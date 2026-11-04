"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    _start_time = time.time()
    _deadline = _start_time + 115.0
    
    def _time_left():
        return _deadline - time.time()

    def _validate(U, V, W):
        """Validate tensor decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
        n, m, p = 4, 4, 9
        rank = len(U[0])
        if len(U) != n * m:
            return False
        if len(V) != m * p:
            return False
        if len(W) != p * n:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def _strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 Strassen.
        U: 4x7, V: 4x7, W: 4x7.
        Indices:
        U: (0,0), (0,1), (1,0), (1,1)
        V: (0,0), (0,1), (1,0), (1,1)
        W: (0,0), (0,1), (1,0), (1,1)
        """
        # M1 = (a+d)(e+h)
        # M2 = d(f-e)
        # M3 = (c+d)e
        # M4 = a(f-h)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        U = [
            [1, 0, 0, 1],  # a
            [0, 0, 0, 1],  # b? No.
            # Let's use the standard representation:
            # U rows: a, b, c, d
            # V rows: e, f, g, h
            # W rows: c00, c01, c10, c11
            
            # M1: a=1, d=1; e=1, h=1; c00=1, c11=1
            [1, 0, 0, 1, 1, -1, 0],
            # M2: d=1; e=-1, f=1; c11=1
            [0, 0, 0, 1, 0, 1, 0],
            # M3: c=1, d=1; e=1; c10=1
            [0, 0, 1, 1, 0, 0, 0],
            # M4: a=1; f=1, h=-1; c01=1
            [1, 0, 0, 0, 0, 0, 0],
            # M5: a=1, b=1; h=1; c01=1
            [1, 1, 0, 0, 0, 0, 0],
            # M6: a=-1, c=1; e=1, f=1; c10=1
            [-1, 0, 1, 0, 0, 0, 0],
            # M7: b=1, d=-1; g=1, h=1; c11=1
            [0, 1, 0, -1, 0, 0, 0]
        ]
        # Wait, the list above is U. Let's define them clearly.
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[3, 1] = 1
        V[0, 1] = -1; V[1, 1] = 1
        W[3, 1] = 1
        
        # M3
        U[2, 2] = 1; U[3, 2] = 1
        V[0, 2] = 1
        W[2, 2] = 1
        
        # M4
        U[0, 3] = 1
        V[1, 3] = 1; V[3, 3] = -1
        W[1, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        
        # M6
        U[0, 5] = -1; U[2, 5] = 1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[3, 6] = 1
        
        return U.tolist(), V.tolist(), W.tolist()

    def _tensor_product_decomp(U1, V1, W1, U2, V2, W2):
        """
        Compute tensor product of two decompositions.
        T1: n1 x m1 x p1, rank R1
        T2: n2 x m2 x p2, rank R2
        Result: (n1*n2) x (m1*m2) x (p1*p2), rank R1*R2
        
        Indexing:
        U1[i1, j1, c1], U2[i2, j2, c2]
        U[i1*n2+i2, j1*m2+j2, c1*R2+c2] = U1[i1, j1, c1] * U2[i2, j2, c2]
        """
        n1, m1 = len(U1), len(U1[0]) // 7 # Hacky, better to pass dims
        # Let's infer dims from the problem context or pass them.
        # For 2x2x2: U is 4x7. n*m=4.
        # We need n, m, p.
        pass

    # Since the dimensions are fixed and small, we can hardcode the combination logic.
    
    def _build_4x4x9():
        n, m, p = 4, 4, 9
        
        # 1. Get 2x2x2 Strassen
        U2, V2, W2 = _strassen_2x2x2()
        
        # 2. Build 4x4x4 Strassen (Tensor Product of 2x2x2 and 2x2x2)
        # Rank 49
        U444 = [[0]*49 for _ in range(16)]
        V444 = [[0]*49 for _ in range(16)]
        W444 = [[0]*49 for _ in range(16)]
        
        for i1 in range(2):
            for j1 in range(2):
                for c1 in range(7):
                    val1 = U2[i1*2+j1][c1]
                    if val1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            for c2 in range(7):
                                val2 = U2[i2*2+j2][c2]
                                if val2 == 0: continue
                                # Map indices
                                i = i1*2 + i2
                                j = j1*2 + j2
                                c = c1*7 + c2
                                U444[i*4+j][c] = val1 * val2
                                
                                # V and W follow similar logic
                                # V1[j1, k1], V2[j2, k2] -> V[j1*2+j2, k1*2+k2]
                                # But we need to iterate over k too.
                                # This nested loop is getting large. 2^6 * 7^2 = 64 * 49 = 3136 ops. Fine.
                                
        # Let's do it more efficiently using numpy broadcasting
        U2_np = np.array(U2)
        V2_np = np.array(V2)
        W2_np = np.array(W2)
        
        # Reshape to 2x2x7
        U2_t = U2_np.reshape(2, 2, 7)
        V2_t = V2_np.reshape(2, 2, 7)
        W2_t = W2_np.reshape(2, 2, 7)
        
        # Tensor product
        # U444[i1, i2, j1, j2, c1, c2] = U2[i1, j1, c1] * U2[i2, j2, c2]
        # We want U444_flat[i1*2+i2, j1*2+j2, c1*7+c2]
        
        U444_np = np.einsum('ijc,klc2->ikjlcc2', U2_t, U2_t) # This is not quite right, indices mixed.
        # Correct einsum:
        # U2[i, j, c]
        # U4[i1, i2, j1, j2, c1, c2] = U2[i1, j1, c1] * U2[i2, j2, c2]
        U444_np = np.einsum('ijc,klc2->ikjlcc2', U2_t, U2_t)
        # Reshape to (4, 4, 49)
        U444_np = U444_np.reshape(4, 4, 49)
        # Flatten first two dims
        U444_np = U444_np.reshape(16, 49)
        
        V444_np = np.einsum('ijc,klc2->ikjlcc2', V2_t, V2_t).reshape(16, 49)
        W444_np = np.einsum('ijc,klc2->ikjlcc2', W2_t, W2_t).reshape(16, 49)
        
        # 3. Build 4x4x1 Schoolbook
        # Rank 16
        U441 = [[0]*16 for _ in range(16)]
        V441 = [[0]*16 for _ in range(4)] # m*p = 4*1 = 4
        W441 = [[0]*16 for _ in range(4)] # p*n = 1*4 = 4
        
        col = 0
        for i in range(4):
            for j in range(4):
                U441[i*4+j][col] = 1
                V441[j*1+0][col] = 1
                W441[0*4+i][col] = 1
                col += 1
                
        # 4. Combine
        # Total rank = 49 + 49 + 16 = 114
        total_rank = 114
        U = [[0]*total_rank for _ in range(16)]
        V = [[0]*total_rank for _ in range(36)] # 4*9
        W = [[0]*total_rank for _ in range(36)] # 9*4
        
        # Block 1: 4x4x4 (cols 0-48)
        # A is 4x4, B1 is 4x4 (cols 0-3), C1 is 4x4 (cols 0-3)
        for i in range(16):
            for c in range(49):
                U[i][c] = U444_np[i][c]
                
        for j in range(4):
            for k in range(4):
                for c in range(49):
                    V[j*9+k][c] = V444_np[j*4+k][c]
                    
        for k in range(4):
            for i in range(4):
                for c in range(49):
                    W[k*4+i][c] = W444_np[k*4+i][c]
                    
        # Block 2: 4x4x4 (cols 49-97)
        # A is 4x4, B2 is 4x4 (cols 4-7), C2 is 4x4 (cols 4-7)
        for i in range(16):
            for c in range(49):
                U[i][49+c] = U444_np[i][c]
                
        for j in range(4):
            for k in range(4):
                for c in range(49):
                    V[j*9+(4+k)][49+c] = V444_np[j*4+k][c]
                    
        for k in range(4):
            for i in range(4):
                for c in range(49):
                    W[(4+k)*4+i][49+c] = W444_np[k*4+i][c]
                    
        # Block 3: 4x4x1 (cols 98-113)
        # A is 4x4, B3 is 4x1 (col 8), C3 is 4x1 (col 8)
        for i in range(16):
            for c in range(16):
                U[i][98+c] = U441[i][c]
                
        for j in range(4):
            for c in range(16):
                V[j*9+8][98+c] = V441[j][c]
                
        for i in range(4):
            for c in range(16):
                W[8*4+i][98+c] = W441[i][c]
                
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook 144
    def _schoolbook():
        rank = 144
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(36)]
        W = [[0]*rank for _ in range(36)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U[i*4+j][col] = 1
                    V[j*9+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = _schoolbook()
    
    if _time_left() > 5:
        candidate = _build_4x4x9()
        if _validate(candidate["U"], candidate["V"], candidate["W"]):
            cand_rank = len(candidate["U"][0])
            if cand_rank < 144:
                best = candidate
    
    return best
# EVOLVE-BLOCK-END
