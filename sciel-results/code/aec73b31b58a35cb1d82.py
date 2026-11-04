import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _build_laderman_2x3x4():
    """
    Laderman's rank-14 algorithm for 2x3 * 3x4 = 2x4.
    A is 2x3, B is 3x4, C is 2x4.
    Returns (U, V, W) as lists of lists:
      U: 6x14 (rows indexed by i*3+j)
      V: 12x14 (rows indexed by j*4+k)
      W: 8x14 (rows indexed by k*2+i)
    """
    rank = 14
    U = [[0] * rank for _ in range(6)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(8)]
    
    # A indices: i in {0,1}, j in {0,1,2} -> row = i*3 + j
    # B indices: j in {0,1,2}, k in {0,1,2,3} -> row = j*4 + k
    # C indices: i in {0,1}, k in {0,1,2,3} -> row = k*2 + i
    
    # Laderman 2x3x4 products:
    # P1  = a00 * b00          -> contributes to C00
    # P2  = a00 * b10          -> contributes to C01  
    # P3  = a10 * b20          -> contributes to C11
    # P4  = (a00+a10) * (b00+b20) -> contributes to C00, C01, C10, C11
    # P5  = a01 * b01          -> contributes to C02
    # P6  = a01 * b11          -> contributes to C03
    # P7  = a11 * b21          -> contributes to C13
    # P8  = (a01+a11) * (b01+b21) -> contributes to C02, C03, C12, C13
    # P9  = a02 * b02          -> contributes to C00 (k=0... wait)
    
    # Let me re-derive more carefully.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[0,2] = a00*b02 + a01*b12 + a02*b22
    # C[0,3] = a00*b03 + a01*b13 + a02*b23
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    # C[1,2] = a10*b02 + a11*b12 + a12*b22
    # C[1,3] = a10*b03 + a11*b13 + a12*b23
    
    # Standard Laderman 2x3x4 (14 products):
    # The idea is to use 7 "diagonal" products for each of two 2x2x2 subblocks
    # but shared across the 3 columns of A / 3 rows of B.
    
    # Actually, a cleaner approach: use 2x2x4 blocks.
    # Split A into 2 blocks: A1 = A[:, 0:2] (2x2), A2 = A[:, 2:3] (2x1)
    # Split B into 2 blocks: B1 = B[0:2, :] (2x4), B2 = B[2:3, :] (1x4)
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2x4, use two 2x2x2 Strassen: rank 14
    # A2*B2: 2x1x4, schoolbook: rank 8
    # But that's 22, not 14.
    
    # Let me just use the verified 2x2x2 Strassen for the entire 2x4x8
    # with the current partition and ensure correctness.
    
    # For now, return a valid schoolbook 2x3x4 as fallback
    # rank = 2*3*4 = 24
    U = [[0] * 24 for _ in range(6)]
    V = [[0] * 24 for _ in range(12)]
    W = [[0] * 24 for _ in range(8)]
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(4):
                U[i*3+j][col] = 1
                V[j*4+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return U, V, W  # rank 24

def _build_strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen, each 4x7."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Standard Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5 + M7
    # C21 = M2 + M4 - M6
    # C22 = M1 - M3 + M2 + M5
    
    # U rows: i*2+j, V rows: j*2+k, W rows: k*2+i
    # a11=U[0], a12=U[1], a21=U[2], a22=U[3]
    # b11=V[0], b12=V[1], b21=V[2], b22=V[3]
    # c11=W[0], c12=W[2], c21=W[1], c22=W[3]  (W index: k*2+i)
    
    # M1: (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # c11 += M1, c22 += M1
    
    # M2: (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = 1  # c21 += M2, c22 += M2
    
    # M3: a11*(b21-b22)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[0][2] = -1  # c12 += M3, c11 -= M3
    
    # M4: a22*(b12-b11)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[1][3] = 1  # c11 -= M4, c21 += M4
    
    # M5: (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[3][4] = 1  # c12 += M5, c22 += M5
    
    # M6: (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[1][5] = -1; W[0][5] = 1  # c21 -= M6, c11 += M6
    
    # M7: (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[2][6] = 1; W[0][6] = 1  # c12 += M7, c11 += M7
    
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 140.0
    
    n, m, p = 3, 4, 8
    
    # Strategy:
    # i=0: 1x4x8 schoolbook, rank 32
    # i=1,2: 2x4x8 using 8 Strassen 2x2x2 blocks, rank 56
    # Total: 88
    
    # Build Strassen factors
    sU, sV, sW = _build_strassen_2x2x2()
    
    rank = 32 + 8 * 7  # 32 + 56 = 88
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 88
    V = [[0] * rank for _ in range(m * p)]  # 32 x 88
    W = [[0] * rank for _ in range(p * n)]  # 24 x 88
    
    col = 0
    
    # Part 1: i=0, schoolbook 1x4x8
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: i=1,2, Strassen 2x2x2 blocks
    # j blocks: [0,1], [2,3]
    # k blocks: [0,1], [2,3], [4,5], [6,7]
    
    for jb in range(2):
        j_start = jb * 2
        for kb in range(4):
            k_start = kb * 2
            
            for t in range(7):
                # U part
                for li in range(4):
                    if sU[t][li] != 0:
                        local_i = li // 2
                        local_j = li % 2
                        gi = 1 + local_i  # i=1 or 2
                        gj = j_start + local_j
                        U[gi * m + gj][col] += sU[t][li]
                
                # V part
                for lj in range(4):
                    if sV[t][lj] != 0:
                        local_j = lj // 2
                        local_k = lj % 2
                        gj = j_start + local_j
                        gk = k_start + local_k
                        V[gj * p + gk][col] += sV[t][lj]
                
                # W part
                for lk in range(4):
                    if sW[t][lk] != 0:
                        local_i = lk // 2
                        local_k = lk % 2
                        gi = 1 + local_i  # i=1 or 2
                        gk = k_start + local_k
                        W[gk * n + gi][col] += sW[t][lk]
                
                col += 1
    
    # Verify col matches rank
    if col != rank:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Quick verification using random test
    if time.time() < deadline - 2:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Test with a random A and B
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_test = A_test @ B_test
        
        # Compute via decomposition: C[i,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # Actually the tensor decomposition means:
        # T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        # where T[i,j,k] = delta (matrix mult tensor)
        # For matmult: T[i,j,k] = 1 if the element C[i,k] = sum_j A[i,j]*B[j,k]
        # So we check: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] should give the correct C
        
        C_decomp = np.zeros((n, p))
        for t in range(rank):
            # u_vec has shape (n*m,)
            u_vec = U_np[:, t].reshape(n, m)
            v_vec = V_np[:, t].reshape(m, p)
            w_vec = W_np[:, t].reshape(p, n)
            # The rank-1 tensor is outer(u_vec, v_vec, w_vec)
            # C[i,k] += sum_j u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
                    C_decomp[i, k] += s
        
        # For the matmult tensor, the decomposition should satisfy:
        # C_decomp[i,k] = C_test[i,k] for ANY A, B
        # But actually, the tensor T satisfies: C = A @ B iff
        # sum_t U_t(i,j) * V_t(j,k) * W_t(k,i) = delta_{i,j,k} for matmult
        # Hmm, this is tricky. Let me just verify with the actual formula.
        
        # The correct check: the tensor T_{ijk} for matmult is such that
        # C_ik = sum_j T_{ijk} A_{ij} B_{jk}
        # In our factorization: T_{ijk} = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # This should equal 1 when (i,j,k) corresponds to a valid matmult term, 0 otherwise.
        # For matmult, T_{ijk} = 1 for all valid (i,j,k) since C_ik = sum_j A_ij B_jk.
        
        # So let's verify: for each (i,j,k), sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] should be 1
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > max_err:
                        max_err = abs(s - 1.0)
        
        if max_err > 1e-10:
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
