"""Search for rank-minimal tensor decomposition of 4x4 x 4x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check if U,V,W with given shapes exactly decompose the matmul tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    rank = U_arr.shape[1]
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    try:
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    except Exception:
        return False
    
    return np.allclose(result, 1.0, atol=1e-10)

def _strassen_2x2():
    """Return U, V, W for 2x2 x 2x2 matrix multiplication, rank 7."""
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    # c11 = M1 - M4 + M5 + M6
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M7
    
    U = np.zeros((4, 7))  # (i*2+j, c)
    V = np.zeros((4, 7))  # (j*2+k, c)
    W = np.zeros((4, 7))  # (k*2+i, c)
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1  # a11 (i=0,j=0 -> 0)
    U[3, 0] = 1  # a22 (i=1,j=1 -> 3)
    V[0, 0] = 1  # b11 (j=0,k=0 -> 0)
    V[3, 0] = 1  # b22 (j=1,k=1 -> 3)
    # c11 += M1: k=0,i=0 -> 0
    W[0, 0] = 1
    # c22 += M1: k=1,i=1 -> 3
    W[3, 0] = 1
    
    # M2 = (a21+a22)b11
    U[2, 1] = 1  # a21 (i=1,j=0 -> 2)
    U[3, 1] = 1  # a22 (i=1,j=1 -> 3)
    V[0, 1] = 1  # b11 (j=0,k=0 -> 0)
    # c21 += M2: k=0,i=1 -> 2
    W[2, 1] = 1
    
    # M3 = a11(b12-b22)
    U[0, 2] = 1  # a11 (i=0,j=0 -> 0)
    V[1, 2] = 1  # b12 (j=0,k=1 -> 1)
    V[3, 2] = -1  # -b22 (j=1,k=1 -> 3)
    # c12 += M3: k=1,i=0 -> 1
    W[1, 2] = 1
    # c22 += M3: k=1,i=1 -> 3
    W[3, 2] = 1
    
    # M4 = a22(b21-b11)
    U[3, 3] = 1  # a22 (i=1,j=1 -> 3)
    V[2, 3] = 1  # b21 (j=1,k=0 -> 2)
    V[0, 3] = -1  # -b11 (j=0,k=0 -> 0)
    # c11 -= M4: k=0,i=0 -> 0
    W[0, 3] = -1
    # c21 += M4: k=0,i=1 -> 2
    W[2, 3] = 1
    
    # M5 = (a11+a12)b22
    U[0, 4] = 1  # a11 (i=0,j=0 -> 0)
    U[1, 4] = 1  # a12 (i=0,j=1 -> 1)
    V[3, 4] = 1  # b22 (j=1,k=1 -> 3)
    # c11 += M5: k=0,i=0 -> 0
    W[0, 4] = 1
    # c12 += M5: k=1,i=0 -> 1
    W[1, 4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1  # a21 (i=1,j=0 -> 2)
    U[0, 5] = -1  # -a11 (i=0,j=0 -> 0)
    V[0, 5] = 1  # b11 (j=0,k=0 -> 0)
    V[1, 5] = 1  # b12 (j=0,k=1 -> 1)
    # c11 += M6: k=0,i=0 -> 0
    W[0, 5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1  # a12 (i=0,j=1 -> 1)
    U[3, 6] = -1  # -a22 (i=1,j=1 -> 3)
    V[2, 6] = 1  # b21 (j=1,k=0 -> 2)
    V[3, 6] = 1  # b22 (j=1,k=1 -> 3)
    # c22 += M7: k=1,i=1 -> 3
    W[3, 6] = 1
    
    return U, V, W

def _kron_extend_4x4(U2, V2, W2):
    """Extend 2x2 Strassen (rank 7) to 4x4 (rank 49) using Kronecker product."""
    n2, m2, p2 = 2, 2, 2
    n4, m4, p4 = 4, 4, 4
    rank2 = U2.shape[1]
    rank4 = rank2 * rank2  # 49
    
    U4 = np.zeros((n4 * m4, rank4))
    V4 = np.zeros((m4 * p4, rank4))
    W4 = np.zeros((p4 * n4, rank4))
    
    for c1 in range(rank2):
        for c2 in range(rank2):
            c4 = c1 * rank2 + c2
            
            # For A (n4 x m4): index (i,j) where i=i1*2+i2, j=j1*2+j2
            # U4[i,j,c4] = U2[i1,j1,c1] * U2[i2,j2,c2]
            for i1 in range(n2):
                for i2 in range(n2):
                    i = i1 * 2 + i2
                    for j1 in range(m2):
                        for j2 in range(m2):
                            j = j1 * 2 + j2
                            U4[i * m4 + j, c4] = U2[i1 * m2 + j1, c1] * U2[i2 * m2 + j2, c2]
            
            # For B (m4 x p4): index (j,k) where j=j1*2+j2, k=k1*2+k2
            # V4[j,k,c4] = V2[j1,k1,c1] * V2[j2,k2,c2]
            for j1 in range(m2):
                for j2 in range(m2):
                    j = j1 * 2 + j2
                    for k1 in range(p2):
                        for k2 in range(p2):
                            k = k1 * 2 + k2
                            V4[j * p4 + k, c4] = V2[j1 * p2 + k1, c1] * V2[j2 * p2 + k2, c2]
            
            # For C (p4 x n4): index (k,i) where k=k1*2+k2, i=i1*2+i2
            # W4[k,i,c4] = W2[k1,i1,c1] * W2[k2,i2,c2]
            for k1 in range(p2):
                for k2 in range(p2):
                    k = k1 * 2 + k2
                    for i1 in range(n2):
                        for i2 in range(n2):
                            i = i1 * 2 + i2
                            W4[k * n4 + i, c4] = W2[k1 * n2 + i1, c1] * W2[k2 * n2 + i2, c2]
    
    return U4, V4, W4

def _replicate_4x8(U4, V4, W4, num_blocks=2):
    """Replicate 4x4 x 4x4 solution to 4x4 x 4x8 by tiling the B and C dimensions."""
    n, m, p4 = 4, 4, 4
    p8 = 8
    rank4 = U4.shape[1]
    rank_total = rank4 * num_blocks
    
    U8 = np.zeros((n * m, rank_total))
    V8 = np.zeros((m * p8, rank_total))
    W8 = np.zeros((p8 * n, rank_total))
    
    for block in range(num_blocks):
        offset = block * rank4
        
        # U is the same for each block (A doesn't change)
        U8[:, offset:offset+rank4] = U4
        
        # V: for B (4 x 8), split into two 4x4 blocks
        # Block 0: B[:, 0:4], Block 1: B[:, 4:8]
        # V4 has shape (m*p4, rank4) = (16, 49)
        # V8 for block: rows j*p8+k where k is in 0..3 for block 0, 4..7 for block 1
        for j in range(m):
            for k4 in range(p4):
                k8 = block * p4 + k4
                V8[j * p8 + k8, offset:offset+rank4] = V4[j * p4 + k4, :]
        
        # W: for C (8 x 4), split into two 4x4 blocks
        # Block 0: C[0:4, :], Block 1: C[4:8, :]
        # W4 has shape (p4*n, rank4) = (16, 49)
        # W8 for block: rows k*n+i where k is in 0..3 for block 0, 4..7 for block 1
        for k4 in range(p4):
            for i in range(n):
                k8 = block * p4 + k4
                W8[k8 * n + i, offset:offset+rank4] = W4[k4 * n + i, :]
    
    return U8, V8, W8

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin (evaluator kills at 150s)
    
    # Get Strassen 2x2 solution
    U2, V2, W2 = _strassen_2x2()
    
    # Extend to 4x4
    U4, V4, W4 = _kron_extend_4x4(U2, V2, W2)
    
    # Verify 4x4 solution
    if not _verify(U4.tolist(), V4.tolist(), W4.tolist(), 4, 4, 4):
        # Fallback: schoolbook for 4x4
        U4 = np.zeros((16, 16))
        V4 = np.zeros((16, 16))
        W4 = np.zeros((16, 16))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U4[i*4+j, col] = 1
                    V4[j*4+k, col] = 1
                    W4[k*4+i, col] = 1
                    col += 1
    
    # Replicate to 4x4 x 4x8
    U8, V8, W8 = _replicate_4x8(U4, V4, W4, num_blocks=2)
    
    # Verify final solution
    if _verify(U8.tolist(), V8.tolist(), V8.tolist(), n, m, p):
        return {"U": U8.tolist(), "V": V8.tolist(), "W": W8.tolist()}
    
    # If verification failed, try schoolbook
    rank_school = n * m * p
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    if _verify(U_sb.tolist(), V_sb.tolist(), W_sb.tolist(), n, m, p):
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Return whatever we have as last resort
    return {"U": U8.tolist(), "V": V8.tolist(), "W": W8.tolist()}

# EVOLVE-BLOCK-END
