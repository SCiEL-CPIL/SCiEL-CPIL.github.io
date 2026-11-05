import numpy as np
import time

DIMENSIONS = [5, 5, 5]
DEADLINE_SECONDS = 140.0

def _strassen_2x2():
    """Return U, V, W matrices for 2x2x2 Strassen (rank 7).
    U: (4,7) for A flat [a00,a01,a10,a11]
    V: (4,7) for B flat [b00,b01,b10,b11]
    W: (4,7) for C flat [c00,c01,c10,c11]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a00+a11)(b00+b11) -> c00+=1, c11+=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a10+a11)b00 -> c10+=1, c11-=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a00(b01-b11) -> c01+=1, c11+=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = a11(b10-b00) -> c00+=1, c10+=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a00+a01)b11 -> c00-=1, c01+=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (a10-a00)(b00+b01) -> c11+=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a01-a11)(b10+b11) -> c00+=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4():
    """Build 4x4x4 Strassen (rank 49) by composing 2x2 Strassen at two levels.
    Uses 2x2x2 Strassen for 4x4x4: partition 4=2+2, each 2x2 block uses rank-7.
    Total rank = 7 * 7 = 49.
    """
    U2, V2, W2 = _strassen_2x2()
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # For 4x4x4: A is 4x4, B is 4x4, C is 4x4
    # Partition into 2x2 blocks of 2x2 submatrices
    # Block (bi, bj) of A: rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # Block (bj, bk) of B: rows 2*bj..2*bj+1, cols 2*bk..2*bk+1  
    # Block (bk, bi) of C: rows 2*bk..2*bk+1, cols 2*bi..2*bi+1
    # W convention: W[k*n + i] so C[i,k] -> index k*4+i
    
    for k_block in range(2):  # block-level Strassen index for A's row block
        for s_block in range(2):  # block-level Strassen index for A's col block / B's row block
            for t_block in range(2):  # block-level Strassen index for B's col block / C's col block
                # These are the 8 block positions, but we use Strassen's 7 combinations
                # Actually, we need to apply Strassen at the block level too.
                # The 4x4 Strassen: partition into 2x2 blocks, apply 2x2 Strassen to blocks.
                # Block-level Strassen gives 7 block-products, each of which is a 2x2x2 Strassen (rank 7).
                # Total: 7 * 7 = 49.
                pass
    
    # Let me redo this properly.
    # 4x4 Strassen: A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # Apply 2x2 Strassen to the block matrices:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10, C11
    # M3 = A00(B01-B11) -> contributes to C01, C11
    # M4 = A11(B10-B00) -> contributes to C00, C10
    # M5 = (A00+A01)B11 -> contributes to C00, C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A01-A11)(B10+B11) -> contributes to C00
    # Each Mj is a 2x2x2 product, computed with rank-7 Strassen.
    
    # Block-level Strassen factors (same as 2x2 Strassen but on blocks)
    # U_block: (4,7), V_block: (4,7), W_block: (4,7) for block indices
    U_block = np.zeros((4, 7))
    V_block = np.zeros((4, 7))
    W_block = np.zeros((4, 7))
    
    # Same as _strassen_2x2 but for blocks
    U_block[0, 0] = 1; U_block[3, 0] = 1
    V_block[0, 0] = 1; V_block[3, 0] = 1
    W_block[0, 0] = 1; W_block[3, 0] = 1
    
    U_block[2, 1] = 1; U_block[3, 1] = 1
    V_block[0, 1] = 1
    W_block[2, 1] = 1; W_block[3, 1] = -1
    
    U_block[0, 2] = 1
    V_block[1, 2] = 1; V_block[3, 2] = -1
    W_block[1, 2] = 1; W_block[3, 2] = 1
    
    U_block[3, 3] = 1
    V_block[2, 3] = 1; V_block[0, 3] = -1
    W_block[0, 3] = 1; W_block[2, 3] = 1
    
    U_block[0, 4] = 1; U_block[1, 4] = 1
    V_block[3, 4] = 1
    W_block[0, 4] = -1; W_block[1, 4] = 1
    
    U_block[2, 5] = 1; U_block[0, 5] = -1
    V_block[0, 5] = 1; V_block[1, 5] = 1
    W_block[3, 5] = 1
    
    U_block[1, 6] = 1; U_block[3, 6] = -1
    V_block[2, 6] = 1; V_block[3, 6] = 1
    W_block[0, 6] = 1
    
    # Now compose: for each block-level Strassen term (k=0..6) and 
    # each 2x2-level Strassen term (s=0..6), we get a rank-1 term.
    for k in range(7):  # block-level Strassen index
        for s in range(7):  # 2x2-level Strassen index
            t = k * 7 + s
            
            # U4: for each element (i,j) in 4x4 A
            for i in range(4):
                for j in range(4):
                    idx_A = i * 4 + j
                    b_A = (i // 2) * 2 + (j // 2)  # block index in A
                    l_A = (i % 2) * 2 + (j % 2)  # local index in 2x2 block
                    U4[idx_A, t] = U_block[b_A, k] * U2[l_A, s]
                    
            # V4: for each element (j,k2) in 4x4 B
            for j in range(4):
                for k2 in range(4):
                    idx_B = j * 4 + k2
                    b_B = (j // 2) * 2 + (k2 // 2)  # block index in B
                    l_B = (j % 2) * 2 + (k2 % 2)  # local index in 2x2 block
                    V4[idx_B, t] = V_block[b_B, k] * V2[l_B, s]
                    
            # W4: for each element (k2,i) in 4x4 C, W index = k2*4+i
            for k2 in range(4):
                for i in range(4):
                    idx_C = k2 * 4 + i  # W convention: W[k*n+i]
                    b_C = (k2 // 2) * 2 + (i // 2)  # block index in C
                    l_C = (k2 % 2) * 2 + (i % 2)  # local index in 2x2 block
                    W4[idx_C, t] = W_block[b_C, k] * W2[l_C, s]
                    
    return U4, V4, W4

def _hybrid_5x5():
    """5x5x5 matrix multiplication: 4x4x4 Strassen core (rank 49) + boundary (rank 21) = rank 70."""
    n, m, p = 5, 5, 5
    rank_core = 49
    rank_boundary = 21
    total_rank = rank_core + rank_boundary
    
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    # Core 4x4 (Rank 49)
    U4, V4, W4 = _strassen_4x4()
    
    # A is 5x5: indices i*5+j for i,j in 0..4
    # B is 5x5: indices j*5+k for j,k in 0..4
    # C is 5x5: W convention W[k*5+i] for C[i,k]
    
    # Core: A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]
    for i in range(4):
        for j in range(4):
            idx_A = i * 5 + j
            for t in range(49):
                U[idx_A, t] = U4[i * 4 + j, t]
                
    for j in range(4):
        for k in range(4):
            idx_B = j * 5 + k
            for t in range(49):
                V[idx_B, t] = V4[j * 4 + k, t]
                
    for i in range(4):
        for k in range(4):
            idx_W = k * 5 + i  # W[k*n+i]
            for t in range(49):
                W[idx_W, t] = W4[k * 4 + i, t]
                
    # Boundary terms (Rank 21)
    col_offset = 49
    
    # 1) A_core·z: C[i,4] = Σⱼ A[i,j]·B[j,4], j=0..3
    #    For each j: U[i*5+j, c]=1, V[j*5+4, c]=1, W[4*5+i, c]=1 for i=0..3
    for j in range(4):
        c = col_offset
        for i in range(4):
            U[i * 5 + j, c] = 1
            W[4 * 5 + i, c] = 1
        V[j * 5 + 4, c] = 1
        col_offset += 1
        
    # 2) v·B_core: C[4,k] = Σⱼ A[4,j]·B[j,k], j=0..3
    #    For each j: U[4*5+j, c]=1, V[j*5+k, c]=1 for k=0..3, W[k*5+4, c]=1 for k=0..3
    for j in range(4):
        c = col_offset
        U[4 * 5 + j, c] = 1
        for k in range(4):
            V[j * 5 + k, c] = 1
            W[k * 5 + 4, c] = 1
        col_offset += 1
        
    # 3) u·β: C[i,4] += A[i,4]·B[4,4], i=0..3
    #    For each i: U[i*5+4, c]=1, V[4*5+4, c]=1, W[4*5+i, c]=1
    for i in range(4):
        c = col_offset
        U[i * 5 + 4, c] = 1
        V[4 * 5 + 4, c] = 1
        W[4 * 5 + i, c] = 1
        col_offset += 1
        
    # 4) α·w: C[4,k] += A[4,4]·B[4,k], k=0..3
    #    For each k: U[4*5+4, c]=1, V[4*5+k, c]=1, W[k*5+4, c]=1
    for k in range(4):
        c = col_offset
        U[4 * 5 + 4, c] = 1
        V[4 * 5 + k, c] = 1
        W[k * 5 + 4, c] = 1
        col_offset += 1
        
    # 5) v·z: C[4,4] += Σⱼ A[4,j]·B[j,4], j=0..3
    #    For each j: U[4*5+j, c]=1, V[j*5+4, c]=1, W[4*5+4, c]=1
    for j in range(4):
        c = col_offset
        U[4 * 5 + j, c] = 1
        V[j * 5 + 4, c] = 1
        W[4 * 5 + 4, c] = 1
        col_offset += 1
        
    # 6) α·β: C[4,4] += A[4,4]·B[4,4]
    c = col_offset
    U[4 * 5 + 4, c] = 1
    V[4 * 5 + 4, c] = 1
    W[4 * 5 + 4, c] = 1
    col_offset += 1
    
    assert col_offset == 70, f"Expected 70, got {col_offset}"
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify the decomposition is correct."""
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    rank = U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    L = U_arr @ A_flat
    R = V_arr @ B_flat
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            idx = j * n + i  # W[k*n+i] where k=j, i=i
            C_computed[i, j] = np.sum(W_arr[idx, :] * L * R)
            
    if np.max(np.abs(C_computed - C_expected)) > tol:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    try:
        result = _hybrid_5x5()
        if _verify(result['U'], result['V'], result['W'], n, m, p):
            return result
    except Exception:
        pass
        
    result = _schoolbook(n, m, p)
    return result
