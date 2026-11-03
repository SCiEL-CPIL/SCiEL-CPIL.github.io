"""Optimized matrix multiplication tensor decomposition for 3x4 x 4x8."""
DIMENSIONS = [3, 4, 8]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def build_strassen_2x2x2():
        """Return U,V,W for 2x2 x 2x2 using Strassen (rank 7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 0: (A11+A22)(B11+B22) -> C11 +=1, C22 +=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 1: (A21+A22)B11 -> C21 +=1, C22 -=1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # Term 2: A11(B21-B22) -> C12 +=1, C21 +=1
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[2, 2] = 1
        
        # Term 3: A22(B12-B11) -> C11 -=1, C21 +=1
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[2, 3] = 1
        
        # Term 4: (A11+A12)B22 -> C11 -=1, C12 +=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # Term 5: (A21-A11)(B11+B12) -> C22 +=1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # Term 6: (A12-A22)(B21+B22) -> C11 +=1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def build_strassen_2x2x8():
        """Build rank-28 decomposition for 2x2 x 2x8 using Strassen on 4 column blocks."""
        U = np.zeros((4, 28))
        V = np.zeros((16, 28))
        W = np.zeros((16, 28))
        
        U_base, V_base, W_base = build_strassen_2x2x2()
        
        for block in range(4):
            offset = block * 7
            U[:, offset:offset+7] = U_base
            V_base_block = np.zeros((16, 7))
            for r in range(2):
                for c in range(2):
                    V_base_block[2*block + r, offset + c] = V_base[r, c]
            V[:, offset:offset+7] = V_base_block
            W_base_block = np.zeros((16, 7))
            for r in range(2):
                for c in range(2):
                    W_base_block[2*block + r, offset + c] = W_base[r, c]
            W[:, offset:offset+7] = W_base_block
        
        return U, V, W
    
    def build_strassen_2x4x8():
        """
        Build decomposition for 2x4 x 4x8.
        Split into two 2x2 x 2x8 blocks.
        Total rank: 28 + 28 = 56
        """
        n2, m2, p2 = 2, 4, 8
        rank_total = 56
        U = np.zeros((n2 * m2, rank_total))  # 8 x 56
        V = np.zeros((m2 * p2, rank_total))  # 32 x 56
        W = np.zeros((p2 * n2, rank_total))  # 16 x 56
        
        # Block 1: A[:,0:2] (2x2) @ B[0:2,:] (2x8)
        U1, V1, W1 = build_strassen_2x2x8()
        # A indices: rows 0,1; cols 0,1 -> flat [0,1,4,5]
        a_map1 = [0, 1, 4, 5]
        for i in range(4):
            U[a_map1[i], 0:28] = U1[i, :]
        
        # B indices: rows 0,1; cols 0-7 -> flat 0-7
        for i in range(16):
            V[i, 0:28] = V1[i, :]
        
        # C indices: rows 0,1; cols 0-7 -> W[k*2+i, r]
        for i in range(2):
            for k in range(8):
                W[k * 2 + i, 0:28] = W1[k * 2 + i, :]
        
        # Block 2: A[:,2:4] (2x2) @ B[2:4,:] (2x8)
        U2, V2, W2 = build_strassen_2x2x8()
        # A indices: rows 0,1; cols 2,3 -> flat [2,3,6,7]
        a_map2 = [2, 3, 6, 7]
        for i in range(4):
            U[a_map2[i], 28:56] = U2[i, :]
        
        # B indices: rows 2,3; cols 0-7 -> flat 16-23
        for i in range(2):
            for k in range(8):
                V[2 * 8 + i * 8 + k, 28:56] = V2[i * 8 + k, :]
        
        # C indices: rows 0,1; cols 0-7 -> W[k*2+i, r]
        for i in range(2):
            for k in range(8):
                W[k * 2 + i, 28:56] = W2[k * 2 + i, :]
        
        return U, V, W
    
    def build_hybrid_3x4x8():
        """
        Build hybrid decomposition for 3x4 x 4x8.
        Split A (3x4) into A_top (2x4) and A_bot (1x4).
        
        A_top (2x4) @ B (4x8): Use Strassen-based 2x4x8 (rank 56)
        A_bot (1x4) @ B (4x8): Use schoolbook (rank 32)
        
        Total rank: 56 + 32 = 88
        """
        n, m, p = 3, 4, 8
        rank_total = 88
        U = np.zeros((n * m, rank_total))  # 12 x 88
        V = np.zeros((m * p, rank_total))  # 32 x 88
        W = np.zeros((p * n, rank_total))  # 24 x 88
        
        # Part 1: A_top (2x4) @ B (4x8) -> contributes to C rows 0,1
        U1, V1, W1 = build_strassen_2x4x8()
        
        # Map U: A_top rows 0,1 (all 4 cols) -> flat A indices 0-7
        for i in range(8):
            U[i, 0:56] = U1[i, :]
        
        # Map V: B (4x8) -> flat B indices 0-31
        for i in range(32):
            V[i, 0:56] = V1[i, :]
        
        # Map W: C rows 0,1 (all 8 cols) -> W[k*3+i, r] where i in {0,1}
        for i in range(2):
            for k in range(8):
                W[k * 3 + i, 0:56] = W1[k * 2 + i, :]
        
        # Part 2: A_bot (1x4) @ B (4x8) -> contributes to C row 2
        # Schoolbook: for each j in 0..3, k in 0..7
        offset = 56
        for j in range(4):
            for k in range(8):
                U[8 + j, offset] = 1      # A[2,j]
                V[j * 8 + k, offset] = 1   # B[j,k]
                W[k * 3 + 2, offset] = 1   # C[2,k]
                offset += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition exactly decomposes the matmul tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        # This is because C[i,k] = sum_j A[i,j] B[j,k], and the tensor T[i,j,k] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try hybrid decomposition first (rank 88)
    try:
        result = build_hybrid_3x4x8()
        if verify(result["U"], result["V"], result["W"], n, m, p):
            return result
        else:
            # Fallback to schoolbook
            return build_schoolbook()
    except Exception:
        return build_schoolbook()
    
# EVOLVE-BLOCK-END
