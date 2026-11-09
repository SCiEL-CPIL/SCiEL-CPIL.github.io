"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    def _build_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int32)
        V = np.zeros((m * p, rank), dtype=np.int32)
        W = np.zeros((p * n, rank), dtype=np.int32)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that U, V, W form a valid tensor decomposition for matrix multiplication."""
        rank = U.shape[1]
        rng = np.random.RandomState(42)
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            U_col = U[:, r].reshape(n, m)
            V_col = V[:, r].reshape(m, p)
            
            alpha = np.array([np.dot(U_col[i, :], A[i, :]) for i in range(n)])
            beta = np.array([np.dot(V_col[:, k], B[:, k]) for k in range(p)])
            
            W_col = W[:, r].reshape(p, n).T
            
            C_computed += W_col * np.outer(alpha, beta)
        
        return np.allclose(C_computed, C_expected, atol=tol)
    
    def _build_strassen_2x2x2():
        """Build a rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
        rank = 7
        U = np.zeros((4, rank), dtype=np.int32)
        V = np.zeros((4, rank), dtype=np.int32)
        W = np.zeros((4, rank), dtype=np.int32)
        
        # M1 = (a11 + a22)(b11 + b22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (a21 + a22)b11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        W[3, 1] = -1
        
        # M3 = a11(b12 - b22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        W[3, 2] = 1
        
        # M4 = a22(b21 - b11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1
        W[0, 3] = 1
        
        # M5 = (a11 + a12)b22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        W[0, 4] = -1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def _build_2x2x2x7():
        """Build 2x2 @ 2x7 decomposition using 3 Strassen + 1 schoolbook."""
        # 2x2 @ 2x7: A is 2x2, B is 2x7
        # Split B into B1 (2x2), B2 (2x2), B3 (2x2), B4 (2x1)
        decomps = []
        
        # Three Strassen blocks for the 2x2 @ 2x2 parts
        for block_idx in range(3):
            U_s, V_s, W_s = _build_strassen_2x2x2()
            
            U_block = np.zeros((4, 7), dtype=np.int32)
            V_block = np.zeros((14, 7), dtype=np.int32)
            W_block = np.zeros((14, 7), dtype=np.int32)
            
            U_block[:, :] = U_s
            for j in range(2):
                for k in range(2):
                    V_block[j*7 + (2*block_idx + k), :] = V_s[j*2 + k, :]
            for i in range(2):
                for k in range(2):
                    W_block[(2*block_idx + k)*2 + i, :] = W_s[k*2 + i, :]
            
            decomps.append((U_block, V_block, W_block))
        
        # One schoolbook block for 2x2 @ 2x1 (columns 6:7)
        U_sb = np.zeros((4, 4), dtype=np.int32)
        V_sb = np.zeros((14, 4), dtype=np.int32)
        W_sb = np.zeros((14, 4), dtype=np.int32)
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    U_sb[i*2 + j, col] = 1
                    V_sb[j*7 + (6 + k), col] = 1
                    W_sb[(6 + k)*2 + i, col] = 1
                    col += 1
        decomps.append((U_sb, V_sb, W_sb))
        
        # Combine
        total_rank = sum(d[0].shape[1] for d in decomps)
        U = np.zeros((4, total_rank), dtype=np.int32)
        V = np.zeros((14, total_rank), dtype=np.int32)
        W = np.zeros((14, total_rank), dtype=np.int32)
        
        col_offset = 0
        for U_i, V_i, W_i in decomps:
            r_i = U_i.shape[1]
            U[:, col_offset:col_offset+r_i] = U_i
            V[:, col_offset:col_offset+r_i] = V_i
            W[:, col_offset:col_offset+r_i] = W_i
            col_offset += r_i
        
        return U, V, W
    
    def _build_2x4x7_decomposition():
        """Build a decomposition for 2x4 @ 4x7 multiplication using Strassen blocks."""
        # A is 2x4, B is 4x7
        # Split A into A1 (2x2), A2 (2x2)
        # Split B into B1 (2x7), B2 (2x7)
        # C = A1@B1 + A2@B2
        
        decomp1 = _build_2x2x2x7()  # for A1 @ B1
        decomp2 = _build_2x2x2x7()  # for A2 @ B2
        
        r1 = decomp1[0].shape[1]
        r2 = decomp2[0].shape[1]
        total_rank = r1 + r2
        
        U = np.zeros((8, total_rank), dtype=np.int32)
        V = np.zeros((28, total_rank), dtype=np.int32)
        W = np.zeros((14, total_rank), dtype=np.int32)
        
        # Embed decomp1 (A1 @ B1)
        U1, V1, W1 = decomp1
        for i in range(2):
            for j in range(2):
                U[i*4 + j, :r1] = U1[i*2 + j, :]
        for j in range(2):
            for k in range(7):
                V[j*7 + k, :r1] = V1[j*7 + k, :]
        W[:, :r1] = W1
        
        # Embed decomp2 (A2 @ B2)
        U2, V2, W2 = decomp2
        for i in range(2):
            for j in range(2):
                U[i*4 + 2 + j, r1:r1+r2] = U2[i*2 + j, :]
        for j in range(2):
            for k in range(7):
                V[(2+j)*7 + k, r1:r1+r2] = V2[j*7 + k, :]
        W[:, r1:r1+r2] = W2
        
        return U, V, W
    
    def _build_2x4x7_strassen_recursive():
        """
        Alternative approach: use Strassen recursively for better rank.
        
        2x4 @ 4x7:
        We can view this as multiplying a 2x4 matrix by a 4x7 matrix.
        
        Key insight: We can pad to 2x8 @ 8x8 or use other decompositions.
        
        Actually, let's try a different block decomposition:
        Split A (2x4) into two 2x2 blocks: A = [A1 | A2]
        Split B (4x7) into two 2x7 blocks: B = [B1; B2]
        C = A1*B1 + A2*B2
        
        For each 2x2 @ 2x7 product, we use the rank-25 decomposition above.
        Total rank: 25 + 25 = 50.
        
        But can we do better? Let's think about 2x4 @ 4x7 differently.
        
        Alternative: Pad A to 4x4 and B to 4x4, use rank-19 for 4x4x4?
        No, that's not standard.
        
        Let's stick with the rank-50 approach but verify it works.
        """
        return _build_2x4x7_decomposition()
    
    # Try the Strassen-based decomposition first
    U, V, W = _build_2x4x7_strassen_recursive()
    rank = U.shape[1]
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If verification fails, fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
