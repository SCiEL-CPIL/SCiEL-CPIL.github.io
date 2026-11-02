"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    start_time = time.time()
    deadline = start_time + 140.0

    def schoolbook(n, m, p):
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
        return U, V, W

    def strassen_2x2x2():
        """Strassen's 2x2x2 rank-7 decomposition."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1
        W[3, 0] = 1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        W[3, 1] = -1
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        W[3, 2] = 1
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1
        W[2, 3] = 1
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1
        W[1, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def verify(U, V, W, n, m, p):
        """Verify the decomposition."""
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomp = np.zeros((n, p))
        for r in range(U.shape[1]):
            M_r = np.zeros(n)
            N_r = np.zeros(p)
            for i in range(n):
                for j in range(m):
                    M_r[i] += U[i*m+j, r] * A[i, j]
            for k in range(p):
                for j in range(m):
                    N_r[k] += V[j*p+k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += W[k*n+i, r] * M_r[i] * N_r[k]
        
        return np.allclose(C_decomp, C_expected, atol=1e-10)

    def compute_2x5x6():
        """Compute 2x5x6 using Strassen for 2x2x2 sub-blocks."""
        n_sub, m_sub, p_sub = 2, 5, 6
        U_parts = []
        V_parts = []
        W_parts = []
        
        # Split m=5 into 4+1, p=6 into 4+2
        # Blocks:
        # 2x4x4 (Rank 28: 4 Strassen 2x2x2)
        # 2x4x2 (Rank 14: 2 Strassen 2x2x2)
        # 2x1x4 (Rank 8: Schoolbook)
        # 2x1x2 (Rank 4: Schoolbook)
        # Total Rank: 28 + 14 + 8 + 4 = 54
        
        # Block 1: 2x4x4 (m=0:4, p=0:4)
        Us, Vs, Ws = strassen_2x2x2()
        rank_s = 7
        # We need 4 such blocks for 2x4x4? No, 2x4x4 is split into 2x2x2 blocks.
        # m=4 -> 2,2. p=4 -> 2,2.
        # 4 blocks of 2x2x2.
        U_block = np.zeros((n_sub * m_sub, rank_s * 4))
        V_block = np.zeros((m_sub * p_sub, rank_s * 4))
        W_block = np.zeros((p_sub * n_sub, rank_s * 4))
        
        col_offset = 0
        for j_block in range(2): # m split
            for k_block in range(2): # p split
                for r in range(rank_s):
                    for i in range(2):
                        for j in range(2):
                            U_block[i * m_sub + (j_block*2 + j), col_offset + r] = Us[i * 2 + j, r]
                    for j in range(2):
                        for k in range(2):
                            V_block[(j_block*2 + j) * p_sub + (k_block*2 + k), col_offset + r] = Vs[j * 2 + k, r]
                    for k in range(2):
                        for i in range(2):
                            W_block[(k_block*2 + k) * n_sub + i, col_offset + r] = Ws[k * 2 + i, r]
                col_offset += rank_s
        
        U_parts.append(U_block)
        V_parts.append(V_block)
        W_parts.append(W_block)
        
        # Block 2: 2x4x2 (m=0:4, p=4:6)
        # Split m=4 -> 2,2. p=2 -> 2.
        # 2 blocks of 2x2x2.
        rank_s = 7
        U_block = np.zeros((n_sub * m_sub, rank_s * 2))
        V_block = np.zeros((m_sub * p_sub, rank_s * 2))
        W_block = np.zeros((p_sub * n_sub, rank_s * 2))
        
        col_offset = 0
        for j_block in range(2): # m split
            for r in range(rank_s):
                for i in range(2):
                    for j in range(2):
                        U_block[i * m_sub + (j_block*2 + j), col_offset + r] = Us[i * 2 + j, r]
                for j in range(2):
                    for k in range(2):
                        V_block[(j_block*2 + j) * p_sub + (4 + k), col_offset + r] = Vs[j * 2 + k, r]
                for k in range(2):
                    for i in range(2):
                        W_block[(4 + k) * n_sub + i, col_offset + r] = Ws[k * 2 + i, r]
            col_offset += rank_s
            
        U_parts.append(U_block)
        V_parts.append(V_block)
        W_parts.append(W_block)
        
        # Block 3: 2x1x4 (m=4:5, p=0:4)
        # Schoolbook: 2*1*4 = 8
        rank_s = 8
        U_block = np.zeros((n_sub * m_sub, rank_s))
        V_block = np.zeros((m_sub * p_sub, rank_s))
        W_block = np.zeros((p_sub * n_sub, rank_s))
        col = 0
        for i in range(2):
            for k in range(4):
                U_block[i * m_sub + 4, col] = 1
                V_block[4 * p_sub + k, col] = 1
                W_block[k * n_sub + i, col] = 1
                col += 1
        U_parts.append(U_block)
        V_parts.append(V_block)
        W_parts.append(W_block)
        
        # Block 4: 2x1x2 (m=4:5, p=4:6)
        # Schoolbook: 2*1*2 = 4
        rank_s = 4
        U_block = np.zeros((n_sub * m_sub, rank_s))
        V_block = np.zeros((m_sub * p_sub, rank_s))
        W_block = np.zeros((p_sub * n_sub, rank_s))
        col = 0
        for i in range(2):
            for k in range(2):
                U_block[i * m_sub + 4, col] = 1
                V_block[4 * p_sub + (4 + k), col] = 1
                W_block[(4 + k) * n_sub + i, col] = 1
                col += 1
        U_parts.append(U_block)
        V_parts.append(V_block)
        W_parts.append(W_block)
        
        U = np.hstack(U_parts)
        V = np.hstack(V_parts)
        W = np.hstack(W_parts)
        
        return U, V, W

    def compute_1x5x6():
        """Compute 1x5x6 schoolbook."""
        return schoolbook(1, 5, 6)

    # Build 3x5x6 from 2x5x6 and 1x5x6
    U1, V1, W1 = compute_2x5x6()
    U2, V2, W2 = compute_1x5x6()
    
    rank_total = U1.shape[1] + U2.shape[1]
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Embed 2x5x6 (rows 0,1 of A)
    for r in range(U1.shape[1]):
        for i in range(2):
            for j in range(5):
                U[i * m + j, r] = U1[i * 5 + j, r]
        for j in range(5):
            for k in range(6):
                V[j * p + k, r] = V1[j * 6 + k, r]
        for k in range(6):
            for i in range(2):
                W[k * n + i, r] = W1[k * 2 + i, r]
    
    # Embed 1x5x6 (row 2 of A)
    offset = U1.shape[1]
    for r in range(U2.shape[1]):
        for j in range(5):
            U[2 * m + j, offset + r] = U2[j, r]
        for j in range(5):
            for k in range(6):
                V[j * p + k, offset + r] = V2[j * 6 + k, r]
        for k in range(6):
            W[k * n + 2, offset + r] = W2[k, r]
    
    # Verify
    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    U, V, W = schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
