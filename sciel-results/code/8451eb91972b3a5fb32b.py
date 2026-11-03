import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Strategy:
    # 1. Define the Strassen decomposition for 2x2x2 programmatically.
    # 2. Verify it works for a random 2x2x2 case.
    # 3. Assemble the 2x4x6 decomposition by tiling 6 such 2x2x2 blocks.
    # 4. Verify the full 2x4x6 decomposition.
    # 5. Return the decomposition if valid, else fall back to trivial.
    
    def get_strassen_2x2x2():
        """
        Returns U, V, W for 2x2 * 2x2 = 2x2.
        Shapes: U (4, 7), V (4, 7), W (4, 7).
        Indices:
        A: [[0, 1], [2, 3]]
        B: [[0, 1], [2, 3]]
        C: [[0, 1], [2, 3]]
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C00 += M1, C11 += M1
        
        # M2 = (A10 + A11) * B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1 # C10 += M2, C11 -= M2
        
        # M3 = A00 * (B10 - B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1 # C01 += M3, C11 += M3
        
        # M4 = A11 * (B01 - B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1 # C00 += M4, C10 += M4
        
        # M5 = (A00 + A01) * B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1 # C00 -= M5, C01 += M5
        
        # M6 = (A10 - A00) * (B00 + B10)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1 # C11 += M6
        
        # M7 = (A01 - A11) * (B01 + B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1 # C00 += M7
        
        return U, V, W

    def verify_decomposition(U, V, W, n, m, p, trials=10):
        """
        Verify that U @ V @ W reconstructs the matrix multiplication tensor.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        rank = U.shape[1]
        # Tensor T[i, j, k] = 1 if i,j,k correspond to a valid multiplication term
        # Actually, we check if the reconstructed matrix C = A B matches.
        # The tensor decomposition implies:
        # C[i, k] = sum_j A[i, j] B[j, k]
        # In tensor form: T_{i, j, k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
        # The decomposition gives: T_{i, j, k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
        
        for _ in range(trials):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_expected = A @ B
            
            # Reconstruct C from U, V, W
            # C_recon[i, k] = sum_j sum_r A[i, j] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is computationally expensive if done naively.
            # Better:
            # Let's form the tensor T_recon of shape (n, m, p)
            # T_recon[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            
            # We can compute this efficiently using einsum
            # T_recon = np.einsum('ir,jkr,kir->ijk', U, V, W) 
            # Wait, W index is k*n+i. So W[k*n+i, r].
            # Let's reshape W to (p, n, rank)
            W_reshaped = W.reshape(p, n, rank)
            # U is (n, m, rank)
            U_reshaped = U.reshape(n, m, rank)
            # V is (m, p, rank)
            V_reshaped = V.reshape(m, p, rank)
            
            # T_recon[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
            # np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            
            if time.time() > deadline - 5:
                return False # Timeout in verification, assume fail
            
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            
            # The true tensor T_true[i, j, k] is 1 for all i, j, k?
            # No.
            # C[i, k] = sum_j A[i, j] B[j, k]
            # So the coefficient of A[i, j] B[j, k] in C[i, k] is 1.
            # Thus T_true[i, j, k] = 1 for all i, j, k.
            
            if not np.allclose(T_recon, np.ones((n, m, p)), atol=1e-9):
                return False
                
        return True

    def trivial_decomposition():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def strassen_decomposition():
        """
        Construct rank-42 decomposition using 6 Strassen blocks.
        A (2x4) is split into A1 (2x2) and A2 (2x2).
        B (4x6) is split into 6 2x2 blocks: B1, B2, B3 (top), B4, B5, B6 (bottom).
        C (2x6) is formed by 6 2x2 blocks: C1, C2, C3 (top), C4, C5, C6 (bottom).
        
        Wait, the multiplication is:
        C = A * B
        C[0, 0:2] = A[0, 0:2] * B[0:2, 0:2] + A[0, 2:4] * B[2:4, 0:2]
        This is NOT 6 independent 2x2 multiplications.
        
        Let's re-evaluate the block structure.
        A = [A1 A2] where A1, A2 are 2x2.
        B = [B1 B2 B3] (top 2x6)
            [B4 B5 B6] (bottom 2x6)
        
        C = A1 * [B1 B2 B3] + A2 * [B4 B5 B6]
        C = [A1*B1 + A2*B4, A1*B2 + A2*B5, A1*B3 + A2*B6]
        
        Each term like A1*B1 is a 2x2 * 2x2 multiplication.
        So we have 6 such multiplications.
        However, the results are summed.
        C_block_1 = A1*B1 + A2*B4
        C_block_2 = A1*B2 + A2*B5
        C_block_3 = A1*B3 + A2*B6
        
        We can use Strassen for each of the 6 multiplications.
        Rank = 6 * 7 = 42.
        
        Let's map the indices.
        A1 indices in U (n*m = 8):
        A1 is rows 0-1, cols 0-1 of A.
        A1_00 -> A[0,0] -> U[0]
        A1_01 -> A[0,1] -> U[1]
        A1_10 -> A[1,0] -> U[2]
        A1_11 -> A[1,1] -> U[3]
        
        A2 is rows 0-1, cols 2-3 of A.
        A2_00 -> A[0,2] -> U[4]
        A2_01 -> A[0,3] -> U[5]
        A2_10 -> A[1,2] -> U[6]
        A2_11 -> A[1,3] -> U[7]
        
        B1 is rows 0-1, cols 0-1 of B.
        B1_00 -> B[0,0] -> V[0]
        B1_01 -> B[0,1] -> V[1]
        B1_10 -> B[1,0] -> V[6]
        B1_11 -> B[1,1] -> V[7]
        
        B2 is rows 0-1, cols 2-3 of B.
        B2_00 -> B[0,2] -> V[2]
        B2_01 -> B[0,3] -> V[3]
        B2_10 -> B[1,2] -> V[8]
        B2_11 -> B[1,3] -> V[9]
        
        B3 is rows 0-1, cols 4-5 of B.
        B3_00 -> B[0,4] -> V[4]
        B3_01 -> B[0,5] -> V[5]
        B3_10 -> B[1,4] -> V[10]
        B3_11 -> B[1,5] -> V[11]
        
        B4 is rows 2-3, cols 0-1 of B.
        B4_00 -> B[2,0] -> V[12]
        B4_01 -> B[2,1] -> V[13]
        B4_10 -> B[3,0] -> V[18]
        B4_11 -> B[3,1] -> V[19]
        
        B5 is rows 2-3, cols 2-3 of B.
        B5_00 -> B[2,2] -> V[14]
        B5_01 -> B[2,3] -> V[15]
        B5_10 -> B[3,2] -> V[20]
        B5_11 -> B[3,3] -> V[21]
        
        B6 is rows 2-3, cols 4-5 of B.
        B6_00 -> B[2,4] -> V[16]
        B6_01 -> B[2,5] -> V[17]
        B6_10 -> B[3,4] -> V[22]
        B6_11 -> B[3,5] -> V[23]
        
        C indices in W (p*n = 12):
        C is 2x6.
        C_block_1 is C[0:2, 0:2].
        C1_00 -> C[0,0] -> W[0*2+0] = W[0]
        C1_01 -> C[0,1] -> W[1*2+0] = W[2] ? 
        Wait, W index is k*n + i.
        C[i, k].
        C1_00: i=0, k=0 -> W[0]
        C1_01: i=0, k=1 -> W[1]
        C1_10: i=1, k=0 -> W[2]
        C1_11: i=1, k=1 -> W[3]
        
        C2 is C[0:2, 2:4].
        C2_00: i=0, k=2 -> W[4]
        C2_01: i=0, k=3 -> W[5]
        C2_10: i=1, k=2 -> W[6]
        C2_11: i=1, k=3 -> W[7]
        
        C3 is C[0:2, 4:6].
        C3_00: i=0, k=4 -> W[8]
        C3_01: i=0, k=5 -> W[9]
        C3_10: i=1, k=4 -> W[10]
        C3_11: i=1, k=5 -> W[11]
        
        Now, we add the Strassen contributions.
        
        Block 1: A1 * B1 -> contributes to C1.
        Block 2: A2 * B4 -> contributes to C1.
        Block 3: A1 * B2 -> contributes to C2.
        Block 4: A2 * B5 -> contributes to C2.
        Block 5: A1 * B3 -> contributes to C3.
        Block 6: A2 * B6 -> contributes to C3.
        
        Total Rank = 42.
        """
        rank = 42
        U = np.zeros((8, rank))
        V = np.zeros((24, rank))
        W = np.zeros((12, rank))
        
        U_s, V_s, W_s = get_strassen_2x2x2()
        
        def add_block(a_idx, b_idx, w_idx, offset):
            # a_idx: 4 indices in U for the 2x2 A block
            # b_idx: 4 indices in V for the 2x2 B block
            # w_idx: 4 indices in W for the 2x2 C block
            # offset: rank offset
            for r in range(7):
                # Add U_s[:, r] to U[a_idx, offset+r]
                U[a_idx, offset+r] += U_s[:, r]
                V[b_idx, offset+r] += V_s[:, r]
                W[w_idx, offset+r] += W_s[:, r]
                
        # Block 1: A1 * B1 -> C1
        a1_idx = [0, 1, 2, 3]
        b1_idx = [0, 1, 6, 7]
        c1_idx = [0, 1, 2, 3]
        add_block(a1_idx, b1_idx, c1_idx, 0)
        
        # Block 2: A2 * B4 -> C1
        a2_idx = [4, 5, 6, 7]
        b4_idx = [12, 13, 18, 19]
        # c1_idx is same
        add_block(a2_idx, b4_idx, c1_idx, 7)
        
        # Block 3: A1 * B2 -> C2
        b2_idx = [2, 3, 8, 9]
        c2_idx = [4, 5, 6, 7]
        add_block(a1_idx, b2_idx, c2_idx, 14)
        
        # Block 4: A2 * B5 -> C2
        b5_idx = [14, 15, 20, 21]
        add_block(a2_idx, b5_idx, c2_idx, 21)
        
        # Block 5: A1 * B3 -> C3
        b3_idx = [4, 5, 10, 11]
        c3_idx = [8, 9, 10, 11]
        add_block(a1_idx, b3_idx, c3_idx, 28)
        
        # Block 6: A2 * B6 -> C3
        b6_idx = [16, 17, 22, 23]
        add_block(a2_idx, b6_idx, c3_idx, 35)
        
        return U, V, W

    try:
        U, V, W = strassen_decomposition()
        # Verify
        if verify_decomposition(U, V, W, n, m, p, trials=5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to trivial
            return trivial_decomposition()
    except Exception as e:
        # If anything goes wrong, fallback to trivial
        return trivial_decomposition()

# EVOLVE-BLOCK-END
