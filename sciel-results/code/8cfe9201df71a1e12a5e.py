import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 4x5x6 matrix multiplication
    # A is 4x5, B is 5x6, C is 4x6
    # T[i, j, k] = 1 if A[i,j] * B[j,k] = C[i,k]
    
    # We use a block decomposition:
    # Split A into 2x2 blocks of size 2x2 (so n=4 -> 2 blocks of 2)
    # Split B into 2x3 blocks of size 2x3 (so m=5, p=6)
    # Wait, m=5 is odd. Let's split m into 2+3.
    
    # Actually, let's use a more general approach.
    # We'll use Strassen's algorithm for 2x2x2 blocks where possible.
    
    # Strategy:
    # 1. Build a schoolbook decomposition (rank 120)
    # 2. Replace 2x2x2 sub-blocks with Strassen (rank 7 instead of 8)
    
    # For 4x5x6:
    # We can decompose into:
    # - Top-left 2x2x2 block: A[0:2, 0:2] * B[0:2, 0:2] -> Strassen (rank 7)
    # - Other blocks: schoolbook
    
    # Let me think about this more carefully.
    # A is 4x5, B is 5x6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's partition the indices:
    # i: 0,1,2,3
    # j: 0,1,2,3,4
    # k: 0,1,2,3,4,5
    
    # We can use block structure:
    # A = [[A00, A01, A02], [A10, A11, A12]] where each is 2x2 or 2x1
    # Actually, let's partition j into {0,1} and {2,3,4}
    # And k into {0,1,2} and {3,4,5}
    
    # This is getting complicated. Let me use a simpler approach:
    # Just use the full Strassen-based block decomposition.
    
    # For 4x4x4: rank 28 (using Strassen)
    # For 4x4x2: rank 28 (using Strassen on 4x4x4, then take first 2 columns of B)
    # Actually, let me just build it directly.
    
    # Let me use the approach from inspiration 2:
    # Combine Strassen 4x4x4 (rank 28) with schoolbook for the rest.
    
    # 4x5x6 = 4x4x6 + 4x1x6
    # 4x4x6 = 4x4x4 + 4x4x2
    # 4x4x4: Strassen rank 28
    # 4x4x2: Can use Strassen 4x4x4 and keep only first 2 columns? No, that's not right.
    
    # Let me just build a good decomposition.
    
    # Approach: Use Strassen for 2x2x2 blocks.
    # Partition:
    # A: rows {0,1}, {2,3}; cols {0,1}, {2,3,4}
    # B: rows {0,1}, {2,3,4}; cols {0,1,2}, {3,4,5}
    
    # This gives 4 sub-problems for A and B partitions.
    
    # Actually, let me use a cleaner approach:
    # Use the fact that matmul(4,5,6) can be decomposed using
    # matmul(4,4,4) [Strassen, rank 28] + matmul(4,1,6) [schoolbook, rank 24]
    # Total: 28 + 24 = 52? No, that's not how it works.
    
    # Let me think again. The tensor for (4,5,6) has 4*5*6 = 120 terms.
    # We need to find a low-rank decomposition.
    
    # Known results:
    # R(4,4,4) <= 28 (Strassen)
    # R(4,4,2) <= ? 
    # R(4,2,4) <= ?
    # R(2,4,4) <= ?
    
    # For 4x4x2: We can use Strassen on 4x4x4 and discard half the output.
    # Actually, R(4,4,2) is related to R(4,4,4).
    
    # Let me just use a block decomposition with Strassen where applicable.
    
    # Partition j into {0,1,2,3} and {4}
    # Partition k into {0,1,2,3} and {4,5}
    # Partition i into {0,1} and {2,3}
    
    # Block 1: A[0:2, 0:4] * B[0:4, 0:4] -> C[0:2, 0:4] : 2x4x4
    # Block 2: A[0:2, 4:5] * B[4:5, 0:4] -> C[0:2, 0:4] : 2x1x4
    # Block 3: A[2:4, 0:4] * B[0:4, 0:4] -> C[2:4, 0:4] : 2x4x4
    # Block 4: A[2:4, 4:5] * B[4:5, 0:4] -> C[2:4, 0:4] : 2x1x4
    # Block 5: A[0:2, 0:4] * B[0:4, 4:6] -> C[0:2, 4:6] : 2x4x2
    # Block 6: A[0:2, 4:5] * B[4:5, 4:6] -> C[0:2, 4:6] : 2x1x2
    # Block 7: A[2:4, 0:4] * B[0:4, 4:6] -> C[2:4, 4:6] : 2x4x2
    # Block 8: A[2:4, 4:5] * B[4:5, 4:6] -> C[2:4, 4:6] : 2x1x2
    
    # R(2,4,4): Can we do better than schoolbook 32?
    # R(2,1,4): schoolbook 8
    # R(2,4,2): Can we do better than 16?
    # R(2,1,2): schoolbook 4
    
    # For 2x4x4: We can think of it as 2 independent 1x4x4 problems? No.
    # Actually, R(2,4,4) <= R(2,4,4). Let me use schoolbook for simplicity first.
    
    # Total schoolbook: 32+8+32+8+16+4+16+4 = 120 (as expected)
    
    # Let me try to use Strassen for the 2x4x4 blocks.
    # 2x4x4 can be seen as: A is 2x4, B is 4x4.
    # We can split A into two 1x4 rows, and B into 2x2 blocks.
    
    # Actually, let me use a different partition that allows Strassen.
    
    # Better partition:
    # Split i into {0,1,2,3}, j into {0,1,2,3,4}, k into {0,1,2,3,4,5}
    
    # Use Strassen for 2x2x2 subblocks:
    # For each pair (i1,i2) in {(0,1),(2,3)}, (j1,j2) in {(0,1),(2,3)}, (k1,k2) in {(0,1),(2,3),(4,5)}
    # Wait, j has 5 elements, so we can't partition into 2x2x2 cleanly.
    
    # Let me use: j partition {0,1}, {2,3}, {4}
    # k partition {0,1}, {2,3}, {4,5}
    # i partition {0,1}, {2,3}
    
    # 2x2x2 blocks: i in {(0,1),(2,3)}, j in {(0,1),(2,3)}, k in {(0,1),(2,3),(4,5)}
    # That's 2*2*3 = 12 blocks of size 2x2x2, each with Strassen rank 7.
    # Total: 12 * 7 = 84
    
    # Plus the remaining blocks involving j=4:
    # i in {(0,1),(2,3)}, j={4}, k in {(0,1),(2,3),(4,5)}
    # 2*1*3 = 6 blocks of size 2x1x2, each schoolbook rank 4.
    # Total: 6 * 4 = 24
    
    # Grand total: 84 + 24 = 108
    
    # That's better than 120! Let me implement this.
    
    # Actually, let me verify: 12*7 + 6*4 = 84 + 24 = 108. Yes.
    
    # Can we do better?
    # For 2x1x2: rank is 4 (schoolbook). Can we do better? R(2,1,2) = 4 (it's minimal).
    # For 2x2x2: Strassen gives 7. R(2,2,2) = 7 (it's known to be 7).
    
    # So 108 is our target.
    
    # Let me implement this.
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        U: (4, 7) for A[0:2, 0:2]
        V: (4, 7) for B[0:2, 0:2]
        W: (4, 7) for C[0:2, 0:2]
        Convention: U[i*m+j, r] = U[i*2+j, r]
                    V[j*p+k, r] = V[j*2+k, r]
                    W[k*n+i, r] = W[k*2+i, r]
        """
        # Strassen's 7 multiplications:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b01+b11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1: (a00+a11)(b00+b11)
        # u: a00=1, a11=1 -> U[0,0]=1, U[3,0]=1
        # v: b00=1, b11=1 -> V[0,0]=1, V[3,0]=1
        # w: C00=1, C11=1 -> W[0,0]=1, W[3,0]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (a10+a11)b00
        # u: a10=1, a11=1 -> U[2,1]=1, U[3,1]=1
        # v: b00=1 -> V[0,1]=1
        # w: C10=1 -> W[2,1]=1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3: a00(b10-b11)
        # u: a00=1 -> U[0,2]=1
        # v: b10=1, b11=-1 -> V[2,2]=1, V[3,2]=-1
        # w: C01=1 -> W[1,2]=1
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4: a11(b01-b00)
        # u: a11=1 -> U[3,3]=1
        # v: b01=1, b00=-1 -> V[1,3]=1, V[0,3]=-1
        # w: C00=1, C10=1 -> W[0,3]=1, W[2,3]=1
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5: (a00+a01)b11
        # u: a00=1, a01=1 -> U[0,4]=1, U[1,4]=1
        # v: b11=1 -> V[3,4]=1
        # w: C00=-1, C01=1 -> W[0,4]=-1, W[1,4]=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6: (a10-a00)(b00+b10)
        # u: a10=1, a00=-1 -> U[2,5]=1, U[0,5]=-1
        # v: b00=1, b10=1 -> V[0,5]=1, V[2,5]=1
        # w: C11=1 -> W[3,5]=1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1
        
        # M7: (a01-a11)(b01+b11)
        # u: a01=1, a11=-1 -> U[1,6]=1, U[3,6]=-1
        # v: b01=1, b11=1 -> V[1,6]=1, V[3,6]=1
        # w: C00=1 -> W[0,6]=1
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def schoolbook_block(n_b, m_b, p_b):
        """Return U, V, W for schoolbook decomposition of n_b x m_b x p_b.
        U: (n_b*m_b, n_b*m_b*p_b)
        V: (m_b*p_b, n_b*m_b*p_b)
        W: (p_b*n_b, n_b*m_b*p_b)
        """
        rank = n_b * m_b * p_b
        U = np.zeros((n_b * m_b, rank), dtype=float)
        V = np.zeros((m_b * p_b, rank), dtype=float)
        W = np.zeros((p_b * n_b, rank), dtype=float)
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i * m_b + j, col] = 1
                    V[j * p_b + k, col] = 1
                    W[k * n_b + i, col] = 1
                    col += 1
        return U, V, W
    
    # Build the full decomposition
    # i partition: {0,1}, {2,3}
    # j partition: {0,1}, {2,3}, {4}
    # k partition: {0,1}, {2,3}, {4,5}
    
    i_blocks = [(0, 2), (2, 4)]
    j_blocks = [(0, 2), (2, 4), (4, 5)]
    k_blocks = [(0, 2), (2, 4), (4, 6)]
    
    U_list = []
    V_list = []
    W_list = []
    
    for i_start, i_end in i_blocks:
        i_size = i_end - i_start
        for j_start, j_end in j_blocks:
            j_size = j_end - j_start
            for k_start, k_end in k_blocks:
                k_size = k_end - k_start
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Use Strassen
                    U_sub, V_sub, W_sub = strassen_2x2x2()
                else:
                    # Use schoolbook
                    U_sub, V_sub, W_sub = schoolbook_block(i_size, j_size, k_size)
                
                # Embed into full matrices
                rank_sub = U_sub.shape[1]
                U_full = np.zeros((n * m, rank_sub), dtype=float)
                V_full = np.zeros((m * p, rank_sub), dtype=float)
                W_full = np.zeros((p * n, rank_sub), dtype=float)
                
                for r in range(rank_sub):
                    for i_local in range(i_size):
                        i_global = i_start + i_local
                        for j_local in range(j_size):
                            j_global = j_start + j_local
                            U_full[i_global * m + j_global, r] = U_sub[i_local * j_size + j_local, r]
                    
                    for j_local in range(j_size):
                        j_global = j_start + j_local
                        for k_local in range(k_size):
                            k_global = k_start + k_local
                            V_full[j_global * p + k_global, r] = V_sub[j_local * k_size + k_local, r]
                    
                    for k_local in range(k_size):
                        k_global = k_start + k_local
                        for i_local in range(i_size):
                            i_global = i_start + i_local
                            W_full[k_global * n + i_global, r] = W_sub[k_local * i_size + i_local, r]
                
                U_list.append(U_full)
                V_list.append(V_full)
                W_list.append(W_full)
    
    # Concatenate all blocks
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    rank = U.shape[1]
    
    # Verify
    def verify(U, V, W, n, m, p):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r_idx in range(r):
                        actual += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U = np.zeros((n * m, n * m * p), dtype=float)
        V = np.zeros((m * p, n * m * p), dtype=float)
        W = np.zeros((p * n, n * m * p), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
