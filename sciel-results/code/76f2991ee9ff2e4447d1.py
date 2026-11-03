import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)  
    # W: (p*n, rank) = (16, rank)
    
    # Strategy: Use Strassen decomposition for 2x2 matrix multiplication
    # applied to blocks, but exploit shared structure to reduce rank.
    # 
    # Key insight: We can use a "tensor product" approach where we
    # decompose the 4x4 middle multiplication using Strassen (rank 14)
    # and combine with the outer 2 and 8 dimensions.
    #
    # Alternative: Use the fact that we have 2 rows in A and 8 columns in B.
    # Split the problem into 4 independent 2x2x2 multiplications (one per
    # pair of B columns), each with Strassen rank 7, giving total rank 28.
    # But wait - that's not right either because the A columns are shared.
    #
    # Let me use a direct approach: The tensor has dimensions 8x32x16.
    # We can find a decomposition using the structure of the problem.
    #
    # Best approach: Use Strassen on 2x2 blocks of the 4x4 middle dimension.
    # Split A (2x4) into two 2x2 blocks: A_left (cols 0-1), A_right (cols 2-3)
    # Split B (4x8) into four 2x2 blocks: B_top_left, B_top_right, B_bot_left, B_bot_right
    # where top/bot refer to rows 0-1 vs 2-3, and left/right refer to cols 0-3 vs 4-7
    #
    # For each of the 4 (A_block, B_block) pairs, we need 2x2 * 2x2 = 2x2 multiplication
    # Using Strassen: 7 multiplications per pair, 4 pairs = 28 total rank.
    #
    # But we can do better: Use a single Strassen decomposition for the entire 4x4
    # middle multiplication. The 4x4 matrix multiplication can be done with rank 14
    # (two independent Strassen 2x2 decompositions).
    #
    # Actually, let me just use the clean 8-Strassen-block approach (rank 56) 
    # but verify it's correct and see if we can optimize.
    
    # Strassen decomposition for 2x2 matrix multiplication
    # Local indices:
    # A: [0,1,2,3] = [A00, A01, A10, A11]
    # B: [0,1,2,3] = [B00, B01, B10, B11]
    # C: [0,1,2,3] = [C00, C01, C10, C11]
    
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1; W_local[3][0] = 1
    
    # M2 = (A10+A11)B00
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1; W_local[3][1] = -1
    
    # M3 = A00(B01-B11)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1; W_local[3][2] = 1
    
    # M4 = A11(B10-B00)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1; W_local[2][3] = 1
    
    # M5 = (A00+A01)B11
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1; W_local[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1
    
    # Now build the full decomposition
    # A is 2x4: A[i,j] with i in {0,1}, j in {0,1,2,3}
    # Flat A index: i*4 + j
    # B is 4x8: B[j,k] with j in {0,1,2,3}, k in {0,...,7}
    # Flat B index: j*8 + k
    # C is 2x8: C[i,k] with i in {0,1}, k in {0,...,7}
    # W convention: W[k*n+i, r] = W[k*2+i, r]
    
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    # 8 blocks: 2 A-blocks (cols 0-1, cols 2-3) x 4 B-blocks (rows 0-1 or 2-3, cols 0-1, 2-3, 4-5, 6-7)
    for a_block in range(2):
        for b_row_block in range(2):
            for b_col_block in range(4):
                block_idx = a_block * 8 + b_row_block * 4 + b_col_block
                base_r = block_idx * 7
                
                # A mapping: 2x2 block from A
                # A_block has cols a_block*2 and a_block*2+1
                a_map = [
                    0 * 4 + a_block * 2,       # A[0, a_block*2]
                    0 * 4 + a_block * 2 + 1,   # A[0, a_block*2+1]
                    1 * 4 + a_block * 2,       # A[1, a_block*2]
                    1 * 4 + a_block * 2 + 1    # A[1, a_block*2+1]
                ]
                
                # B mapping: 2x2 block from B
                # B_block has rows b_row_block*2 and b_row_block*2+1, cols b_col_block*2 and b_col_block*2+1
                b_map = [
                    (b_row_block * 2) * 8 + b_col_block * 2,       # B[b_row_block*2, b_col_block*2]
                    (b_row_block * 2) * 8 + b_col_block * 2 + 1,   # B[b_row_block*2, b_col_block*2+1]
                    (b_row_block * 2 + 1) * 8 + b_col_block * 2,   # B[b_row_block*2+1, b_col_block*2]
                    (b_row_block * 2 + 1) * 8 + b_col_block * 2 + 1  # B[b_row_block*2+1, b_col_block*2+1]
                ]
                
                # C mapping: 2x2 block from C
                # C_block has cols b_col_block*2 and b_col_block*2+1
                # W index: k*2 + i
                c_map = [
                    (b_col_block * 2) * 2 + 0,       # C[0, b_col_block*2]
                    (b_col_block * 2 + 1) * 2 + 0,   # C[0, b_col_block*2+1]
                    (b_col_block * 2) * 2 + 1,       # C[1, b_col_block*2]
                    (b_col_block * 2 + 1) * 2 + 1    # C[1, b_col_block*2+1]
                ]
                
                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        val = U_local[la][l]
                        if val != 0:
                            U[a_map[la]][r] += val
                    for lb in range(4):
                        val = V_local[lb][l]
                        if val != 0:
                            V[b_map[lb]][r] += val
                    for lc in range(4):
                        val = W_local[lc][l]
                        if val != 0:
                            W[c_map[lc]][r] += val
    
    # Verify the decomposition with a random test
    if time.time() - start_time < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for r in range(total_rank):
                        C_computed[i, k] += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
        
        error = np.max(np.abs(C_computed - C_expected))
        if error > 1e-10:
            # Fallback: use schoolbook decomposition (rank 64)
            total_rank = 64
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U[i*m + j][r] = 1
                        V[j*p + k][r] = 1
                        W[k*n + i][r] = 1
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
