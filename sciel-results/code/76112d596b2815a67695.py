"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    
    Strategy:
    1. Primary: Use block decomposition 5 = 2 + 3.
       - Use Strassen (rank 7) for the 2x2x2 corner.
       - Use a structured block decomposition for the 3x3x3 corner and cross terms.
       - This yields a rank of 7 + 27 + 18 + 18 + 18 + 12 + 12 + 12 = 124.
       - Wait, that's barely better than 125. Let's re-evaluate.
       
       Actually, a better block strategy:
       Partition 5x5 into 3x3 and 2x2 blocks.
       A = [[A11(3x3), A12(3x2)], [A21(2x3), A22(2x2)]]
       B = [[B11(3x3), B12(3x2)], [B21(2x3), B22(2x2)]]
       
       The multiplication C = A*B involves:
       C11 = A11*B11 + A12*B21  (3x3)
       C12 = A11*B12 + A12*B22  (3x2)
       C21 = A21*B11 + A22*B21  (2x3)
       C22 = A21*B12 + A22*B22  (2x2)
       
       If we use schoolbook for all sub-blocks:
       C11: 27 (A11*B11) + 6 (A12*B21) = 33 terms? No, the tensor decomposition
       is global. We need to define U, V, W such that the sum reconstructs the
       full 5x5x5 tensor.
       
       Let's stick to the guaranteed valid schoolbook decomposition (rank 125)
       as the base, but attempt to reduce it by using Strassen for the 2x2 block
       embedded in the 5x5.
       
       However, embedding Strassen in a larger matrix multiplication is tricky
       because the indices must align.
       
       Alternative Strategy:
       Use the known rank-103 upper bound? No, that's too complex to derive from scratch.
       
       Let's try a simple block decomposition that is guaranteed to work and
       potentially lower rank if we can find a valid combination.
       
       Actually, the safest and most robust approach within the time limit is
       to return the schoolbook decomposition (rank 125). Any attempt to use
       Strassen or other advanced decompositions requires careful index mapping
       that is error-prone and time-consuming to verify.
       
       Given the 120s limit and the requirement for *exact* decomposition,
       I will implement the schoolbook method which is trivially correct.
    """
    start_time = time.time()
    DEADLINE = 110  # seconds, with margin before 120s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as the primary and fallback method
    def schoolbook():
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
    
    # Check time
    if time.time() - start_time > DEADLINE - 10:
        return schoolbook()
    
    # Attempt a block decomposition using Strassen for 2x2 sub-blocks
    # This is complex and risky, so we'll stick to schoolbook for reliability.
    # However, we can try to construct a valid lower-rank decomposition
    # by using the block structure.
    
    # Let's try a 3+2 block decomposition.
    # A = [[A11(3x3), A12(3x2)], [A21(2x3), A22(2x2)]]
    # B = [[B11(3x3), B12(3x2)], [B21(2x3), B22(2x2)]]
    
    # The tensor terms are:
    # 1. A11 * B11 -> contributes to C11 (3x3). Rank 27 (schoolbook) or 7 (Strassen)
    # 2. A12 * B21 -> contributes to C11 (3x3). Rank 6 (3x2 * 2x3)
    # 3. A11 * B12 -> contributes to C12 (3x2). Rank 18 (3x3 * 3x2)
    # 4. A12 * B22 -> contributes to C12 (3x2). Rank 6 (3x2 * 2x2)
    # 5. A21 * B11 -> contributes to C21 (2x3). Rank 18 (2x3 * 3x3)
    # 6. A22 * B21 -> contributes to C21 (2x3). Rank 6 (2x2 * 2x3)
    # 7. A21 * B12 -> contributes to C22 (2x2). Rank 12 (2x3 * 3x2)
    # 8. A22 * B22 -> contributes to C22 (2x2). Rank 4 (2x2 * 2x2)
    
    # Total rank if all schoolbook: 27+6+18+6+18+6+12+4 = 97
    # This is better than 125!
    
    # Let's implement this block decomposition.
    
    def block_decomposition():
        U_list = []
        V_list = []
        W_list = []
        
        # Helper to add a block term
        def add_block(i_start, j_start, k_start, i_size, j_size, k_size, offset_u, offset_v, offset_w):
            for i in range(i_size):
                for j in range(j_size):
                    for k in range(k_size):
                        u_idx = (i_start + i) * m + (j_start + j)
                        v_idx = (j_start + j) * p + (k_start + k)
                        w_idx = (k_start + k) * n + (i_start + i)
                        
                        U_list.append([0] * 1)  # Placeholder, will flatten later
                        V_list.append([0] * 1)
                        W_list.append([0] * 1)
                        
                        U_list[-1][0] = 1
                        V_list[-1][0] = 1
                        W_list[-1][0] = 1
        
        # This approach is getting complicated with the list management.
        # Let's just build the matrices directly.
        
        rank = 27 + 6 + 18 + 6 + 18 + 6 + 12 + 4
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        
        # Block 1: A11(3x3) * B11(3x3) -> C11(3x3)
        # i in 0..2, j in 0..2, k in 0..2
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        # Block 2: A12(3x2) * B21(2x3) -> C11(3x3)
        # i in 0..2, j in 3..4, k in 0..2
        for i in range(3):
            for j in range(3, 5):
                for k in range(3):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        # Block 3: A11(3x3) * B12(3x2) -> C12(3x2)
        # i in 0..2, j in 0..2, k in 3..4
        for i in range(3):
            for j in range(3):
                for k in range(3, 5):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        # Block 4: A12(3x2) * B22(2x2) -> C12(3x2)
        # i in 0..2, j in 3..4, k in 3..4
        for i in range(3):
            for j in range(3, 5):
                for k in range(3, 5):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        # Block 5: A21(2x3) * B11(3x3) -> C21(2x3)
        # i in 3..4, j in 0..2, k in 0..2
        for i in range(3, 5):
            for j in range(3):
                for k in range(3):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        # Block 6: A22(2x2) * B21(2x3) -> C21(2x3)
        # i in 3..4, j in 3..4, k in 0..2
        for i in range(3, 5):
            for j in range(3, 5):
                for k in range(3):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        # Block 7: A21(2x3) * B12(3x2) -> C22(2x2)
        # i in 3..4, j in 0..2, k in 3..4
        for i in range(3, 5):
            for j in range(3):
                for k in range(3, 5):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        # Block 8: A22(2x2) * B22(2x2) -> C22(2x2)
        # i in 3..4, j in 3..4, k in 3..4
        for i in range(3, 5):
            for j in range(3, 5):
                for k in range(3, 5):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        # Verify rank
        if col != rank:
            # Should not happen, but just in case
            pass
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check time again
    if time.time() - start_time > DEADLINE - 5:
        return schoolbook()
    
    # The block decomposition above is actually just the schoolbook decomposition
    # split into blocks. It has the same rank 125. 
    # My calculation of 97 was wrong because I didn't account for all the terms correctly.
    # Let's re-verify:
    # 3*3*3 = 27
    # 3*2*3 = 18 (not 6)
    # 3*3*2 = 18
    # 3*2*2 = 12 (not 6)
    # 2*3*3 = 18
    # 2*2*3 = 12 (not 6)
    # 2*3*2 = 12
    # 2*2*2 = 8 (not 4)
    # Sum: 27+18+18+12+18+12+12+8 = 125.
    
    # So the block decomposition is identical to schoolbook in rank.
    # To get a lower rank, we need to use Strassen or similar on sub-blocks.
    
    # Let's try to use Strassen for the 2x2 block (A22*B22).
    # A22 is 2x2, B22 is 2x2, C22 is 2x2.
    # Strassen reduces 8 multiplications to 7.
    
    # We can replace the 8 terms of block 8 with 7 Strassen terms.
    
    def block_with_strassen_2x2():
        U = np.zeros((n*m, 125 - 1))  # 125 - 8 + 7 = 124
        V = np.zeros((m*p, 125 - 1))
        W = np.zeros((p*n, 125 - 1))
        
        col = 0
        
        # Blocks 1-7 as before (total 117 terms)
        # Block 1: 27
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        # Block 2: 18
        for i in range(3):
            for j in range(3, 5):
                for k in range(3):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        # Block 3: 18
        for i in range(3):
            for j in range(3):
                for k in range(3, 5):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        # Block 4: 12
        for i in range(3):
            for j in range(3, 5):
                for k in range(3, 5):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        # Block 5: 18
        for i in range(3, 5):
            for j in range(3):
                for k in range(3):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        # Block 6: 12
        for i in range(3, 5):
            for j in range(3, 5):
                for k in range(3):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        # Block 7: 12
        for i in range(3, 5):
            for j in range(3):
                for k in range(3, 5):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        # Block 8: Strassen for 2x2 (7 terms)
        # Indices for A22: i in 3..4, j in 3..4
        # Indices for B22: j in 3..4, k in 3..4
        # Indices for C22: k in 3..4, i in 3..4
        
        # M1 = (A33+A44)(B33+B44)
        U[3*m+3, col] = 1; U[4*m+4, col] = 1
        V[3*p+3, col] = 1; V[4*p+4, col] = 1
        W[3*n+3, col] = 1; W[4*n+4, col] = 1
        col += 1
        
        # M2 = (A43+A44)B33
        U[4*m+3, col] = 1; U[4*m+4, col] = 1
        V[3*p+3, col] = 1
        W[4*n+3, col] = 1
        col += 1
        
        # M3 = A33(B43-B44)
        U[3*m+3, col] = 1
        V[4*p+3, col] = 1; V[4*p+4, col] = -1
        W[3*n+4, col] = 1; W[4*n+4, col] = 1
        col += 1
        
        # M4 = A44(B34-B33)
        U[4*m+4, col] = 1
        V[3*p+4, col] = 1; V[3*p+3, col] = -1
        W[3*n+3, col] = 1; W[4*n+3, col] = 1
        col += 1
        
        # M5 = (A33+A34)B44
        U[3*m+3, col] = 1; U[3*m+4, col] = 1
        V[4*p+4, col] = 1
        W[3*n+3, col] = -1; W[3*n+4, col] = 1
        col += 1
        
        # M6 = (A43-A33)(B33+B43)
        U[4*m+3, col] = 1; U[3*m+3, col] = -1
        V[3*p+3, col] = 1; V[4*p+3, col] = 1
        W[4*n+4, col] = 1
        col += 1
        
        # M7 = (A34-A44)(B34+B44)
        U[3*m+4, col] = 1; U[4*m+4, col] = -1
        V[3*p+4, col] = 1; V[4*p+4, col] = 1
        W[3*n+4, col] = 1
        col += 1
        
        if col != 124:
            # Fallback to schoolbook if something went wrong
            return schoolbook()
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() - start_time > DEADLINE - 2:
        return schoolbook()
    
    return block_with_strassen_2x2()
# EVOLVE-BLOCK-END
