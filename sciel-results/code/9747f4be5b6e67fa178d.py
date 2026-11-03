"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 145  # seconds, safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Use 2x2 block decomposition to reduce rank from 128 to 64
    # Partition A (4x4) into 2x2 blocks, B (4x8) into 2x2 blocks
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # With 2x2 blocks: C_block = [A1 A2] * [B1; B2] where each is 2x2
    # This gives 8 sub-problems of type (2,4,2), each with schoolbook rank 8
    # Total rank = 8 * 8 = 64
    
    rank = 64
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 64
    V = [[0] * rank for _ in range(m * p)]  # 32 x 64
    W = [[0] * rank for _ in range(p * n)]  # 32 x 64
    
    col = 0
    # For each block row i_block (0,1) and block col k_block (0,1,2,3)
    # C_block[i_block, k_block] = A_block[i_block, 0] * B_block[0, k_block] + A_block[i_block, 1] * B_block[1, k_block]
    # A_block[i_block, j_block] is 2x2, B_block[j_block, k_block] is 2x2
    # 
    # For sub-problem (i_block, k_block):
    #   We need to compute C = M1*N1 + M2*N2 where M1,M2,N1,N2 are 2x2
    #   This is equivalent to [M1 M2] (2x4) times [N1; N2] (4x2) = C (2x2)
    #   Schoolbook for (2,4,2): 8 scalar multiplications
    #
    # U indices: (i, j) where i in 0..3, j in 0..3 -> i*m+j = i*4+j
    # V indices: (j, k) where j in 0..3, k in 0..7 -> j*p+k = j*8+k
    # W indices: (k, i) where k in 0..7, i in 0..3 -> k*n+i = k*4+i
    
    for i_block in range(2):  # block row of A and C
        for k_block in range(4):  # block col of B and C
            # For each of the 8 scalar multiplications in this (2,4,2) sub-problem
            for i_local in range(2):  # local row in 2x2 block
                for j_local in range(2):  # local col in 2x2 block (which is the "middle" dim split)
                    # M1 part: A[i_block*2+i_local, j_local] * B[j_local, k_block*2]
                    # M2 part: A[i_block*2+i_local, 2+j_local] * B[2+j_local, k_block*2]
                    # Wait, I need to be more careful.
                    
                    # Let me re-derive. C[i,k] = sum_j A[i,j]*B[j,k]
                    # i = i_block*2 + i_local, k = k_block*2 + k_local
                    # j ranges over 0..3, split into j = j_block*2 + j_local where j_block in {0,1}
                    
                    # For fixed (i_block, k_block, i_local, k_local):
                    # C[i_block*2+i_local, k_block*2+k_local] = 
                    #   sum_{j_block=0}^{1} sum_{j_local=0}^{1} A[i_block*2+i_local, j_block*2+j_local] * B[j_block*2+j_local, k_block*2+k_local]
                    
                    # Each term is one scalar multiplication. There are 2*2 = 4 terms for each (i_local, k_local).
                    # Total for this block: 2 (i_local) * 2 (k_local) * 4 (j terms) = 16? No...
                    
                    # Wait, I'm overcomplicating. Let me just do the schoolbook for each (i,k) pair but group them.
                    pass
    
    # Let me just do the direct schoolbook but with a different indexing to get rank 64
    # Actually, the schoolbook IS rank 128. To get rank 64, I need a genuine algorithmic improvement.
    
    # Let me reconsider. The (2,4,2) sub-problem:
    # C (2x2) = M (2x4) * N (4x2)
    # M = [M1 M2] where M1, M2 are 2x2
    # N = [N1; N2] where N1, N2 are 2x2
    # C = M1*N1 + M2*N2
    #
    # Schoolbook for (2,4,2): for each of 2*2=4 entries of C, sum over 4 middle indices = 16? No.
    # C[i,k] = sum_{j=0}^{3} M[i,j]*N[j,k] for i in {0,1}, k in {0,1}
    # That's 2*2*4 = 16 scalar multiplications for one (2,4,2) sub-problem.
    # 8 sub-problems * 16 = 128. Same as before!
    
    # I made an error. Let me recount.
    # Original: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k] for i in 0..3, k in 0..7
    # Total: 4*8*4 = 128 scalar multiplications.
    
    # With 2x2 blocks:
    # C_block[i_b, k_b] (2x2) = sum_{j_b=0}^{1} A_block[i_b, j_b] * B_block[j_b, k_b]
    # Each A_block is 2x2, B_block is 2x2, product is 2x2.
    # For one block multiplication (2x2)*(2x2): 4*4 = 16 scalar mults? No, 2*2*2 = 8.
    # C_block[i_b,k_b][i_l, k_l] = sum_{j_l=0}^{1} A_block[i_b,j_b][i_l,j_l]*B_block[j_b,k_b][j_l,k_l]
    # For fixed (i_b, k_b, i_l, k_l): sum over j_b (2 values) and j_l (2 values) = 4 terms.
    # Total for one (i_b, k_b) block: 2 (i_l) * 2 (k_l) * 4 (j_b, j_l) = 16 scalar mults.
    # 8 blocks * 16 = 128. Same as schoolbook!
    
    # So the block decomposition doesn't help unless we use a faster algorithm for the sub-problems.
    
    # For the (2,2,2) sub-problem (2x2 * 2x2 = 2x2), Strassen gives rank 7 instead of 8.
    # But our sub-problem is (2,4,2) which is not (2,2,2).
    
    # Let me try a different approach. Use Strassen on the full problem.
    # View 4x4 * 4x8 as 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B11, B12, B13, B14], [B21, B22, B23, B24]] (each 2x2)
    # C = [[C11, C12, C13, C14], [C21, C22, C23, C24]] (each 2x2)
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # etc.
    #
    # Each Cij = Ai1*B1j + Ai2*B2j is a (2,4,2) problem as before.
    
    # Alternative: Can we use a rank-7 algorithm for (2,4,2)?
    # (2,4,2) tensor: C[i,k] = sum_{j=0}^{3} M[i,j]*N[j,k], i in {0,1}, k in {0,1}
    # This is equivalent to multiplying a 2x4 matrix by a 4x2 matrix.
    # We can view this as 4 independent (1,1,1) products summed, but that's rank 4*2*2... no.
    
    # Actually, (2,4,2) has schoolbook rank 2*4*2 = 16.
    # Is there a known lower rank? I'm not sure.
    
    # Let me just implement the schoolbook for now and see if there's a simpler improvement.
    # Actually, let me try rank 64 by using a different decomposition.
    
    # Hmm, let me just return the schoolbook for now since I'm not sure how to beat 128.
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
# EVOLVE-BLOCK-END
