"""Program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # Strategy: Decompose the 2x3 * 3x9 multiplication into 3 independent 2x3 * 3x3 multiplications
    # Each 2x3 * 3x3 multiplication has a known rank-6 decomposition
    # Total rank = 3 * 6 = 18
    
    num_blocks = p // m  # 9 // 3 = 3
    rank_per_block = 6   # rank of 2x3x3 tensor
    total_rank = num_blocks * rank_per_block  # 18
    
    # Build the factor matrices using a systematic approach for the 2x3x3 rank-6 decomposition
    # For 2x3x3, we use the following structure:
    # The tensor T[i,j,k] = delta(i,j,k) for the multiplication (A is 2x3, B is 3x3, C is 2x3)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # T has T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) where i in [0,2), j in [0,3), k in [0,3)
    # This is a 3-way tensor of size 2x3x3 = 18 entries, all 1s
    # We need rank-6 decomposition
    
    # For 2x3x3, one known rank-6 decomposition:
    # U is 6x6 (n*m x rank), V is 9x6 (m*p x rank), W is 6x6 (p*n x rank)
    # where n=2, m=3, p=3
    
    # Let's construct the rank-6 decomposition for a single 2x3x3 block
    # Using the standard decomposition for rectangular matrix multiplication
    
    def build_2x3x3_block():
        """Build rank-6 decomposition for 2x3 * 3x3 -> 2x3"""
        nn, mm, pp = 2, 3, 3
        r = 6
        U_blk = np.zeros((nn * mm, r))  # 6x6
        V_blk = np.zeros((mm * pp, r))  # 9x6
        W_blk = np.zeros((pp * nn, r))  # 6x6
        
        # For 2x3x3, we can use a direct construction
        # The tensor T[i,j,k] = delta_{i,k} * 1 (actually T[i,j,k]=1 for all, but structured)
        # Wait, the multiplication tensor is: T[i,j,k] = 1 if A[i,j]*B[j,k] -> C[i,k]
        # So T[i,j,k] = 1 for all i,j,k in valid ranges (it's the all-ones tensor of shape 2x3x3)
        # No wait, that's not right either. The multiplication tensor has T[i,j,k] = 1 
        # when the (i,j) entry of A times (j,k) entry of B gives (i,k) entry of C.
        # So T is indexed by (i,j) for A, (j,k) for B, (i,k) for C
        # T[(i,j),(j,k),(i,k)] = 1
        # Flattened: U has rows (i*m+j), V has rows (j*p+k), W has rows (k*n+i)
        
        # For 2x3x3: U rows 0..5 (i=0,1; j=0,1,2), V rows 0..8 (j=0,1,2; k=0,1,2), W rows 0..5 (k=0,1,2; i=0,1)
        
        # Rank-6 decomposition for 2x3x3:
        # We can use the following scheme based on Strassen-like ideas or direct enumeration
        # Let me use a known construction:
        
        # Actually, for 2x3x3, the rank is known to be 6 (same as schoolbook since 2*3*3=6)
        # So we can just use the schoolbook decomposition which already gives rank 6
        
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    col = i * mm * pp + j * pp + k  # sequential
                    U_blk[i * mm + j, col] = 1
                    V_blk[j * pp + k, col] = 1
                    W_blk[k * nn + i, col] = 1
        
        return U_blk, V_blk, W_blk
    
    # Build the full 2x3x9 decomposition by combining 3 blocks
    U = np.zeros((n * m, total_rank))  # 6x18
    V = np.zeros((m * p, total_rank))  # 27x18
    W = np.zeros((p * n, total_rank))  # 18x18
    
    for t in range(num_blocks):
        U_blk, V_blk, W_blk = build_2x3x3_block()
        start_col = t * rank_per_block
        end_col = start_col + rank_per_block
        
        # U is the same for each block (same A)
        U[:, start_col:end_col] = U_blk
        
        # V: for block t, the B matrix is B[:, t*m:(t+1)*m]
        # V rows are indexed by j*p + k, where k is the global column index in B
        # For block t, local k' maps to global k = t*m + k'
        for j in range(m):
            for k_local in range(m):
                k_global = t * m + k_local
                v_row = j * p + k_global
                v_blk_row = j * m + k_local  # in the 3x3 block, V row is j*3 + k_local
                V[v_row, start_col:end_col] = V_blk[v_blk_row, :]
        
        # W: for block t, the C matrix columns are t*m to (t+1)*m
        # W rows are indexed by k*n + i, where k is the global row index in C
        # For block t, local k' maps to global k = t*m + k'
        for k_local in range(m):
            for i in range(n):
                k_global = t * m + k_local
                w_row = k_global * n + i
                w_blk_row = k_local * n + i  # in the 3x3 block, W row is k_local*2 + i
                W[w_row, start_col:end_col] = W_blk[w_blk_row, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
