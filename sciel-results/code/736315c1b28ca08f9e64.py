"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose the 3x5x5 matrix multiplication tensor.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We use a block decomposition:
    - Split j-dimension (5) into [2, 2, 1]  
    - Split k-dimension (5) into [2, 2, 1]
    
    For the 2x2x2 sub-blocks, use Strassen (rank 7 instead of 8).
    For other blocks, use schoolbook.
    
    Block structure (i=3 rows, j-blocks, k-blocks):
    (3,2,2): rank 3*2*2 = 12 (schoolbook, no Strassen benefit for 3x2x2)
    (3,2,2): rank 12
    (3,2,1): rank 6
    (3,2,2): rank 12
    (3,2,2): rank 12
    (3,2,1): rank 6
    (3,1,2): rank 6
    (3,1,2): rank 6
    (3,1,1): rank 3
    Total schoolbook: 12+12+6+12+12+6+6+6+3 = 75
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # We'll build the decomposition term by term
    # Each term is (u_i, v_j, w_k) contributing to T[i,j,k]
    # T[i,j,k] = 1 for all i in [0,3), j in [0,5), k in [0,5)
    
    # Use schoolbook but try to save rank via Strassen on 2x2x2 blocks
    # Split j into [2,2,1], k into [2,2,1]
    
    U_rows = []  # will be list of (row_index, value) per rank column
    V_rows = []
    W_rows = []
    
    rank = 0
    terms = []  # list of (i, j, k, u, v, w)
    
    # For each (j_block, k_block) pair, handle the 3 x j_len x k_len sub-tensor
    j_splits = [2, 2, 1]
    k_splits = [2, 2, 1]
    
    j_offsets = []
    off = 0
    for s in j_splits:
        j_offsets.append(off)
        off += s
    
    k_offsets = []
    off = 0
    for s in k_splits:
        k_offsets.append(off)
        off += s
    
    for jb in range(3):
        for kb in range(3):
            jl = j_splits[jb]
            kl = k_splits[kb]
            jo = j_offsets[jb]
            ko = k_offsets[kb]
            
            if jl == 2 and kl == 2:
                # 3 x 2 x 2 block: use Strassen-style for the 2x2 part
                # C[i, ko+0] = A[i, jo+0]*B[jo+0, ko+0] + A[i, jo+1]*B[jo+1, ko+1]
                # C[i, ko+1] = A[i, jo+0]*B[jo+0, ko+1] + A[i, jo+1]*B[jo+1, ko+1]
                # 
                # Strassen for 2x2: 7 multiplications
                # But here we have 3 rows of A, so it's 3 separate 1x2x2 problems?
                # No - it's 3x2x2, meaning for each of the 3 i-values, we compute
                # a 1x2 times 2x2 = 1x2 result.
                # 
                # Actually, the tensor is T[i, jo+a, ko+b] = 1 for all i,a,b in the block.
                # This is equivalent to 3 copies of the 1x2x2 tensor.
                # Rank of 1x2x2 matmul tensor = 2 (since C[0,0]=A[0,0]B[0,0]+A[0,1]B[1,0], C[0,1]=A[0,0]B[0,1]+A[0,1]B[1,1])
                # Wait, 1x2x2: C[0,k] = sum_j A[0,j] B[j,k]. This has rank 2 (schoolbook).
                # So 3x2x2 has rank 3*2 = 6? No, that's not right either.
                # 
                # The 3x2x2 tensor T[i,j,k]=1: rank = 3*2*2 = 12 (schoolbook).
                # Can Strassen help? Strassen is for 2x2x2 (rank 7 vs 8).
                # For 3x2x2, we can think of it as 3 independent 1x2x2 tensors stacked.
                # Each 1x2x2 has rank 2, so total rank = 3*2 = 6? 
                # No! The rank of a direct sum is the sum of ranks only if they share no structure.
                # Actually T_3x2x2[i,j,k] = 1 = delta structure... 
                # 
                # Let me think more carefully. The 3x2x2 matmul tensor:
                # C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]
                # T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1}, k in {0,1}
                # 
                # Rank lower bound: max(3*2, 2*2, 2*3) = max(6,4,6) = 6
                # Schoolbook: 3*2*2 = 12
                # 
                # Can we achieve rank 6? That would be optimal.
                # Actually I think the rank of 3x2x2 is 12 (no savings known).
                # Let me just use schoolbook for all blocks.
                
                pass  # fall through to schoolbook
            
            # Schoolbook for this block
            for i in range(n):
                for a in range(jl):
                    for b in range(kl):
                        terms.append((i, jo + a, ko + b))
    
    # Now build U, V, W from terms
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (i, j, k) in enumerate(terms):
        U[i * m + j][c] = 1
        V[j * p + k][c] = 1
        W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
