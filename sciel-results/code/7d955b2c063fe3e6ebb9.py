"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    
    Approach:
    Decompose the 3x3x5 tensor into a direct sum of five 3x3x1 tensors.
    Each 3x3x1 tensor corresponds to computing the dot products of the rows of the 3x3 matrix A
    with a specific column of the 3x5 matrix B.
    
    The 3x3x1 multiplication tensor has a bilinear rank of 9 (3 dot products of length 3).
    Since the 5 components are independent (disjoint supports in the output index),
    the total rank is 5 * 9 = 45.
    
    We construct the factor matrices U, V, W by iterating over each of the 5 columns k
    and adding the rank-9 decomposition for the 3x3x1 block.
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # The rank of a 3x3x1 tensor is 9 (schoolbook).
    rank_per_block = n * m  # 3 * 3 = 9
    total_rank = p * rank_per_block  # 5 * 9 = 45
    
    U = [[0] * total_rank for _ in range(n * m)]  # 9 x 45
    V = [[0] * total_rank for _ in range(m * p)]  # 15 x 45
    W = [[0] * total_rank for _ in range(p * n)]  # 15 x 45
    
    current_col = 0
    for k in range(p):
        # For each column k, we decompose the 3x3x1 tensor
        # C[i, k] = sum_j A[i, j] * B[j, k]
        # This is 3 dot products.
        # Each dot product (i fixed) is sum_j A[i,j] B[j,k]
        # We assign 3 terms for each i.
        for i in range(n):
            for j in range(m):
                # This term contributes to C[i, k] via A[i, j] and B[j, k]
                # U index: i * m + j
                # V index: j * p + k
                # W index: k * n + i
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][current_col] = 1
                V[v_idx][current_col] = 1
                W[w_idx][current_col] = 1
                
                current_col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
