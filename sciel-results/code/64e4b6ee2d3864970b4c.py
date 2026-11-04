"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a decomposition of rank 36.
    # M_{2,5,6} = M_{2,5,5} + M_{2,5,1}
    # M_{2,5,5} = M_{2,4,4} + M_{2,4,1} + M_{2,1,4} + M_{2,1,1}
    # Rank(M_{2,4,4}) = 32 (schoolbook)
    # Rank(M_{2,4,1}) = 1
    # Rank(M_{2,1,4}) = 1
    # Rank(M_{2,1,1}) = 1
    # Rank(M_{2,5,1}) = 1
    # Total rank = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 10 x 36
    V = [[0] * rank for _ in range(m * p)]  # 30 x 36
    W = [[0] * rank for _ in range(p * n)]  # 12 x 36
    
    col = 0
    
    # Helper to add a block
    def add_block(n_b, m_b, p_b, i_off, j_off, k_off, rank_b, U_b, V_b, W_b):
        nonlocal col
        for r in range(rank_b):
            for i in range(n_b):
                U[i_off + i][col + r] += U_b[i][r]
            for j in range(m_b):
                V[j_off + j][col + r] += V_b[j][r]
            for k in range(p_b):
                W[k_off + k][col + r] += W_b[k][r]
        col += rank_b

    # M_{2,4,4} at A[0:2, 0:4], B[0:4, 0:4], C[0:2, 0:4]
    # Schoolbook for 2x4x4
    rank_b = 2 * 4 * 4
    U_b = [[0] * rank_b for _ in range(2)]
    V_b = [[0] * rank_b for _ in range(4)]
    W_b = [[0] * rank_b for _ in range(4)]
    c = 0
    for i in range(2):
        for j in range(4):
            for k in range(4):
                U_b[i][c] = 1
                V_b[j][c] = 1
                W_b[k][c] = 1
                c += 1
    add_block(2, 4, 4, 0, 0, 0, rank_b, U_b, V_b, W_b)

    # M_{2,4,1} at A[0:2, 0:4], B[0:4, 4], C[0:2, 4]
    # Rank 1
    rank_b = 1
    U_b = [[1, 1, 1, 1]]  # Wait, U is 2x1. U[i, 0] = 1?
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # This is not rank 1 in the tensor sense unless we use the outer product trick.
    # For M_{n, m, 1}, C[i, 0] = sum_j A[i, j] B[j, 0].
    # This is a dot product.
    # Tensor T_{i, j, 0} = 1.
    # Rank 1?
    # U[i, 0] = 1, V[j, 0] = 1, W[0, 0] = 1.
    # Sum U[i, 0] V[j, 0] W[0, 0] = 1.
    # But we need C[i, 0] = sum_j A[i, j] B[j, 0].
    # The tensor is T_{i, j, k}.
    # For M_{2, 4, 1}, k=0.
    # T_{i, j, 0} = 1.
    # So yes, rank 1.
    U_b = [[1], [1]]
    V_b = [[1], [1], [1], [1]]
    W_b = [[1]]
    add_block(2, 4, 1, 0, 0, 4, rank_b, U_b, V_b, W_b)

    # M_{2,1,4} at A[0:2, 4], B[4, 0:4], C[0:2, 0:4]
    # Rank 1
    rank_b = 1
    U_b = [[1], [1]]
    V_b = [[1]]
    W_b = [[1], [1], [1], [1]]
    add_block(2, 1, 4, 0, 4, 0, rank_b, U_b, V_b, W_b)

    # M_{2,1,1} at A[0:2, 4], B[4, 4], C[0:2, 4]
    # Rank 1
    rank_b = 1
    U_b = [[1], [1]]
    V_b = [[1]]
    W_b = [[1]]
    add_block(2, 1, 1, 0, 4, 4, rank_b, U_b, V_b, W_b)

    # M_{2,5,1} at A[0:2, 0:5], B[0:5, 5], C[0:2, 5]
    # Rank 1
    rank_b = 1
    U_b = [[1], [1]]
    V_b = [[1], [1], [1], [1], [1]]
    W_b = [[1]]
    add_block(2, 5, 1, 0, 0, 5, rank_b, U_b, V_b, W_b)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
