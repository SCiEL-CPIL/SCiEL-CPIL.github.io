import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-28 decomposition for 2x4 * 4x4 matrix multiplication.
    
    This implementation uses a block-Strassen approach.
    The 2x4 * 4x4 multiplication is decomposed into four 2x2 * 2x2 multiplications.
    Each 2x2 * 2x2 multiplication is handled by Strassen's algorithm (rank 7).
    Total rank = 4 * 7 = 28.
    
    The tensor decomposition is constructed by mapping the local Strassen factors
    to the global U, V, W matrices.
    """
    n, m, p = DIMENSIONS # 2, 4, 4
    
    # U shape: n*m = 8
    # V shape: m*p = 16
    # W shape: p*n = 8
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Adds a 7-rank Strassen decomposition for a 2x2 * 2x2 block.
        
        Parameters:
        a_rows: [r0, r1] global row indices in A
        a_cols: [c0, c1] global col indices in A
        b_rows: [r0, r1] global row indices in B
        b_cols: [c0, c1] global col indices in B
        c_rows: [r0, r1] global row indices in C
        c_cols: [c0, c1] global col indices in C
        """
        nonlocal col
        
        # Map local 2x2 indices (00, 01, 10, 11) to global indices
        # U index = i * m + j
        u_00 = a_rows[0] * m + a_cols[0]
        u_01 = a_rows[0] * m + a_cols[1]
        u_10 = a_rows[1] * m + a_cols[0]
        u_11 = a_rows[1] * m + a_cols[1]
        
        # V index = j * p + k
        v_00 = b_rows[0] * p + b_cols[0]
        v_01 = b_rows[0] * p + b_cols[1]
        v_10 = b_rows[1] * p + b_cols[0]
        v_11 = b_rows[1] * p + b_cols[1]
        
        # W index = k * n + i
        w_00 = c_cols[0] * n + c_rows[0]
        w_01 = c_cols[1] * n + c_rows[0]
        w_10 = c_cols[0] * n + c_rows[1]
        w_11 = c_cols[1] * n + c_rows[1]
        
        # Strassen M-factors
        # M1 = (P00 + P11) * (Q00 + Q11)
        U[u_00][col] += 1
        U[u_11][col] += 1
        V[v_00][col] += 1
        V[v_11][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M2 = (P10 + P11) * Q00
        U[u_10][col] += 1
        U[u_11][col] += 1
        V[v_00][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1
        
        # M3 = P00 * (Q01 - Q11)
        U[u_00][col] += 1
        V[v_01][col] += 1
        V[v_11][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M4 = P11 * (Q10 - Q00)
        U[u_11][col] += 1
        V[v_10][col] += 1
        V[v_00][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1
        
        # M5 = (P00 + P01) * Q11
        U[u_00][col] += 1
        U[u_01][col] += 1
        V[v_11][col] += 1
        W[w_01][col] += 1
        W[w_00][col] += -1
        col += 1
        
        # M6 = (P10 - P00) * (Q00 + Q01)
        U[u_10][col] += 1
        U[u_00][col] += -1
        V[v_00][col] += 1
        V[v_01][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M7 = (P01 - P11) * (Q10 + Q11)
        U[u_01][col] += 1
        U[u_11][col] += -1
        V[v_10][col] += 1
        V[v_11][col] += 1
        W[w_00][col] += 1
        col += 1

    # Define the 4 blocks
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    assert col == rank, f"Rank mismatch: expected {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
