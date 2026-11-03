import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-28 decomposition for 2x4 * 4x4 matrix multiplication.
    This implementation uses NumPy arrays for the factor matrices and follows
    the block-Strassen strategy: splitting the 4x4 matrix into four 2x2 blocks
    and applying Strassen's algorithm (rank 7) to each resulting 2x2 * 2x2
    multiplication, summing the contributions to the output.
    """
    n, m, p = DIMENSIONS # 2, 4, 4
    
    # Total rank is 4 blocks * 7 (Strassen) = 28
    rank = 28
    
    # Initialize factor matrices as NumPy arrays of zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
        """
        Adds a 7-rank Strassen decomposition for a 2x2 * 2x2 block.
        a_r, a_c: row and col offsets for A (2x4)
        b_r, b_c: row and col offsets for B (4x4)
        c_r, c_c: row and col offsets for C (2x4)
        """
        nonlocal col
        
        # Global indices for A (U)
        # A is 2x4. Index = row * 4 + col
        u_00 = a_r[0] * m + a_c[0]
        u_01 = a_r[0] * m + a_c[1]
        u_10 = a_r[1] * m + a_c[0]
        u_11 = a_r[1] * m + a_c[1]
        
        # Global indices for B (V)
        # B is 4x4. Index = row * 4 + col
        v_00 = b_r[0] * p + b_c[0]
        v_01 = b_r[0] * p + b_c[1]
        v_10 = b_r[1] * p + b_c[0]
        v_11 = b_r[1] * p + b_c[1]
        
        # Global indices for C (W)
        # C is 2x4. W index is k*n + i (col * 2 + row)
        w_00 = c_c[0] * n + c_r[0]
        w_01 = c_c[1] * n + c_r[0]
        w_10 = c_c[0] * n + c_r[1]
        w_11 = c_c[1] * n + c_r[1]
        
        # Strassen M-factors
        # M1 = (P00 + P11) * (Q00 + Q11)
        U[u_00, col] += 1
        U[u_11, col] += 1
        V[v_00, col] += 1
        V[v_11, col] += 1
        W[w_00, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # M2 = (P10 + P11) * Q00
        U[u_10, col] += 1
        U[u_11, col] += 1
        V[v_00, col] += 1
        W[w_10, col] += 1
        W[w_11, col] += -1
        col += 1
        
        # M3 = P00 * (Q01 - Q11)
        U[u_00, col] += 1
        V[v_01, col] += 1
        V[v_11, col] += -1
        W[w_01, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # M4 = P11 * (Q10 - Q00)
        U[u_11, col] += 1
        V[v_10, col] += 1
        V[v_00, col] += -1
        W[w_00, col] += 1
        W[w_10, col] += 1
        col += 1
        
        # M5 = (P00 + P01) * Q11
        U[u_00, col] += 1
        U[u_01, col] += 1
        V[v_11, col] += 1
        W[w_01, col] += 1
        W[w_00, col] += -1
        col += 1
        
        # M6 = (P10 - P00) * (Q00 + Q01)
        U[u_10, col] += 1
        U[u_00, col] += -1
        V[v_00, col] += 1
        V[v_01, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # M7 = (P01 - P11) * (Q10 + Q11)
        U[u_01, col] += 1
        U[u_11, col] += -1
        V[v_10, col] += 1
        V[v_11, col] += 1
        W[w_00, col] += 1
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
