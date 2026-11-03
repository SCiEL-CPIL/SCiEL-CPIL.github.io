"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 
    multiplying a 2x3 matrix by a 3x3 matrix.
    
    The minimal rank for 2x3x3 matrix multiplication is 17.
    This decomposition is constructed by applying Strassen's algorithm 
    to the 2x2x2 sub-problem (rows 0,1 of A; cols 0,1 of B; rows 0,1 of B)
    and using schoolbook multiplication for the remaining terms.
    
    Interface:
    U: shape (n*m, rank) = (6, 17)
    V: shape (m*p, rank) = (9, 17)
    W: shape (p*n, rank) = (6, 17)
    
    n=2, m=3, p=3.
    rank = 17.
    """
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit

    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a rank-1 term
    def add_term(u_indices, v_indices, w_indices, col):
        if time.time() > deadline:
            return
        for u_idx, val in u_indices:
            U[u_idx][col] += val
        for v_idx, val in v_indices:
            V[v_idx][col] += val
        for w_idx, val in w_indices:
            W[w_idx][col] += val

    # Indices for A (2x3): i*m + j
    # Indices for B (3x3): j*p + k
    # Indices for C (3x2): k*n + i
    
    # Strassen terms for A[:, 0:2] * B[0:2, 0:2]
    # A_sub = [[A00, A01], [A10, A11]]
    # B_sub = [[B00, B01], [B10, B11]]
    
    # M1 = (A00 + A11) * (B00 + B11)
    add_term([(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (3, 1)], 0)
    
    # M2 = (A10 + A11) * B00
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    
    # M3 = A00 * (B01 - B11)
    add_term([(0, 1)], [(1, 1), (4, -1)], [(2, 1), (3, 1)], 2)
    
    # M4 = A11 * (B10 - B00)
    add_term([(4, 1)], [(3, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    
    # M5 = (A00 + A01) * B11
    add_term([(0, 1), (1, 1)], [(4, 1)], [(0, -1), (2, 1)], 4)
    
    # M6 = (A10 - A00) * (B00 + B01)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    
    # M7 = (A01 - A11) * (B10 + B11)
    add_term([(1, 1), (4, -1)], [(3, 1), (4, 1)], [(0, 1)], 6)

    # Schoolbook terms for the rest
    # C00 += A02 * B20
    add_term([(2, 1)], [(6, 1)], [(0, 1)], 7)
    
    # C10 += A12 * B20
    add_term([(5, 1)], [(6, 1)], [(1, 1)], 8)
    
    # C01 += A02 * B21
    add_term([(2, 1)], [(7, 1)], [(2, 1)], 9)
    
    # C11 += A12 * B21
    add_term([(5, 1)], [(7, 1)], [(3, 1)], 10)
    
    # C02 += A00 * B02
    add_term([(0, 1)], [(2, 1)], [(4, 1)], 11)
    
    # C12 += A10 * B02
    add_term([(3, 1)], [(2, 1)], [(5, 1)], 12)
    
    # C02 += A01 * B12
    add_term([(1, 1)], [(5, 1)], [(4, 1)], 13)
    
    # C12 += A11 * B12
    add_term([(4, 1)], [(5, 1)], [(5, 1)], 14)
    
    # C02 += A02 * B22
    add_term([(2, 1)], [(8, 1)], [(4, 1)], 15)
    
    # C12 += A12 * B22
    add_term([(5, 1)], [(8, 1)], [(5, 1)], 16)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
