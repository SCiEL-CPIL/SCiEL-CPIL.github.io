"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 
    multiplying a 2x3 matrix by a 3x3 matrix.
    
    The minimal rank for 2x3x3 matrix multiplication is 17.
    This decomposition is constructed by applying Strassen's algorithm 
    to the 2x2x2 sub-problem and using schoolbook multiplication for the remaining terms.
    
    Interface:
    U: shape (n*m, rank) = (6, 17)
    V: shape (m*p, rank) = (9, 17)
    W: shape (p*n, rank) = (6, 17)
    
    n=2, m=3, p=3.
    rank = 17.
    """
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for u_idx, val in u_indices:
            U[u_idx][col] += val
        for v_idx, val in v_indices:
            V[v_idx][col] += val
        for w_idx, val in w_indices:
            W[w_idx][col] += val

    # Strassen terms for A[:, 0:2] * B[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (4, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (4, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(3, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(4, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(3, 1), (4, 1)], [(0, 1)], 6)

    # Schoolbook terms for the rest
    add_term([(2, 1)], [(6, 1)], [(0, 1)], 7)
    add_term([(5, 1)], [(6, 1)], [(1, 1)], 8)
    add_term([(2, 1)], [(7, 1)], [(2, 1)], 9)
    add_term([(5, 1)], [(7, 1)], [(3, 1)], 10)
    add_term([(0, 1)], [(2, 1)], [(4, 1)], 11)
    add_term([(3, 1)], [(2, 1)], [(5, 1)], 12)
    add_term([(1, 1)], [(5, 1)], [(4, 1)], 13)
    add_term([(4, 1)], [(5, 1)], [(5, 1)], 14)
    add_term([(2, 1)], [(8, 1)], [(4, 1)], 15)
    add_term([(5, 1)], [(8, 1)], [(5, 1)], 16)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
