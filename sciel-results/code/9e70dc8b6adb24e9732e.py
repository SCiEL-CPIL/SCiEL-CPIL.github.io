"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Decomposition of 4x5 x 5x5 with rank 92
    # Partition A (4x5) into rows 2+2, cols 3+2
    # A11 (2x3), A12 (2x2)
    # A21 (2x3), A22 (2x2)
    
    # Partition B (5x5) into rows 3+2, cols 3+2
    # B11 (3x3), B12 (3x2)
    # B21 (2x3), B22 (2x2)
    
    # C (4x5)
    # C11 (2x3) = A11@B11 + A12@B21
    # C12 (2x2) = A11@B12 + A12@B22
    # C21 (2x3) = A21@B11 + A22@B21
    # C22 (2x2) = A21@B12 + A22@B22
    
    terms = []
    
    def add_schoolbook(a_i_range, a_j_range, b_j_range, b_k_range, c_i_range, c_k_range):
        for i in c_i_range:
            for j in a_j_range:
                for k in b_k_range:
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    terms.append((u_idx, v_idx, w_idx))
    
    # A11@B11 -> C11
    add_schoolbook([0,1], [0,1,2], [0,1,2], [0,1,2], [0,1], [0,1,2])
    # A12@B21 -> C11
    add_schoolbook([0,1], [3,4], [3,4], [0,1,2], [0,1], [0,1,2])
    # A11@B12 -> C12
    add_schoolbook([0,1], [0,1,2], [0,1,2], [3,4], [0,1], [3,4])
    # A12@B22 -> C12
    add_schoolbook([0,1], [3,4], [3,4], [3,4], [0,1], [3,4])
    # A21@B11 -> C21
    add_schoolbook([2,3], [0,1,2], [0,1,2], [0,1,2], [2,3], [0,1,2])
    # A22@B21 -> C21
    add_schoolbook([2,3], [3,4], [3,4], [0,1,2], [2,3], [0,1,2])
    # A21@B12 -> C22
    add_schoolbook([2,3], [0,1,2], [0,1,2], [3,4], [2,3], [3,4])
    # A22@B22 -> C22
    add_schoolbook([2,3], [3,4], [3,4], [3,4], [2,3], [3,4])
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_idx, v_idx, w_idx) in enumerate(terms):
        U[u_idx][r] = 1
        V[v_idx][r] = 1
        W[w_idx][r] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
