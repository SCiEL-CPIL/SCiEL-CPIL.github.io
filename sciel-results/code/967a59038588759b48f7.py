"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 8
    rank = 68
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
        col += 1

    # 1. Handle k=2 terms (Outer product of A[:,2] and B[2,:])
    # A[i, 2] * B[2, j]
    for i in range(3):
        for j in range(8):
            add_term(
                [(i * m + 2, 1)],
                [(2 * p + j, 1)],
                [(j * n + i, 1)]
            )
            
    # 2. Handle k in {0, 1} terms using Strassen on 2x2 blocks
    # Split B columns into 4 groups of 2
    for g in range(4):
        j0 = 2 * g
        j1 = 2 * g + 1
        
        # Strassen Term 1: M1 = A[0,0] * (B[0,j0] + B[1,j1])
        add_term(
            [(0 * m + 0, 1)],
            [(0 * p + j0, 1), (1 * p + j1, 1)],
            [(j0 * n + 0, 1), (j1 * n + 1, 1)]
        )
        
        # Strassen Term 2: M2 = A[1,0] * (B[0,j0] + B[0,j1])
        add_term(
            [(1 * m + 0, 1)],
            [(0 * p + j0, 1), (0 * p + j1, 1)],
            [(j0 * n + 1, 1)]
        )
        
        # Strassen Term 3: M3 = A[0,1] * (B[1,j0] - B[1,j1])
        add_term(
            [(0 * m + 1, 1)],
            [(1 * p + j0, 1), (1 * p + j1, -1)],
            [(j1 * n + 0, 1), (j1 * n + 1, -1)]
        )
        
        # Strassen Term 4: M4 = A[1,1] * B[1,j1]
        add_term(
            [(1 * m + 1, 1)],
            [(1 * p + j1, 1)],
            [(j0 * n + 0, 1), (j0 * n + 1, 1)]
        )
        
        # Strassen Term 5: M5 = (A[0,0] + A[1,1]) * (B[0,j0] - B[1,j1])
        add_term(
            [(0 * m + 0, 1), (1 * m + 1, 1)],
            [(0 * p + j0, 1), (1 * p + j1, -1)],
            [(j0 * n + 0, -1), (j1 * n + 0, 1)]
        )
        
        # Strassen Term 6: M6 = (A[0,1] + A[1,1]) * (B[1,j0] - B[0,j0])
        add_term(
            [(0 * m + 1, 1), (1 * m + 1, 1)],
            [(1 * p + j0, 1), (0 * p + j0, -1)],
            [(j1 * n + 1, 1)]
        )
        
        # Strassen Term 7: M7 = (A[0,0] + A[1,0]) * (B[0,j1] + B[1,j1])
        add_term(
            [(0 * m + 0, 1), (1 * m + 0, 1)],
            [(0 * p + j1, 1), (1 * p + j1, 1)],
            [(j0 * n + 0, 1), (j1 * n + 1, -1)]
        )
        
        # Schoolbook for row i=2
        # C[2, j0] = A[2,0]B[0,j0] + A[2,1]B[1,j0]
        add_term(
            [(2 * m + 0, 1)],
            [(0 * p + j0, 1)],
            [(j0 * n + 2, 1)]
        )
        
        add_term(
            [(2 * m + 1, 1)],
            [(1 * p + j0, 1)],
            [(j0 * n + 2, 1)]
        )
        
        # C[2, j1] = A[2,0]B[0,j1] + A[2,1]B[1,j1]
        add_term(
            [(2 * m + 0, 1)],
            [(0 * p + j1, 1)],
            [(j1 * n + 2, 1)]
        )
        
        add_term(
            [(2 * m + 1, 1)],
            [(1 * p + j1, 1)],
            [(j1 * n + 2, 1)]
        )
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
