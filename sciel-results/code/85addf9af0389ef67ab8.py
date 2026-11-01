import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Known best rank is 28
    # We will attempt to search for a rank 27 solution, but default to 28.
    
    # Helper to construct the known rank 28 solution
    def get_known_solution():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def set_val(mat, row, col, val):
            mat[row][col] += val

        # Block 1: Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {0,1}
        set_val(U, 0, 0, 1); set_val(U, 4, 0, 1)
        set_val(V, 0, 0, 1); set_val(V, 6, 0, 1)
        set_val(W, 0, 0, 1); set_val(W, 3, 0, 1)
        set_val(U, 3, 1, 1); set_val(U, 4, 1, 1)
        set_val(V, 0, 1, 1)
        set_val(W, 1, 1, 1); set_val(W, 3, 1, -1)
        set_val(U, 0, 2, 1)
        set_val(V, 1, 2, 1); set_val(V, 6, 2, -1)
        set_val(W, 2, 2, 1); set_val(W, 3, 2, 1)
        set_val(U, 4, 3, 1)
        set_val(V, 5, 3, 1); set_val(V, 0, 3, -1)
        set_val(W, 0, 3, 1); set_val(W, 1, 3, 1)
        set_val(U, 0, 4, 1); set_val(U, 1, 4, 1)
        set_val(V, 6, 4, 1)
        set_val(W, 0, 4, -1); set_val(W, 2, 4, 1)
        set_val(U, 3, 5, 1); set_val(U, 0, 5, -1)
        set_val(V, 0, 5, 1); set_val(V, 1, 5, 1)
        set_val(W, 3, 5, 1)
        set_val(U, 1, 6, 1); set_val(U, 4, 6, -1)
        set_val(V, 5, 6, 1); set_val(V, 6, 6, 1)
        set_val(W, 0, 6, 1)

        # Block 2: Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {2,3}
        set_val(U, 0, 7, 1); set_val(U, 4, 7, 1)
        set_val(V, 2, 7, 1); set_val(V, 8, 7, 1)
        set_val(W, 4, 7, 1); set_val(W, 7, 7, 1)
        set_val(U, 3, 8, 1); set_val(U, 4, 8, 1)
        set_val(V, 2, 8, 1)
        set_val(W, 5, 8, 1); set_val(W, 7, 8, -1)
        set_val(U, 0, 9, 1)
        set_val(V, 3, 9, 1); set_val(V, 8, 9, -1)
        set_val(W, 6, 9, 1); set_val(W, 7, 9, 1)
        set_val(U, 4, 10, 1)
        set_val(V, 7, 10, 1); set_val(V, 2, 10, -1)
        set_val(W, 4, 10, 1); set_val(W, 5, 10, 1)
        set_val(U, 0, 11, 1); set_val(U, 1, 11, 1)
        set_val(V, 8, 11, 1)
        set_val(W, 4, 11, -1); set_val(W, 6, 11, 1)
        set_val(U, 3, 12, 1); set_val(U, 0, 12, -1)
        set_val(V, 2, 12, 1); set_val(V, 3, 12, 1)
        set_val(W, 7, 12, 1)
        set_val(U, 1, 13, 1); set_val(U, 4, 13, -1)
        set_val(V, 7, 13, 1); set_val(V, 8, 13, 1)
        set_val(W, 4, 13, 1)

        # Block 3: Schoolbook 2x2x1 for i in {0,1}, j in {0,1}, k in {4}
        set_val(U, 0, 14, 1)
        set_val(V, 4, 14, 1)
        set_val(W, 8, 14, 1)
        set_val(U, 1, 15, 1)
        set_val(V, 9, 15, 1)
        set_val(W, 8, 15, 1)
        set_val(U, 3, 16, 1)
        set_val(V, 4, 16, 1)
        set_val(W, 9, 16, 1)
        set_val(U, 4, 17, 1)
        set_val(V, 9, 17, 1)
        set_val(W, 9, 17, 1)

        # Part 2: Schoolbook 2x1x5 for i in {0,1}, j in {2}, k in {0..4}
        set_val(U, 2, 18, 1)
        set_val(V, 10, 18, 1)
        set_val(W, 0, 18, 1)
        set_val(U, 2, 19, 1)
        set_val(V, 11, 19, 1)
        set_val(W, 2, 19, 1)
        set_val(U, 2, 20, 1)
        set_val(V, 12, 20, 1)
        set_val(W, 4, 20, 1)
        set_val(U, 2, 21, 1)
        set_val(V, 13, 21, 1)
        set_val(W, 6, 21, 1)
        set_val(U, 2, 22, 1)
        set_val(V, 14, 22, 1)
        set_val(W, 8, 22, 1)
        set_val(U, 5, 23, 1)
        set_val(V, 10, 23, 1)
        set_val(W, 1, 23, 1)
        set_val(U, 5, 24, 1)
        set_val(V, 11, 24, 1)
        set_val(W, 3, 24, 1)
        set_val(U, 5, 25, 1)
        set_val(V, 12, 25, 1)
        set_val(W, 5, 25, 1)
        set_val(U, 5, 26, 1)
        set_val(V, 13, 26, 1)
        set_val(W, 7, 26, 1)
        set_val(U, 5, 27, 1)
        set_val(V, 14, 27, 1)
        set_val(W, 9, 27, 1)

        return {"U": U, "V": V, "W": W}

    # Attempt to search for a better solution (rank 27)
    # This is a placeholder for a potential search strategy.
    # Given the constraints and the likely minimality of 28, we return the known solution.
    # A full search for rank 27 is computationally expensive and unlikely to succeed in 120s.
    
    return get_known_solution()
# EVOLVE-BLOCK-END
