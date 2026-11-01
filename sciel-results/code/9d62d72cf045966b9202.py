"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_val(mat, row, col, val):
        mat[row][col] += val

    # Block 1: Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {0,1}
    # A idx: 0,1,3,4; B idx: 0,1,5,6; C idx: 0,2,1,3
    # M1
    set_val(U, 0, 0, 1); set_val(U, 4, 0, 1)
    set_val(V, 0, 0, 1); set_val(V, 6, 0, 1)
    set_val(W, 0, 0, 1); set_val(W, 3, 0, 1)
    # M2
    set_val(U, 3, 1, 1); set_val(U, 4, 1, 1)
    set_val(V, 0, 1, 1)
    set_val(W, 1, 1, 1); set_val(W, 3, 1, -1)
    # M3
    set_val(U, 0, 2, 1)
    set_val(V, 1, 2, 1); set_val(V, 6, 2, -1)
    set_val(W, 2, 2, 1); set_val(W, 3, 2, 1)
    # M4
    set_val(U, 4, 3, 1)
    set_val(V, 5, 3, 1); set_val(V, 0, 3, -1)
    set_val(W, 0, 3, 1); set_val(W, 1, 3, 1)
    # M5
    set_val(U, 0, 4, 1); set_val(U, 1, 4, 1)
    set_val(V, 6, 4, 1)
    set_val(W, 0, 4, -1); set_val(W, 2, 4, 1)
    # M6
    set_val(U, 3, 5, 1); set_val(U, 0, 5, -1)
    set_val(V, 0, 5, 1); set_val(V, 1, 5, 1)
    set_val(W, 3, 5, 1)
    # M7
    set_val(U, 1, 6, 1); set_val(U, 4, 6, -1)
    set_val(V, 5, 6, 1); set_val(V, 6, 6, 1)
    set_val(W, 0, 6, 1)

    # Block 2: Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {2,3}
    # A idx: 0,1,3,4; B idx: 2,3,7,8; C idx: 4,6,5,7
    # M1
    set_val(U, 0, 7, 1); set_val(U, 4, 7, 1)
    set_val(V, 2, 7, 1); set_val(V, 8, 7, 1)
    set_val(W, 4, 7, 1); set_val(W, 7, 7, 1)
    # M2
    set_val(U, 3, 8, 1); set_val(U, 4, 8, 1)
    set_val(V, 2, 8, 1)
    set_val(W, 5, 8, 1); set_val(W, 7, 8, -1)
    # M3
    set_val(U, 0, 9, 1)
    set_val(V, 3, 9, 1); set_val(V, 8, 9, -1)
    set_val(W, 6, 9, 1); set_val(W, 7, 9, 1)
    # M4
    set_val(U, 4, 10, 1)
    set_val(V, 7, 10, 1); set_val(V, 2, 10, -1)
    set_val(W, 4, 10, 1); set_val(W, 5, 10, 1)
    # M5
    set_val(U, 0, 11, 1); set_val(U, 1, 11, 1)
    set_val(V, 8, 11, 1)
    set_val(W, 4, 11, -1); set_val(W, 6, 11, 1)
    # M6
    set_val(U, 3, 12, 1); set_val(U, 0, 12, -1)
    set_val(V, 2, 12, 1); set_val(V, 3, 12, 1)
    set_val(W, 7, 12, 1)
    # M7
    set_val(U, 1, 13, 1); set_val(U, 4, 13, -1)
    set_val(V, 7, 13, 1); set_val(V, 8, 13, 1)
    set_val(W, 4, 13, 1)

    # Block 3: Schoolbook 2x2x1 for i in {0,1}, j in {0,1}, k in {4}
    # A idx: 0,1,3,4; B idx: 4,9; C idx: 8,9
    # P1: A00*B04 -> C04
    set_val(U, 0, 14, 1)
    set_val(V, 4, 14, 1)
    set_val(W, 8, 14, 1)
    # P2: A01*B14 -> C04
    set_val(U, 1, 15, 1)
    set_val(V, 9, 15, 1)
    set_val(W, 8, 15, 1)
    # P3: A10*B04 -> C14
    set_val(U, 3, 16, 1)
    set_val(V, 4, 16, 1)
    set_val(W, 9, 16, 1)
    # P4: A11*B14 -> C14
    set_val(U, 4, 17, 1)
    set_val(V, 9, 17, 1)
    set_val(W, 9, 17, 1)

    # Part 2: Schoolbook 2x1x5 for i in {0,1}, j in {2}, k in {0..4}
    # A idx: 2,5; B idx: 10..14; C idx: 0,2,4,6,8 and 1,3,5,7,9
    # Q1: A02*B20 -> C00
    set_val(U, 2, 18, 1)
    set_val(V, 10, 18, 1)
    set_val(W, 0, 18, 1)
    # Q2: A02*B21 -> C01
    set_val(U, 2, 19, 1)
    set_val(V, 11, 19, 1)
    set_val(W, 2, 19, 1)
    # Q3: A02*B22 -> C02
    set_val(U, 2, 20, 1)
    set_val(V, 12, 20, 1)
    set_val(W, 4, 20, 1)
    # Q4: A02*B23 -> C03
    set_val(U, 2, 21, 1)
    set_val(V, 13, 21, 1)
    set_val(W, 6, 21, 1)
    # Q5: A02*B24 -> C04
    set_val(U, 2, 22, 1)
    set_val(V, 14, 22, 1)
    set_val(W, 8, 22, 1)
    # Q6: A12*B20 -> C10
    set_val(U, 5, 23, 1)
    set_val(V, 10, 23, 1)
    set_val(W, 1, 23, 1)
    # Q7: A12*B21 -> C11
    set_val(U, 5, 24, 1)
    set_val(V, 11, 24, 1)
    set_val(W, 3, 24, 1)
    # Q8: A12*B22 -> C12
    set_val(U, 5, 25, 1)
    set_val(V, 12, 25, 1)
    set_val(W, 5, 25, 1)
    # Q9: A12*B23 -> C13
    set_val(U, 5, 26, 1)
    set_val(V, 13, 26, 1)
    set_val(W, 7, 26, 1)
    # Q10: A12*B24 -> C14
    set_val(U, 5, 27, 1)
    set_val(V, 14, 27, 1)
    set_val(W, 9, 27, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
