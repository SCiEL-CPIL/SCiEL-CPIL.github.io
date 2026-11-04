<<<<<<< SEARCH
def _build_strassen_2x2():
    """Build the 2x2 schoolbook decomposition (rank 4)."""
    U = np.zeros((4, 4), dtype=float)
    V = np.zeros((4, 4), dtype=float)
    W = np.zeros((4, 4), dtype=float)

    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i * 2 + j, col] = 1
                V[j * 2 + k, col] = 1
                W[k * 2 + i, col] = 1
                col += 1

    return U, V, W
=======
def _build_strassen_2x2():
    """Build the 2x2 Strassen decomposition (rank 7)."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Indices:
    # A: 0:A00, 1:A01, 2:A10, 3:A11
    # B: 0:B00, 1:B01, 2:B10, 3:B11
    # C: 0:C00, 2:C01, 1:C10, 3:C11 (W[k*2+i])
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[2, 0] += 1; W[1, 0] += -1; W[3, 0] += 1
    
    # M2 = (A10+A11)B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1; W[3, 1] += -1
    
    # M3 = A00(B01-B11)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] += -1
    W[2, 2] += 1; W[3, 2] += 1
    
    # M4 = A11(B10-B00)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] += -1
    W[0, 3] += 1; W[1, 3] += 1
    
    # M5 = (A00+A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] += -1; W[2, 4] += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] += 1; U[0, 5] += -1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] += 1; U[3, 6] += -1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    return U, V, W
>>>>>>> REPLACE
<<<<<<< SEARCH
def _build_6x6_decomposition():
    """
    Build the 6x6 decomposition using 3x3 schoolbook x 2x2 Strassen.
    Total rank = 27 * 7 = 189.
    """
    U22, V22, W22 = _build_strassen_2x2()
    U33, V33, W33 = _build_schoolbook_3x3()

    rank = 27 * 4  # 108

    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)

    for c33 in range(27):
        for c22 in range(4):
            c_final = c33 * 4 + c22
=======
def _build_6x6_decomposition():
    """
    Build the 6x6 decomposition using 3x3 schoolbook x 2x2 Strassen.
    Total rank = 27 * 7 = 189.
    """
    U22, V22, W22 = _build_strassen_2x2()
    U33, V33, W33 = _build_schoolbook_3x3()

    rank = 27 * 7  # 189

    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)

    for c33 in range(27):
        for c22 in range(7):
            c_final = c33 * 7 + c22
>>>>>>> REPLACE

I also need to update the docstring in `construct` to reflect rank 189.

