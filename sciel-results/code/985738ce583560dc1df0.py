"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-7 exact tensor decomposition for 2x2 matrix multiplication
    using the Strassen algorithm.

    The matrix multiplication tensor T_{i,j,k} corresponds to:
      C[i,k] = sum_j A[i,j] * B[j,k]

    We need U (4 x 7), V (4 x 7), W (4 x 7) such that:
      T = sum_r U[:,r] * V[:,r] * W[:,r]  (outer product sum)

    Where the indices are:
      U index: i*m + j = i*2 + j  (rows of A)
      V index: j*p + k = j*2 + k  (columns of B, row index j)
      W index: k*n + i = k*2 + i  (output index, following trace convention W[k*n+i, r])

    Strassen's 7 products for 2x2 matrices:
      A = [[a11, a12], [a21, a22]]
      B = [[b11, b12], [b21, b22]]

      P1 = (a11 + a22)(b11 + b22)
      P2 = (a21 + a22)b11
      P3 = a11(b12 - b22)
      P4 = a22(b21 - b11)
      P5 = (a11 + a12)b22
      P6 = (a21 - a11)(b11 + b12)
      P7 = (a12 - a22)(b21 + b22)

      C11 = P1 + P4 - P5 + P7
      C12 = P3 + P5
      C21 = P2 + P4
      C22 = P1 - P2 + P3 + P6

    For each product r, we extract the coefficients for A (U), B (V), and C (W).
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    rank = 7

    # U: shape (4, 7), index i*2+j
    # V: shape (4, 7), index j*2+k
    # W: shape (4, 7), index k*2+i  (trace convention: W[k*n+i, r])

    # Let's define the 7 Strassen products with their A, B, C coefficient vectors.
    # A coefficients: [a11, a12, a21, a22] -> indices [0,1,2,3]
    # B coefficients: [b11, b12, b21, b22] -> indices [0,1,2,3]
    # C coefficients: [c11, c12, c21, c22] -> indices [0,1,2,3]
    #   where c11 = C[0,0], c12 = C[0,1], c21 = C[1,0], c22 = C[1,1]
    #   W index for C[i,k] is k*n + i = k*2 + i
    #   So c11 -> W[0], c12 -> W[1], c21 -> W[2], c22 -> W[3]

    # Product 1: (a11 + a22)(b11 + b22)
    # A: [1, 0, 0, 1]
    # B: [1, 0, 0, 1]
    # C: [1, 0, 0, 1]  (c11 += P1, c22 += P1)

    # Product 2: (a21 + a22)b11
    # A: [0, 0, 1, 1]
    # B: [1, 0, 0, 0]
    # C: [0, 0, 1, -1] (c21 += P2, c22 -= P2)

    # Product 3: a11(b12 - b22)
    # A: [1, 0, 0, 0]
    # B: [0, 1, 0, -1]
    # C: [0, 1, 0, 0] (c12 += P3)

    # Product 4: a22(b21 - b11)
    # A: [0, 0, 0, 1]
    # B: [-1, 0, 1, 0]
    # C: [1, 0, 1, 0] (c11 += P4, c21 += P4)

    # Product 5: (a11 + a12)b22
    # A: [1, 1, 0, 0]
    # B: [0, 0, 0, 1]
    # C: [-1, 1, 0, 0] (c11 -= P5, c12 += P5)

    # Product 6: (a21 - a11)(b11 + b12)
    # A: [-1, 0, 1, 0]
    # B: [1, 1, 0, 0]
    # C: [0, 0, 0, 1] (c22 += P6)

    # Product 7: (a12 - a22)(b21 + b22)
    # A: [0, 1, 0, -1]
    # B: [0, 0, 1, 1]
    # C: [1, 0, 0, 0] (c11 += P7)

    U_coeffs = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, 1],   # P2
        [1, 0, 0, 0],   # P3
        [0, 0, 0, 1],   # P4
        [1, 1, 0, 0],   # P5
        [-1, 0, 1, 0],  # P6
        [0, 1, 0, -1],  # P7
    ]

    V_coeffs = [
        [1, 0, 0, 1],   # P1
        [1, 0, 0, 0],   # P2
        [0, 1, 0, -1],  # P3
        [-1, 0, 1, 0],  # P4
        [0, 0, 0, 1],   # P5
        [1, 1, 0, 0],   # P6
        [0, 0, 1, 1],   # P7
    ]

    W_coeffs = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, -1],  # P2
        [0, 1, 0, 0],   # P3
        [1, 0, 1, 0],   # P4
        [-1, 1, 0, 0],  # P5
        [0, 0, 0, 1],   # P6
        [1, 0, 0, 0],   # P7
    ]

    U = [list(row) for row in U_coeffs]  # 4 x 7
    V = [list(row) for row in V_coeffs]  # 4 x 7
    W = [list(row) for row in W_coeffs]  # 4 x 7

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
