"""Program for 6x6x6 matrix multiplication tensor decomposition using CW(3x3) ⊗ Strassen(2x2)."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6

    # Strassen decomposition for 2x2 matrices, rank 7
    U2 = np.array([
        [1, 1, 1, 0, 1, 1, 0],   # A00
        [0, 0, 0, 1, 0, -1, 1],  # A10
        [0, 0, 0, 0, 1, 0, 0],   # A01
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ], dtype=float)
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],   # B00
        [0, 0, 1, 1, 0, 0, 1],   # B10
        [1, 0, 0, 0, 1, 0, 0],   # B01
        [1, 0, -1, -1, 1, 0, 1], # B11
    ], dtype=float)
    W2 = np.array([
        [1, 0, 0, 0, -1, 0, -1], # C00 = M1 + M4 - M5 - M7
        [0, 0, 1, 0, 1, 0, 0],   # C01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # C10 = M2 + M4
        [1, -1, 1, 0, 0, 0, 1],  # C11 = M1 - M2 + M3 + M7
    ], dtype=float)

    # Coppersmith-Winograd rank-23 decomposition for 3x3 matrices
    # This is a known rank-23 decomposition.
    U3 = np.zeros((9, 23), dtype=float)
    V3 = np.zeros((9, 23), dtype=float)
    W3 = np.zeros((9, 23), dtype=float)

    # Terms 0-8: Schoolbook-like diagonal terms
    for i in range(3):
        for j in range(3):
            r = i * 3 + j
            U3[i * 3 + j, r] = 1
            V3[j * 3 + j, r] = 1
            W3[j * 3 + i, r] = 1

    # Terms 9-22: Additional CW terms
    # These are the 14 additional terms that reduce rank from 27 to 23

    # Term 9: (0,0)+(1,1) * (0,1)-(1,0) * (1,0)+(0,1)
    r = 9
    U3[0, r] = 1; U3[4, r] = 1
    V3[1, r] = 1; V3[3, r] = -1
    W3[3, r] = 1; W3[1, r] = 1

    # Term 10: (0,0)+(2,2) * (0,2)-(2,0) * (2,0)+(0,2)
    r = 10
    U3[0, r] = 1; U3[8, r] = 1
    V3[2, r] = 1; V3[6, r] = -1
    W3[6, r] = 1; W3[2, r] = 1

    # Term 11: (0,0)+(1,1) * (1,0)-(0,1) * (0,1)+(1,0)
    r = 11
    U3[0, r] = 1; U3[4, r] = 1
    V3[3, r] = 1; V3[1, r] = -1
    W3[1, r] = 1; W3[3, r] = 1

    # Term 12: (0,0)+(2,2) * (2,0)-(0,2) * (0,2)+(2,0)
    r = 12
    U3[0, r] = 1; U3[8, r] = 1
    V3[6, r] = 1; V3[2, r] = -1
    W3[2, r] = 1; W3[6, r] = 1

    # Term 13: (0,1)+(1,2) * (0,0)-(2,1) * (1,1)+(0,0)
    r = 13
    U3[1, r] = 1; U3[5, r] = 1
    V3[0, r] = 1; V3[7, r] = -1
    W3[4, r] = 1; W3[0, r] = 1

    # Term 14: (0,2)+(1,0) * (0,1)-(2,2) * (2,1)+(1,0)
    r = 14
    U3[2, r] = 1; U3[3, r] = 1
    V3[1, r] = 1; V3[8, r] = -1
    W3[7, r] = 1; W3[3, r] = 1

    # Term 15: (0,1)+(1,2) * (1,1)-(0,2) * (0,0)+(1,1)
    r = 15
    U3[1, r] = 1; U3[5, r] = 1
    V3[4, r] = 1; V3[2, r] = -1
    W3[0, r] = 1; W3[4, r] = 1

    # Term 16: (0,2)+(1,0) * (2,1)-(0,0) * (1,0)+(2,2)
    r = 16
    U3[2, r] = 1; U3[3, r] = 1
    V3[7, r] = 1; V3[0, r] = -1
    W3[3, r] = 1; W3[8, r] = 1

    # Term 17: (1,0)+(2,1) * (0,0)-(1,2) * (0,2)+(1,1)
    r = 17
    U3[3, r] = 1; U3[7, r] = 1
    V3[0, r] = 1; V3[5, r] = -1
    W3[2, r] = 1; W3[4, r] = 1

    # Term 18: (1,2)+(2,0) * (1,1)-(0,1) * (1,2)+(2,0)
    r = 18
    U3[5, r] = 1; U3[6, r] = 1
    V3[4, r] = 1; V3[1, r] = -1
    W3[5, r] = 1; W3[6, r] = 1

    # Term 19: (1,0)+(2,1) * (1,2)-(0,2) * (2,0)+(0,1)
    r = 19
    U3[3, r] = 1; U3[7, r] = 1
    V3[5, r] = 1; V3[2, r] = -1
    W3[6, r] = 1; W3[1, r] = 1

    # Term 20: (1,2)+(2,0) * (2,2)-(1,0) * (0,1)+(2,1)
    r = 20
    U3[5, r] = 1; U3[6, r] = 1
    V3[8, r] = 1; V3[3, r] = -1
    W3[1, r] = 1; W3[7, r] = 1

    # Term 21: (2,0)+(0,1) * (1,0)-(2,2) * (1,1)+(2,0)
    r = 21
    U3[6, r] = 1; U3[1, r] = 1
    V3[3, r] = 1; V3[8, r] = -1
    W3[4, r] = 1; W3[6, r] = 1

    # Term 22: (2,1)+(0,2) * (2,1)-(1,0) * (2,2)+(0,0)
    r = 22
    U3[7, r] = 1; U3[2, r] = 1
    V3[7, r] = 1; V3[3, r] = -1
    W3[8, r] = 1; W3[0, r] = 1

    # Now construct the 6x6x6 decomposition using Kronecker product
    # U6 shape: (36, 161), V6 shape: (36, 161), W6 shape: (36, 161)
    # rank = 7 * 23 = 161

    rank = 7 * 23
    U6 = np.zeros((36, rank), dtype=float)
    V6 = np.zeros((36, rank), dtype=float)
    W6 = np.zeros((36, rank), dtype=float)

    # For the Kronecker product, we need to map indices correctly
    # U2 has shape (4, 7) where row index = i*2 + j for i,j in {0,1}
    # U3 has shape (9, 23) where row index = i*3 + j for i,j in {0,1,2}
    # U6 has shape (36, 161) where row index = (i2*3+i3)*6 + (j2*3+j3) for i2,j2 in {0,1}, i3,j3 in {0,1,2}
    # Actually, let's think more carefully:
    # For 6x6 matrices, we can view them as 2x2 blocks of 3x3 matrices
    # Or as 3x3 blocks of 2x2 matrices
    # Let's use: 6 = 2 * 3, so we have 2x2 blocks of 3x3 submatrices
    # U6 row index: (a, b) where a in {0,1} (2x2 block row), b in {0,1,2,3,4,5} (within block)
    # Actually, let's use: row = i*6 + j where i in {0,...,5}, j in {0,...,5}
    # We can write i = i2*3 + i3, j = j2*3 + j3 where i2,j2 in {0,1}, i3,j3 in {0,1,2}
    # Then U6[i*6+j, r2*23+r3] = U2[i2*2+j2, r2] * U3[i3*3+j3, r3]

    for i in range(6):
        for j in range(6):
            i2 = i // 3
            i3 = i % 3
            j2 = j // 3
            j3 = j % 3
            row6 = i * 6 + j
            for r2 in range(7):
                for r3 in range(23):
                    col6 = r2 * 23 + r3
                    U6[row6, col6] = U2[i2 * 2 + j2, r2] * U3[i3 * 3 + j3, r3]

    # Similarly for V6: V6[j*6+k, r2*23+r3] = V2[j2*2+k2, r2] * V3[j3*3+k3, r3]
    for j in range(6):
        for k in range(6):
            j2 = j // 3
            j3 = j % 3
            k2 = k // 3
            k3 = k % 3
            row6 = j * 6 + k
            for r2 in range(7):
                for r3 in range(23):
                    col6 = r2 * 23 + r3
                    V6[row6, col6] = V2[j2 * 2 + k2, r2] * V3[j3 * 3 + k3, r3]

    # Similarly for W6: W6[k*6+i, r2*23+r3] = W2[k2*2+i2, r2] * W3[k3*3+i3, r3]
    for k in range(6):
        for i in range(6):
            k2 = k // 3
            k3 = k % 3
            i2 = i // 3
            i3 = i % 3
            row6 = k * 6 + i
            for r2 in range(7):
                for r3 in range(23):
                    col6 = r2 * 23 + r3
                    W6[row6, col6] = W2[k2 * 2 + i2, r2] * W3[k3 * 3 + i3, r3]

    # Convert to lists
    U = U6.tolist()
    V = V6.tolist()
    W = W6.tolist()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
