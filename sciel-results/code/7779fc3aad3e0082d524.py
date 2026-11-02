"""Matrix multiplication 2x3x10 tensor decomposition via Kronecker factorization."""
import numpy as np

DIMENSIONS = [2, 3, 10]


def _rank11_2x3x2():
    """Rank-11 exact decomposition of 2x3x2 (A:2x3, B:3x2, C:2x2).

    Split: A = [A1(2x2) | A2(2x1)], B = [B1(2x2); B2(1x2)].
    A1*B1 uses Strassen (rank 7), A2*B2 uses schoolbook (rank 4).
    """
    U = np.zeros((6, 11))  # A entries indexed i*3+j
    V = np.zeros((6, 11))  # B entries indexed j*2+k
    W = np.zeros((4, 11))  # C entries indexed k*2+i

    # Strassen for A1*B1: A1 has entries (0,0),(0,1),(1,0),(1,1)
    # B1 has entries (0,0),(0,1),(1,0),(1,1)
    # C1 has entries (0,0),(0,1),(1,0),(1,1)
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # A1 entry indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # B1 entry indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C1 entry indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3

    # M1: (A00+A11)(B00+B11)
    U[0, 0] += 1; U[4, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1

    # M2: (A10+A11)B00
    U[3, 1] += 1; U[4, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1

    # M3: A00(B10-B11)
    U[0, 2] += 1
    V[2, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1; W[3, 2] += 1

    # M4: A11(B01-B00)
    U[4, 3] += 1
    V[1, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1; W[2, 3] += 1

    # M5: (A00+A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1; W[1, 4] += 1

    # M6: (A10-A00)(B00+B01)
    U[3, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1

    # M7: (A01-A11)(B10+B11)
    U[1, 6] += 1; U[4, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1

    # A2*B2: A2 has entries (0,2)->2, (1,2)->5
    # B2 has entries (2,0)->4, (2,1)->5
    # C2 has entries (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    for a_idx, i in [(2, 0), (5, 1)]:
        for b_idx, k in [(4, 0), (5, 1)]:
            col = 7 + i * 2 + k
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[k * 2 + i, col] = 1

    return U, V, W


def _rank5_1x1x5():
    """Rank-5 decomposition of 1x1x5 (trivial: each output element is one product)."""
    U = np.eye(1, 5)
    V = np.eye(5, 5)
    W = np.eye(5, 5)
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    assert n == 2 and m == 3 and p == 10

    # Get rank-11 decomposition of 2x3x2
    U1, V1, W1 = _rank11_2x3x2()
    # Get rank-5 decomposition of 1x1x5
    U2, V2, W2 = _rank5_1x1x5()

    # Kronecker product: rank = 11 * 5 = 55
    # U1 is 6x11 (A: 2x3 -> row index i*3+j), U2 is 1x5
    # For 2x3x10: A is 2x3 (row index i*3+j), B is 3x10 (row index j*10+k), C is 10x2 (row index k*2+i)
    #
    # We decompose B's k dimension: k = k1*5 + k2 where k1 in {0,1}, k2 in {0,1,2,3,4}
    # And C's k dimension: k = k1*5 + k2
    #
    # U: (n*m, rank) = (6, 55). U[i*3+j, c] where c = c1*5 + c2
    #   U[i*3+j, c1*5+c2] = U1[i*3+j, c1] * U2[0, c2] = U1[i*3+j, c1] * delta(0,c2)
    #   Wait, U2 is 1x5 with U2[0,c2]=1 for all c2 (since it's the identity-like for 1x1x5)
    #   Actually for 1x1x5: A has 1 entry, B has 5 entries, C has 5 entries.
    #   U2[0, c2] = 1 for all c2 (the single A entry multiplies with each B entry)
    #   V2[j2, c2] = delta(j2, c2) (each B entry is selected)
    #   W2[k2, c2] = delta(k2, c2) (each C entry receives the product)
    #
    # So for the Kronecker:
    # U[i*3+j, c1*5+c2] = U1[i*3+j, c1] * U2[0, c2] = U1[i*3+j, c1]
    # V[j*10+k, c1*5+c2] where k = k1*5+k2, so row = j*10 + k1*5 + k2
    #   = V1[j*2+k1, c1] * V2[k2, c2] = V1[j*2+k1, c1] * delta(k2,c2)
    # W[k*2+i, c1*5+c2] where k = k1*5+k2, so row = k1*5*2 + k2*2 + i = (k1*5+k2)*2+i
    #   = W1[k1*2+i, c1] * W2[k2, c2] = W1[k1*2+i, c1] * delta(k2,c2)

    rank = 55
    U = np.zeros((6, rank))
    V = np.zeros((30, rank))
    W = np.zeros((20, rank))

    for c1 in range(11):
        for c2 in range(5):
            c = c1 * 5 + c2
            # U
            for i in range(2):
                for j in range(3):
                    val = U1[i * 3 + j, c1] * U2[0, c2]
                    if val != 0:
                        U[i * 3 + j, c] = val
            # V
            for j in range(3):
                for k1 in range(2):
                    for k2 in range(5):
                        val = V1[j * 2 + k1, c1] * V2[k2, c2]
                        if val != 0:
                            V[j * 10 + k1 * 5 + k2, c] = val
            # W
            for k1 in range(2):
                for i in range(2):
                    for k2 in range(5):
                        val = W1[k1 * 2 + i, c1] * W2[k2, c2]
                        if val != 0:
                            W[(k1 * 5 + k2) * 2 + i, c] = val

    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    # Round to integers (should all be -1, 0, or 1)
    U_list = [[int(round(v)) for v in row] for row in U_list]
    V_list = [[int(round(v)) for v in row] for row in V_list]
    W_list = [[int(round(v)) for v in row] for row in W_list]

    return {"U": U_list, "V": V_list, "W": W_list}
