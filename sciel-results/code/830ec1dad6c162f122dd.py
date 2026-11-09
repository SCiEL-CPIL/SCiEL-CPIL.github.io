"""6x6x6 matrix multiplication via 3x3 tiling of 2x2 Strassen (rank 7) blocks."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _strassen22():
    """Return (U2, V2, W2) for 2x2x2 Strassen (rank 7), official trace convention.

    U[ i*2+j , c], V[ j*2+k , c], W[ k*2+i , c].
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    # A11 = A[0,0] -> U[0,0]; A12 = A[0,1] -> U[0,1]
    # A21 = A[1,0] -> U[1,0]; A22 = A[1,1] -> U[1,1]
    U[0][0] = 1
    U[0][1] = 1
    U[1][0] = 1
    U[1][1] = 1

    # B11 = B[0,0] -> V[0,0]; B12 = B[0,1] -> V[0,1]
    # B21 = B[1,0] -> V[1,0]; B22 = B[1,1] -> V[1,1]
    V[0][0] = 1
    V[0][1] = 1
    V[1][0] = 1
    V[1][1] = 1

    # Strassen products (row, col) -> sign
    S = [
        (0, 0, +1),  # M1 = A11*(B11+B22)
        (0, 1, +1),  # M2 = (A11+A12)*B22
        (1, 0, +1),  # M3 = A12*(B21-B22)
        (1, 1, +1),  # M4 = A21*(B11+B22)
        (0, 0, +1),  # M5 = (A11-A21)*(B11+B21)
        (1, 1, +1),  # M6 = (A12-A22)*(B12+B22)
        (0, 1, +1),  # M7 = (A22-A11)*(B11+B21)
    ]
    # B block-row, B block-col -> sign
    Vsigns = [
        (+1, +1),  # M1
        (0, +1),   # M2
        (1, +1),   # M3
        (0, +1),   # M4
        (0, +1),   # M5
        (1, +1),   # M6
        (0, +1),   # M7
    ]
    for c, (ar, ac) in enumerate(S):
        # A part: A[ar, ac]
        U[ar * 2 + ac][c] = +1
        # B part: (B11+B22) for c in {0,3,6}, else single B
        if c in (0, 3, 6):
            V[0 * 2 + 0][c] = +1
            V[1 * 2 + 1][c] = +1
        else:
            br, bc = Vsigns[c]
            V[br * 2 + bc][c] = +1

    # W: result block (r,c) = sum over 7 products with signs
    # C11 = M1 - M3 + M5 + M7 ; C12 = M2 + M4
    # C21 = M6 + M4           ; C22 = M1 - M2 + M3 + M6
    Wcoeff = {
        (0, 0): {0: +1, 2: -1, 4: +1, 6: +1},
        (0, 1): {1: +1, 3: +1},
        (1, 0): {3: +1, 5: +1},
        (1, 1): {0: +1, 1: -1, 2: +1, 5: +1},
    }
    for (r, c), mapping in Wcoeff.items():
        for prod, sign in mapping.items():
            W[c * 2 + r][prod] = sign

    return U, V, W


def construct():
    deadline = time.time() + 110.0  # safety margin well under 120s
    n, m, p = DIMENSIONS
    b = 2            # block size
    t = n // b       # 3 blocks per side

    U2, V2, W2 = _strassen22()

    rank = t * t * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for bi in range(t):
        for bj in range(t):
            base = (bi * t + bj) * 7
            for a in range(b):
                for c2 in range(b):
                    for u in range(4):
                        val = U2[u][c2]
                        if val:
                            U[(bi * b + a) * m + (bj * b + u // b)] \
                                [base + (u % b)] = val
            for c2 in range(b):
                for v in range(4):
                    val = V2[v][c2]
                    if val:
                        V[(bj * b + v // b) * p + (bj * b + v % b)] \
                            [base + c2] = val
            for c2 in range(b):
                for w in range(4):
                    val = W2[w][c2]
                    if val:
                        W[(bj * b + w % b) * n + (bi * b + w // b)] \
                            [base + c2] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
