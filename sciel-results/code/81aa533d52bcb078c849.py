import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 145.0

def _build_strassen_block(U, V, W, base_r, k0, k1, p):
    """Add a Strassen 2x2x2 block for columns k0,k1 at rank offset base_r."""
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3
    r5 = base_r + 4
    r6 = base_r + 5
    r7 = base_r + 6

    # P1: (A00+A11)(B0k0+B1k1)
    U[0][r1] = 1
    U[3][r1] = 1
    V[0 * p + k0][r1] = 1
    V[1 * p + k1][r1] = 1

    # P2: (A10+A11)B0k0
    U[2][r2] = 1
    U[3][r2] = 1
    V[0 * p + k0][r2] = 1

    # P3: A00(B0k1-B1k1)
    U[0][r3] = 1
    V[0 * p + k1][r3] = 1
    V[1 * p + k1][r3] = -1

    # P4: A11(B1k0-B0k0)
    U[3][r4] = 1
    V[1 * p + k0][r4] = 1
    V[0 * p + k0][r4] = -1

    # P5: (A00+A01)B1k1
    U[0][r5] = 1
    U[1][r5] = 1
    V[1 * p + k1][r5] = 1

    # P6: (A10-A00)(B0k0+B0k1)
    U[2][r6] = 1
    U[0][r6] = -1
    V[0 * p + k0][r6] = 1
    V[0 * p + k1][r6] = 1

    # P7: (A01-A11)(B1k0+B1k1)
    U[1][r7] = 1
    U[3][r7] = -1
    V[1 * p + k0][r7] = 1
    V[1 * p + k1][r7] = 1

    # Reconstruction: C[i,k] = sum_j A[i,j]*B[j,k]
    # C[0,k0] = A00*B0k0 + A01*B1k0
    # C[1,k0] = A10*B0k0 + A11*B1k0
    # C[0,k1] = A00*B0k1 + A01*B1k1
    # C[1,k1] = A10*B0k1 + A11*B1k1
    #
    # Using W indices: W[2*k+i, r] where i is row of C, k is col of C
    # W[2*k0][r] = coefficient for C[0,k0]
    # W[2*k0+1][r] = coefficient for C[1,k0]
    # W[2*k1][r] = coefficient for C[0,k1]
    # W[2*k1+1][r] = coefficient for C[1,k1]

    # C[0,k0] = A00*B0k0 + A01*B1k0
    # P1 contributes: A00*B0k0 (from A00 part) + A11*B1k1 (not relevant for k0)
    # Let me re-derive properly.
    #
    # P1 = (A00+A11)(B0k0+B1k1) = A00*B0k0 + A00*B1k1 + A11*B0k0 + A11*B1k1
    # P2 = (A10+A11)B0k0 = A10*B0k0 + A11*B0k0
    # P3 = A00(B0k1-B1k1) = A00*B0k1 - A00*B1k1
    # P4 = A11(B1k0-B0k0) = A11*B1k0 - A11*B0k0
    # P5 = (A00+A01)B1k1 = A00*B1k1 + A01*B1k1
    # P6 = (A10-A00)(B0k0+B0k1) = A10*B0k0 + A10*B0k1 - A00*B0k0 - A00*B0k1
    # P7 = (A01-A11)(B1k0+B1k1) = A01*B1k0 + A01*B1k1 - A11*B1k0 - A11*B1k1
    #
    # C[0,k0] = A00*B0k0 + A01*B1k0
    #   From P1: A00*B0k0 (coeff 1)
    #   From P4: A11*B1k0 (not A00)
    #   From P7: A01*B1k0 (coeff 1)
    #   So C[0,k0] = P1's A00*B0k0 + P7's A01*B1k0
    #   W[2*k0][r1] = 1 (from P1)
    #   W[2*k0][r7] = 1 (from P7)
    #
    # C[1,k0] = A10*B0k0 + A11*B1k0
    #   From P2: A10*B0k0 (coeff 1)
    #   From P4: A11*B1k0 (coeff 1)
    #   W[2*k0+1][r2] = 1
    #   W[2*k0+1][r4] = 1
    #
    # C[0,k1] = A00*B0k1 + A01*B1k1
    #   From P3: A00*B0k1 (coeff 1)
    #   From P5: A01*B1k1 (coeff 1)
    #   W[2*k1][r3] = 1
    #   W[2*k1][r5] = 1
    #
    # C[1,k1] = A10*B0k1 + A11*B1k1
    #   From P6: A10*B0k1 (coeff 1)
    #   From P1: A11*B1k1 (coeff 1)
    #   From P2: -A11*B0k0... wait
    #   Let me check: P1 has A11*B1k1 with coeff 1
    #   P6 has A10*B0k1 with coeff 1
    #   But also P1 has A11*B0k0 which shouldn't contribute to C[1,k1]
    #   Actually the W factor separates things. Let me re-examine.
    #
    # The key insight: W[i,k,r] multiplies the r-th term. So:
    # C[i,k] = sum_r W[2*k+i, r] * (sum_j A[i,j]*U[i*2+j, r]) * (sum_j B[j,k]*V[j*p+k, r])
    #
    # For r=r1: U gives A00+A11, V gives B0k0+B1k1
    #   Contribution to C[0,k0]: W[2*k0, r1] * (A00+A11) * (B0k0+B1k1)
    #   But we only want A00*B0k0 from this. The other terms must cancel.
    #
    # This is the standard Strassen verification. The W coefficients are chosen
    # so that all unwanted terms cancel across different r values.
    #
    # Standard Strassen W values:
    # W[2*k0, r1] = 1,  W[2*k0, r4] = 1,  W[2*k0, r5] = -1,  W[2*k0, r7] = 1
    # W[2*k0+1, r2] = 1,  W[2*k0+1, r4] = 1
    # W[2*k1, r3] = 1,  W[2*k1, r5] = 1
    # W[2*k1+1, r1] = 1,  W[2*k1+1, r2] = -1,  W[2*k1+1, r3] = 1,  W[2*k1+1, r6] = 1

    W[2 * k0][r1] = 1
    W[2 * k0][r4] = 1
    W[2 * k0][r5] = -1
    W[2 * k0][r7] = 1

    W[2 * k0 + 1][r2] = 1
    W[2 * k0 + 1][r4] = 1

    W[2 * k1][r3] = 1
    W[2 * k1][r5] = 1

    W[2 * k1 + 1][r1] = 1
    W[2 * k1 + 1][r2] = -1
    W[2 * k1 + 1][r3] = 1
    W[2 * k1 + 1][r6] = 1


def _verify_decomposition(U, V, W, n=2, m=2, p=15, rank=None):
    """Verify that the tensor decomposition is exact."""
    if rank is None:
        rank = len(U[0])
    # Check T[i,j,k] = sum_r U[i*2+j, r] * V[j*p+k, r] * W[k*2+i, r] = 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def construct():
    n, m, p = 2, 2, 15

    # Strategy: Use Strassen blocks for pairs of columns, plus handle remaining columns.
    # 15 columns = 7 pairs (14 cols) + 1 remaining col
    # Each Strassen block: rank 7
    # Trivial single column: rank 4
    # Total: 7*7 + 4 = 53

    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x rank
    V = [[0] * rank for _ in range(m * p)]  # 30 x rank
    W = [[0] * rank for _ in range(p * n)]  # 30 x rank

    # Build 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        base_r = 7 * block
        _build_strassen_block(U, V, W, base_r, k0, k1, p)

    # Handle column 14 trivially (rank 4)
    base_r = 49
    # C[0,14] = A00*B014 + A01*B114
    # C[1,14] = A10*B014 + A11*B114

    # Term 1: A00 * B014 -> C[0,14]
    U[0][base_r] = 1
    V[0 * p + 14][base_r] = 1
    W[14 * n + 0][base_r] = 1  # W[28, base_r]

    # Term 2: A01 * B114 -> C[0,14]
    U[1][base_r + 1] = 1
    V[1 * p + 14][base_r + 1] = 1
    W[14 * n + 0][base_r + 1] = 1  # W[28, base_r+1]

    # Term 3: A10 * B014 -> C[1,14]
    U[2][base_r + 2] = 1
    V[0 * p + 14][base_r + 2] = 1
    W[14 * n + 1][base_r + 2] = 1  # W[29, base_r+2]

    # Term 4: A11 * B114 -> C[1,14]
    U[3][base_r + 3] = 1
    V[1 * p + 14][base_r + 3] = 1
    W[14 * n + 1][base_r + 3] = 1  # W[29, base_r+3]

    # Verify
    if time.time() < _DEADLINE - 5:
        max_err = _verify_decomposition(U, V, W, n, m, p, rank)
        if max_err > 1e-10:
            # Fallback: return a known-good trivial decomposition (rank 60)
            rank = 60
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][r] = 1
                        V[j * p + k][r] = 1
                        W[k * n + i][r] = 1
                        r += 1

    return {"U": U, "V": V, "W": W}
