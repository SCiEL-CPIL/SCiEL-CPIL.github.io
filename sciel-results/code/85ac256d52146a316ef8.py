"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin

    # Rank 44 decomposition
    # Part 1: A1 (2x2) * B1 (2x8) using 4 Strassen blocks (28 mults)
    # Part 2: A2 (2x1) * B2 (1x8) using outer product (16 mults)

    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    col = 0

    # Part 1: Strassen for A1 * B1
    # A1 is A[0:2, 0:2] -> indices 0,1,3,4
    # B1 is B[0:2, 0:8] -> indices 0..7, 8..15
    # C1 is C[0:2, 0:8] -> indices 0..15 (k*n+i)

    for b in range(4):
        # Block b corresponds to B columns 2b, 2b+1
        b00_idx = 0 * p + 2 * b
        b01_idx = 0 * p + 2 * b + 1
        b10_idx = 1 * p + 2 * b
        b11_idx = 1 * p + 2 * b + 1

        # C indices for this block:
        c00_idx = (2 * b) * n + 0
        c01_idx = (2 * b + 1) * n + 0
        c10_idx = (2 * b) * n + 1
        c11_idx = (2 * b + 1) * n + 1

        # A indices for A1:
        a00_idx = 0
        a01_idx = 1
        a10_idx = 3
        a11_idx = 4

        # Strassen M1: (A00+A11)(B00+B11)
        U[a00_idx][col] += 1
        U[a11_idx][col] += 1
        V[b00_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # Strassen M2: (A10+A11)B00
        U[a10_idx][col] += 1
        U[a11_idx][col] += 1
        V[b00_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1

        # Strassen M3: A00(B01-B11)
        U[a00_idx][col] += 1
        V[b01_idx][col] += 1
        V[b11_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # Strassen M4: A11(B10-B00)
        U[a11_idx][col] += 1
        V[b10_idx][col] += 1
        V[b00_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

        # Strassen M5: (A00+A01)B11
        U[a00_idx][col] += 1
        U[a01_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1

        # Strassen M6: (A10-A00)(B00+B01)
        U[a10_idx][col] += 1
        U[a00_idx][col] -= 1
        V[b00_idx][col] += 1
        V[b01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # Strassen M7: (A01-A11)(B10+B11)
        U[a01_idx][col] += 1
        U[a11_idx][col] -= 1
        V[b10_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    # Part 2: Outer product for A2 * B2
    # A2 is A[0:2, 2] -> indices 2, 5
    # B2 is B[2, 0:8] -> indices 16..23
    # C2 is C[0:2, 0:8] -> indices 0..15

    for i in range(n):
        a_idx = i * m + 2
        for k in range(p):
            b_idx = 2 * p + k
            c_idx = k * n + i

            U[a_idx][col] += 1
            V[b_idx][col] += 1
            W[c_idx][col] += 1
            col += 1

    # Convert to numpy arrays for the return value
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    return {"U": U_arr, "V": V_arr, "W": W_arr}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = U.shape[1] if U is not None else 0
    # T[i,j,k] = 1 for all valid i,j,k (full matmul tensor)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = np.sum(U[i, :] * V[j, :] * W[k, :])
                if abs(actual - 1) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
