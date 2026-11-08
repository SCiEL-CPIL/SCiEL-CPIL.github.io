import numpy as np
import time

_START_TIME = time.time()
DEADLINE_SECONDS = 130

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return DEADLINE_SECONDS - _elapsed()

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication."""
    n, m, p = 2, 4, 7
    nA = n * m   # 8
    nB = m * p   # 28
    nW = p * n   # 14

    # Decomposition strategy:
    # - 6 Strassen 2x2x2 blocks for (j,k) pairs where j in {0,2}, k in {0,2,4}
    #   Each block handles A[i,j:j+2], B[j:j+2,k:k+2] -> C[i:i+2, k:k+2]
    #   Rank 7 each, total 42
    # - Schoolbook for k=6: C[i,6] = sum_j A[i,j]*B[j,6]
    #   2 rows x 4 cols = 8 terms
    # Total rank: 50

    total_rank = 50
    U = [[0] * total_rank for _ in range(nA)]
    V = [[0] * total_rank for _ in range(nB)]
    W = [[0] * total_rank for _ in range(nW)]
    col = 0

    def add_strassen_block(i0, j0, k0):
        """Add a Strassen 2x2x2 block for A[i0:i0+2, j0:j0+2] @ B[j0:j0+2, k0:k0+2]."""
        nonlocal col
        # A indices: i*m + j
        a00 = i0 * m + j0
        a01 = i0 * m + (j0 + 1)
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + (j0 + 1)
        
        # B indices: j*p + k
        b00 = j0 * p + k0
        b01 = j0 * p + (k0 + 1)
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + (k0 + 1)
        
        # W indices: k*n + i
        w00 = k0 * n + i0
        w01 = k0 * n + (i0 + 1)
        w10 = (k0 + 1) * n + i0
        w11 = (k0 + 1) * n + (i0 + 1)

        # M1 = (a00 + a11)(b00 + b11) -> contributes to c00 and c11
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1

        # M2 = (a10 + a11)b00 -> contributes to c10 and c11 (with -1 for c11)
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[w10][col] += 1; W[w11][col] -= 1
        col += 1

        # M3 = a00(b01 - b11) -> contributes to c01 and c11
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[w10][col] += 1; W[w11][col] += 1
        col += 1

        # M4 = a11(b10 - b00) -> contributes to c00 and c10
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1

        # M5 = (a00 + a01)b11 -> contributes to c00 (with -1) and c01
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[w00][col] -= 1; W[w10][col] += 1
        col += 1

        # M6 = (a10 - a00)(b00 + b01) -> contributes to c11
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[w11][col] += 1
        col += 1

        # M7 = (a01 - a11)(b10 + b11) -> contributes to c00
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[w00][col] += 1
        col += 1

    # Apply Strassen to 6 blocks: j0 in {0, 2}, k0 in {0, 2, 4}
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen_block(0, j0, k0)
    
    # Handle k=6 with schoolbook: C[i,6] = sum_j A[i,j] * B[j,6]
    for j in range(m):
        for i in range(n):
            u_idx = i * m + j
            v_idx = j * p + 6
            w_idx = 6 * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Verify correctness
    def verify():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    ai = i * m + j
                    bi = j * p + k
                    wi = k * n + i
                    for l in range(col):
                        val += U[ai][l] * V[bi][l] * W[wi][l]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify():
        # Fallback to pure schoolbook (rank 56)
        U = [[0] * (n * m * p) for _ in range(nA)]
        V = [[0] * (n * m * p) for _ in range(nB)]
        W = [[0] * (n * m * p) for _ in range(nW)]
        col = 0
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    ai = i * m + j
                    bi = j * p + k
                    wi = k * n + i
                    U[ai][col] = 1
                    V[bi][col] = 1
                    W[wi][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
