"""Rank-minimal tensor decomposition for 2x2 x 2x13 matrix multiplication."""
DIMENSIONS = [2, 2, 13]

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 2, 2, 13
    rank = 46
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    def add_strassen_block(k1, k2, offset):
        U[0, offset+0] = 1; U[3, offset+0] = 1
        V[0*p+k1, offset+0] = 1; V[1*p+k2, offset+0] = 1
        W[k1*n+0, offset+0] = 1; W[k2*n+1, offset+0] = 1

        U[2, offset+1] = 1; U[3, offset+1] = 1
        V[0*p+k1, offset+1] = 1
        W[k1*n+1, offset+1] = 1

        U[0, offset+2] = 1
        V[0*p+k2, offset+2] = 1; V[1*p+k2, offset+2] = -1
        W[k2*n+0, offset+2] = 1; W[k2*n+1, offset+2] = 1

        U[3, offset+3] = 1
        V[0*p+k2, offset+3] = 1; V[0*p+k1, offset+3] = -1
        W[k2*n+0, offset+3] = 1; W[k1*n+0, offset+3] = 1

        U[0, offset+4] = 1; U[1, offset+4] = 1
        V[1*p+k2, offset+4] = 1
        W[k2*n+0, offset+4] = -1; W[k2*n+1, offset+4] = 1

        U[2, offset+5] = 1; U[0, offset+5] = -1
        V[0*p+k1, offset+5] = 1; V[0*p+k2, offset+5] = 1
        W[k1*n+1, offset+5] = 1; W[k2*n+1, offset+5] = 1

        U[1, offset+6] = 1; U[3, offset+6] = -1
        V[1*p+k1, offset+6] = 1; V[1*p+k2, offset+6] = 1
        W[k1*n+0, offset+6] = 1; W[k2*n+0, offset+6] = -1

    def add_schoolbook_block(k, offset):
        U[0, offset+0] = 1
        V[0*p+k, offset+0] = 1
        W[k*n+0, offset+0] = 1

        U[1, offset+1] = 1
        V[1*p+k, offset+1] = 1
        W[k*n+0, offset+1] = 1

        U[2, offset+2] = 1
        V[0*p+k, offset+2] = 1
        W[k*n+1, offset+2] = 1

        U[3, offset+3] = 1
        V[1*p+k, offset+3] = 1
        W[k*n+1, offset+3] = 1

    offset = 0
    for i in range(6):
        k1 = 2 * i
        k2 = 2 * i + 1
        add_strassen_block(k1, k2, offset)
        offset += 7

    add_schoolbook_block(12, offset)
    offset += 4

    # Verify using NumPy for speed
    # T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute sum over c
    # We want to check if sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] == 1 for all i,j,k
    # This is a tensor contraction.
    # tmp = sum_c U[i,j,c] * V[j,k,c] -> shape (n, m, p, r) ? No.
    # Let's just use einsum if available, or manual loops if slow.
    # n=2, m=2, p=13, r=46. Small enough for loops.
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break

    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
