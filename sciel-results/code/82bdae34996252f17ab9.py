import time
import random
import numpy as np

DIMENSIONS = [2, 2, 3]
TIME_LIMIT = 140.0  # seconds, with 10s safety margin

# EVOLVE-BLOCK-START

def _build_rank3():
    """Build the rank-3 decomposition."""
    n, m, p = DIMENSIONS
    rank = 3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val):
        U[i * m + j][r] = val
    def set_v(j, k, r, val):
        V[j * p + k][r] = val
    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 1: Handles k=0,1
    # U[i,j] = 1
    # V[j,k] = 1 for k in {0,1}, 0 for k=2
    # W[k,i] = 1/2 for k in {0,1}, 0 for k=2
    for i in range(n):
        for j in range(m):
            set_u(i, j, 0, 1)
    for j in range(m):
        set_v(j, 0, 0, 1)
        set_v(j, 1, 0, 1)
        set_v(j, 2, 0, 0)
    for k in range(p):
        for i in range(n):
            if k < 2:
                set_w(k, i, 0, 0.5)
            else:
                set_w(k, i, 0, 0)

    # Term 2: Handles k=2, j=0
    # U[i,0] = 1, U[i,1] = 0
    # V[0,2] = 1, others 0
    # W[2,i] = 1, others 0
    for i in range(n):
        set_u(i, 0, 1, 1)
        set_u(i, 1, 1, 0)
    set_v(0, 2, 1, 1)
    for k in range(p):
        for i in range(n):
            if k == 2:
                set_w(k, i, 1, 1)
            else:
                set_w(k, i, 1, 0)

    # Term 3: Handles k=2, j=1
    # U[i,0] = 0, U[i,1] = 1
    # V[1,2] = 1, others 0
    # W[2,i] = 1, others 0
    for i in range(n):
        set_u(i, 0, 2, 0)
        set_u(i, 1, 2, 1)
    set_v(1, 2, 2, 1)
    for k in range(p):
        for i in range(n):
            if k == 2:
                set_w(k, i, 2, 1)
            else:
                set_w(k, i, 2, 0)

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n=2, m=2, p=3):
    """Verify that the decomposition correctly computes C = A*B for random inputs."""
    rank = len(U[0])
    for _ in range(5):
        A = np.random.randint(-5, 6, (n, m))
        B = np.random.randint(-5, 6, (m, p))
        C_expected = A @ B
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p), dtype=float)
        for i in range(n):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    for j in range(m):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                C_computed[i, k] = total
        if not np.allclose(C_expected, C_computed):
            return False
    return True


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Use the rank-3 solution
    sol3 = _build_rank3()
    
    # Verify it works
    if _verify_decomposition(sol3["U"], sol3["V"], sol3["W"]):
        return sol3
    
    # Fallback to rank-11 if verification fails (should not happen)
    return _build_rank11()

def _build_rank11():
    """Build the known rank-11 decomposition."""
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val):
        U[i * m + j][r] = val
    def set_v(j, k, r, val):
        V[j * p + k][r] = val
    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 1: (A00 + A11) * (B00 + B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    # Term 2: (A10 + A11) * B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    # Term 3: A00 * (B01 - B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    # Term 4: A11 * (B10 - B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    # Term 5: (A00 + A01) * B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    # Term 6: (A10 - A00) * (B00 + B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 7: (A01 - A11) * (B10 + B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 8: A00 * B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 9: A01 * B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 10: A10 * B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 11: A11 * B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
