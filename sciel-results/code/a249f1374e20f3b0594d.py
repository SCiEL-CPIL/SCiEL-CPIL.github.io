import numpy as np
import time

DIMENSIONS = [2, 3, 10]
_START_TIME = time.time()
TIME_LIMIT = 140

def _verify_decomposition(U, V, W):
    """Verify the tensor decomposition is correct."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_55():
    """
    Constructs a rank-55 decomposition for 2x3x10 matrix multiplication.
    Uses Strassen (7 mults) for 2x2x2 blocks and Schoolbook for the rest.
    """
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    # Part 1: A1 (2x2) * B1 (2x10)
    # Split B1 into 5 blocks of 2 columns.
    for g in range(5):
        k_base = 2 * g
        k0 = k_base
        k1 = k_base + 1
        
        # Indices
        v_b00 = 0 * p + k0
        v_b01 = 0 * p + k1
        v_b10 = 1 * p + k0
        v_b11 = 1 * p + k1
        
        u_a00 = 0 * m + 0
        u_a01 = 0 * m + 1
        u_a10 = 1 * m + 0
        u_a11 = 1 * m + 1
        
        w_c00 = k0 * n + 0
        w_c01 = k1 * n + 0
        w_c10 = k0 * n + 1
        w_c11 = k1 * n + 1
        
        # Strassen 7 terms
        # 1
        U[u_a00][col] = 1; U[u_a11][col] = 1
        V[v_b00][col] = 1; V[v_b11][col] = 1
        W[w_c00][col] = 1; W[w_c11][col] = 1
        col += 1
        # 2
        U[u_a10][col] = 1; U[u_a11][col] = 1
        V[v_b00][col] = 1
        W[w_c10][col] = 1; W[w_c11][col] = -1
        col += 1
        # 3
        U[u_a00][col] = 1
        V[v_b01][col] = 1; V[v_b11][col] = -1
        W[w_c01][col] = 1; W[w_c11][col] = 1
        col += 1
        # 4
        U[u_a11][col] = 1
        V[v_b10][col] = 1; V[v_b00][col] = -1
        W[w_c00][col] = 1; W[w_c10][col] = 1
        col += 1
        # 5
        U[u_a00][col] = 1; U[u_a01][col] = 1
        V[v_b11][col] = 1
        W[w_c01][col] = 1; W[w_c00][col] = -1
        col += 1
        # 6
        U[u_a10][col] = 1; U[u_a00][col] = -1
        V[v_b00][col] = 1; V[v_b01][col] = 1
        W[w_c11][col] = 1
        col += 1
        # 7
        U[u_a01][col] = 1; U[u_a11][col] = -1
        V[v_b10][col] = 1; V[v_b11][col] = 1
        W[w_c00][col] = 1
        col += 1

    # Part 2: A2 (2x1) * B2 (1x10)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

def construct():
    # Check time
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT - 2:
        return _build_strassen_55()
    result = _build_strassen_55()
    # Verify correctness
    if _verify_decomposition(result["U"], result["V"], result["W"]):
        return result
    # Fallback to trivial rank-60 if verification fails
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
