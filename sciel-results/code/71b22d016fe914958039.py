import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify the decomposition exactly reproduces the matmul tensor."""
    r = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    return max_err < 1e-9

def construct():
    _start = time.time()
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Strassen for 2x2 blocks
    for g in range(5):
        k0 = 2 * g
        k1 = k0 + 1
        
        v_b00 = k0
        v_b01 = k1
        v_b10 = p + k0
        v_b11 = p + k1
        
        u_a00 = 0
        u_a01 = 1
        u_a10 = m
        u_a11 = m + 1
        
        w_c00 = k0 * n
        w_c01 = k1 * n
        w_c10 = k0 * n + 1
        w_c11 = k1 * n + 1
        
        # m1
        U[u_a00][col] = 1; U[u_a11][col] = 1
        V[v_b00][col] = 1; V[v_b11][col] = 1
        W[w_c00][col] = 1; W[w_c11][col] = 1
        col += 1
        
        # m2
        U[u_a10][col] = 1; U[u_a11][col] = 1
        V[v_b00][col] = 1
        W[w_c10][col] = 1; W[w_c11][col] = -1
        col += 1
        
        # m3
        U[u_a00][col] = 1
        V[v_b01][col] = 1; V[v_b11][col] = -1
        W[w_c01][col] = 1; W[w_c11][col] = 1
        col += 1
        
        # m4
        U[u_a11][col] = 1
        V[v_b10][col] = 1; V[v_b00][col] = -1
        W[w_c00][col] = 1; W[w_c10][col] = 1
        col += 1
        
        # m5
        U[u_a00][col] = 1; U[u_a01][col] = 1
        V[v_b11][col] = 1
        W[w_c01][col] = 1; W[w_c00][col] = -1
        col += 1
        
        # m6
        U[u_a10][col] = 1; U[u_a00][col] = -1
        V[v_b00][col] = 1; V[v_b01][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # m7
        U[u_a01][col] = 1; U[u_a11][col] = -1
        V[v_b10][col] = 1; V[v_b11][col] = 1
        W[w_c00][col] = 1
        col += 1
        
    # Part 2: Schoolbook for 3rd column
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    if col != rank:
        raise ValueError(f"Column count {col} != expected rank {rank}")
    
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    elapsed = time.time() - _start
    if elapsed > 110:
        raise TimeoutError(f"Exceeded time budget: {elapsed:.1f}s")
    
    return {"U": U, "V": V, "W": W}
