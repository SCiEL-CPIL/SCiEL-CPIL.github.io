import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _add_term(U, V, W, u_coeffs, v_coeffs, w_coeffs, col_idx, n, m, p):
    for (i, j), val in u_coeffs.items():
        if val != 0:
            U[i * m + j][col_idx] += val
    for (j, k), val in v_coeffs.items():
        if val != 0:
            V[j * p + k][col_idx] += val
    for (k, i), val in w_coeffs.items():
        if val != 0:
            W[k * n + i][col_idx] += val

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Schoolbook rank is 64
    # Strassen-based rank is 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # We have 8 blocks of 2x2x2
    # j0 in {0, 2}
    # k0 in {0, 2, 4, 6}
    
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # Local indices:
            # i in {0, 1}
            # j in {j0, j0+1}
            # k in {k0, k0+1}
            
            # Term 1: M1 = (A00 + A11)(B00 + B11)
            _add_term(U, V, W,
                {(0, j0): 1, (1, j0+1): 1},
                {(j0, k0): 1, (j0+1, k0+1): 1},
                {(k0, 0): 1, (k0+1, 1): 1},
                col, n, m, p)
            col += 1
            
            # Term 2: M2 = (A10 + A11)B00
            _add_term(U, V, W,
                {(1, j0): 1, (1, j0+1): 1},
                {(j0, k0): 1},
                {(k0, 1): 1, (k0+1, 1): -1},
                col, n, m, p)
            col += 1
            
            # Term 3: M3 = A00(B01 - B11)
            _add_term(U, V, W,
                {(0, j0): 1},
                {(j0, k0+1): 1, (j0+1, k0+1): -1},
                {(k0+1, 0): 1, (k0+1, 1): 1},
                col, n, m, p)
            col += 1
            
            # Term 4: M4 = A11(B10 - B00)
            _add_term(U, V, W,
                {(1, j0+1): 1},
                {(j0+1, k0): 1, (j0, k0): -1},
                {(k0, 0): 1, (k0, 1): 1},
                col, n, m, p)
            col += 1
            
            # Term 5: M5 = (A00 + A01)B11
            _add_term(U, V, W,
                {(0, j0): 1, (0, j0+1): 1},
                {(j0+1, k0+1): 1},
                {(k0, 0): -1, (k0+1, 0): 1},
                col, n, m, p)
            col += 1
            
            # Term 6: M6 = (A10 - A00)(B00 + B01)
            _add_term(U, V, W,
                {(1, j0): 1, (0, j0): -1},
                {(j0, k0): 1, (j0, k0+1): 1},
                {(k0+1, 1): 1},
                col, n, m, p)
            col += 1
            
            # Term 7: M7 = (A01 - A11)(B10 + B11)
            _add_term(U, V, W,
                {(0, j0+1): 1, (1, j0+1): -1},
                {(j0+1, k0): 1, (j0+1, k0+1): 1},
                {(k0, 0): 1},
                col, n, m, p)
            col += 1
    
    # Validation step to ensure correctness
    # If validation fails, fallback to schoolbook (rank 64)
    # This is fast enough for small dimensions
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_recon = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m+j
                    v = j*p+k
                    w = k*n+i
                    U[u][col] = 1
                    V[v][col] = 1
                    W[w][col] = 1
                    col += 1

    return {
        "U": U,
        "V": V,
        "W": W
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
