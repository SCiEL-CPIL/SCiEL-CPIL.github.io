"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 15
    pn = p * n  # 15
    
    # Precompute indices for fast access
    # U index: a = i*m + j
    # V index: b = j*p + k
    # W index: c = k*n + i
    
    # Target tensor T is non-zero at (a, b, c) if:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # T[a, b, c] = 1
    
    def get_U_system(a, V, W, r):
        # a = i*m + j
        i = a // m
        j = a % m
        # Equations for k in 0..p-1
        # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
        # b[k] = 1
        A = np.zeros((p, r))
        for k in range(p):
            b_idx = j * p + k
            c_idx = k * n + i
            A[k, :] = V[b_idx, :] * W[c_idx, :]
        b_vec = np.ones(p)
        return A, b_vec

    def get_V_system(b, U, W, r):
        # b = j*p + k
        j = b // p
        k = b % p
        # Equations for i in 0..n-1
        # M[i, c] = U[i*m+j, c] * W[k*n+i, c]
        # b[i] = 1
        A = np.zeros((n, r))
        for i in range(n):
            a_idx = i * m + j
            c_idx = k * n + i
            A[i, :] = U[a_idx, :] * W[c_idx, :]
        b_vec = np.ones(n)
        return A, b_vec

    def get_W_system(c, U, V, r):
        # c = k*n + i
        k = c // n
        i = c % n
        # Equations for j in 0..m-1
        # M[j, c] = U[i*m+j, c] * V[j*p+k, c]
        # b[j] = 1
        A = np.zeros((m, r))
        for j in range(m):
            a_idx = i * m + j
            b_idx = j * p + k
            A[j, :] = U[a_idx, :] * V[b_idx, :]
        b_vec = np.ones(m)
        return A, b_vec

    def check_error(U, V, W, r):
        err = 0.0
        for i in range(n):
            for j in range(m):
                a_idx = i * m + j
                for k in range(p):
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U[a_idx, :], V[b_idx, :] * W[c_idx, :])
                    if abs(val - 1.0) > err:
                        err = abs(val - 1.0)
        return err

    def try_rank(r, seed=42):
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, r)
        V = rng.randn(mp, r)
        W = rng.randn(pn, r)
        
        for it in range(1000):
            # Update U
            for a in range(nm):
                A, b_vec = get_U_system(a, V, W, r)
                U[a, :], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
            
            # Update V
            for b in range(mp):
                A, b_vec = get_V_system(b, U, W, r)
                V[b, :], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
            
            # Update W
            for c in range(pn):
                A, b_vec = get_W_system(c, U, V, r)
                W[c, :], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
            
            if it % 10 == 0:
                err = check_error(U, V, W, r)
                if err < 1e-8:
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    err2 = check_error(U_r, V_r, W_r, r)
                    if err2 < 1e-8:
                        return U_r, V_r, W_r
        return None

    # Try increasing ranks
    deadline = time.time() + 110  # Leave 40s margin for safety
    best_result = None
    
    # Start from a reasonable guess. Schoolbook is 45.
    # Try to find something better.
    for r in range(30, 46):
        if time.time() > deadline:
            break
        for seed in range(20):
            if time.time() > deadline:
                break
            result = try_rank(r, seed)
            if result is not None:
                U_r, V_r, W_r = result
                U_list = [[float(x) for x in row] for row in U_r]
                V_list = [[float(x) for x in row] for row in V_r]
                W_list = [[float(x) for x in row] for row in W_r]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook (rank 45)
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
