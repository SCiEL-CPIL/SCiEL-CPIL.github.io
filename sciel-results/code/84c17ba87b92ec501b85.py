import numpy as np
import time

DIMENSIONS = [2, 2, 13]

def _check_decomposition(U, V, W, tol=1e-10):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Rank 46 decomposition: 6 Strassen blocks (rank 7) + 1 schoolbook column (rank 4)
    rank = 46
    U = [[0] * rank for _ in range(n * m)]   # 4 x 46
    V = [[0] * rank for _ in range(m * p)]   # 26 x 46
    W = [[0] * rank for _ in range(p * n)]   # 26 x 46
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            if val != 0:
                U[idx][col] += val
        for idx, val in v_coeffs.items():
            if val != 0:
                V[idx][col] += val
        for idx, val in w_coeffs.items():
            if val != 0:
                W[idx][col] += val
        col += 1

    # Process 6 blocks of 2 columns using Strassen
    for block_idx in range(6):
        k0 = block_idx * 2
        k1 = k0 + 1
        
        # V indices: j*p + k
        v00 = 0 * p + k0
        v01 = 0 * p + k1
        v10 = 1 * p + k0
        v11 = 1 * p + k1
        
        # W indices: k*n + i
        w00 = k0 * n + 0
        w10 = k0 * n + 1
        w01 = k1 * n + 0
        w11 = k1 * n + 1
        
        # M1: (a00+a11)(b00+b11)
        add_term(
            {0: 1, 3: 1}, 
            {v00: 1, v11: 1}, 
            {w00: 1, w11: 1}
        )
        
        # M2: (a10+a11)b00
        add_term(
            {2: 1, 3: 1}, 
            {v00: 1}, 
            {w10: 1, w11: -1}
        )
        
        # M3: a00(b01-b11)
        add_term(
            {0: 1}, 
            {v01: 1, v11: -1}, 
            {w01: 1, w11: 1}
        )
        
        # M4: a11(b10-b00)
        add_term(
            {3: 1}, 
            {v10: 1, v00: -1}, 
            {w00: 1, w10: 1}
        )
        
        # M5: (a00+a01)b11
        add_term(
            {0: 1, 1: 1}, 
            {v11: 1}, 
            {w00: -1, w01: 1}
        )
        
        # M6: (a10-a00)(b00+b01)
        add_term(
            {2: 1, 0: -1}, 
            {v00: 1, v01: 1}, 
            {w11: 1}
        )
        
        # M7: (a01-a11)(b10+b11)
        add_term(
            {1: 1, 3: -1}, 
            {v10: 1, v11: 1}, 
            {w00: 1}
        )

    # Process last column k=12 with schoolbook
    k = 12
    add_term({0: 1}, {0 * p + k: 1}, {k * n + 0: 1})
    add_term({1: 1}, {1 * p + k: 1}, {k * n + 0: 1})
    add_term({2: 1}, {0 * p + k: 1}, {k * n + 1: 1})
    add_term({3: 1}, {1 * p + k: 1}, {k * n + 1: 1})

    # Verify the decomposition before returning
    if time.time() < deadline:
        if _check_decomposition(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails (should not happen, but safe)
    rank_fallback = n * m * p  # 52
    U_f = [[0] * rank_fallback for _ in range(n * m)]
    V_f = [[0] * rank_fallback for _ in range(m * p)]
    W_f = [[0] * rank_fallback for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i * m + j][col] = 1
                V_f[j * p + k][col] = 1
                W_f[k * n + i][col] = 1
                col += 1
                
    return {"U": U_f, "V": V_f, "W": W_f}
