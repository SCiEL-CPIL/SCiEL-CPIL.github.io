import time
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    
    This is Strassen's algorithm, which is the known minimal rank for 2x2 matrix
    multiplication over the reals (and rationals).
    
    Indexing conventions:
      U shape: (n*m, rank) = (4, 7); row index = i*m + j  (A[i,j])
      V shape: (m*p, rank) = (4, 7); row index = j*p + k  (B[j,k])
      W shape: (p*n, rank) = (4, 7); row index = k*n + i  (C[i,k], trace convention)
    
    The tensor identity is:
      T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i,j,k] = 1 if j is the connecting index in C[i,k] = sum_j A[i,j]B[j,k], else 0.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    rank = 7
    
    # Strassen's 7 intermediate products.
    # Each product r is defined by coefficients for U, V, W.
    # Keys are flat indices:
    #   U: i*m + j  (a_ij)
    #   V: j*p + k  (b_jk)
    #   W: k*n + i  (c_ik)
    
    # Product 1: (a00 + a11)(b00 + b11)
    # U: a00(0), a11(3) -> 1
    # V: b00(0), b11(3) -> 1
    # W: c00(0), c11(3) -> 1
    u1 = {0: 1, 3: 1}
    v1 = {0: 1, 3: 1}
    w1 = {0: 1, 3: 1}
    
    # Product 2: (a10 + a11)b00
    # U: a10(2), a11(3) -> 1
    # V: b00(0) -> 1
    # W: c10(1), c11(3) -> 1, -1
    u2 = {2: 1, 3: 1}
    v2 = {0: 1}
    w2 = {1: 1, 3: -1}
    
    # Product 3: a00(b01 - b11)
    # U: a00(0) -> 1
    # V: b01(1), b11(3) -> 1, -1
    # W: c01(2), c11(3) -> 1, 1
    u3 = {0: 1}
    v3 = {1: 1, 3: -1}
    w3 = {2: 1, 3: 1}
    
    # Product 4: a11(b10 - b00)
    # U: a11(3) -> 1
    # V: b10(2), b00(0) -> 1, -1
    # W: c00(0), c10(1) -> 1, 1
    u4 = {3: 1}
    v4 = {2: 1, 0: -1}
    w4 = {0: 1, 1: 1}
    
    # Product 5: (a00 + a01)b11
    # U: a00(0), a01(1) -> 1
    # V: b11(3) -> 1
    # W: c00(0), c01(2) -> -1, 1
    u5 = {0: 1, 1: 1}
    v5 = {3: 1}
    w5 = {0: -1, 2: 1}
    
    # Product 6: (a10 - a00)(b00 + b01)
    # U: a10(2), a00(0) -> 1, -1
    # V: b00(0), b01(1) -> 1
    # W: c11(3) -> 1
    u6 = {2: 1, 0: -1}
    v6 = {0: 1, 1: 1}
    w6 = {3: 1}
    
    # Product 7: (a01 - a11)(b10 + b11)
    # U: a01(1), a11(3) -> 1, -1
    # V: b10(2), b11(3) -> 1
    # W: c00(0) -> 1
    u7 = {1: 1, 3: -1}
    v7 = {2: 1, 3: 1}
    w7 = {0: 1}
    
    products = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r, (u_coeffs, v_coeffs, w_coeffs) in enumerate(products):
        for idx, val in u_coeffs.items():
            U[idx][r] = val
        for idx, val in v_coeffs.items():
            V[idx][r] = val
        for idx, val in w_coeffs.items():
            W[idx][r] = val
            
    if time.time() > deadline:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
