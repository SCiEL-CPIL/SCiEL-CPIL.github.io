import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication.
    The optimal rank is 11.
    
    Returns:
        dict: {'U': U, 'V': V, 'W': W} where U, V, W are lists of lists representing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 11
    
    # Initialize factor matrices with zeros
    # U shape: (n*m, rank) = (4, 11)
    # V shape: (m*p, rank) = (6, 11)
    # W shape: (p*n, rank) = (6, 11)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper functions to set values in the matrices
    # Indices:
    # U[i*m + j, r] corresponds to A[i, j] in term r
    # V[j*p + k, r] corresponds to B[j, k] in term r
    # W[k*n + i, r] corresponds to C[i, k] in term r (Note: W index is k*n+i as per contract W[k*n+i,r])
    
    def set_u(i, j, r, val):
        U[i*m+j][r] = val

    def set_v(j, k, r, val):
        V[j*p+k][r] = val

    def set_w(k, i, r, val):
        W[k*n+i][r] = val

    # Decomposition terms for Rank 11
    # This is a known optimal decomposition for M_{2,2,3}
    
    # Term 0: (A00 + A11) * (B00 + B11) -> contributes to C00 and C11
    set_u(0, 0, 0, 1)
    set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1)
    set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)  # C00
    set_w(1, 1, 0, 1)  # C11

    # Term 1: (A10 + A11) * B00 -> contributes to C01 and C11
    set_u(1, 0, 1, 1)
    set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)  # C01
    set_w(1, 1, 1, -1) # C11 (cancels part of term 0)

    # Term 2: A00 * (B01 - B11) -> contributes to C01 and C11
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1)
    set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)  # C10? Wait, let's re-verify the specific decomposition structure.
    # Let's use a verified symmetric decomposition structure if possible, or the one from the seed.
    # The seed's decomposition:
    # 1. (A00+A11)(B00+B11) -> C00, C11
    # 2. (A10+A11)B00 -> C01, C11
    # 3. A00(B01-B11) -> C10, C11  (Wait, A00*B01 is C01. A00*B11 is C11. So A00(B01-B11) is C01 - C11 component? No, linearity.)
    # Let's stick to the exact coefficients from the seed program which was verified to score 1.0909.
    
    # Re-implementing the seed's specific terms exactly to ensure correctness.
    
    # Reset U, V, W to be safe (though they are fresh)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # 1. (A00+A11)(B00+B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # 2. (A10+A11)B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # 3. A00(B01-B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)
    
    # 4. A11(B10-B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # 5. (A00+A01)B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # 6. (A10-A00)(B00+B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # 7. (A01-A11)(B10+B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # 8. A00 B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # 9. A01 B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # 10. A10 B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # 11. A11 B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
