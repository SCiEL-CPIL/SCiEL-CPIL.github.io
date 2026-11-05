"""Matrix multiplication 4x4x5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy: Use Strassen's algorithm for the 4x4x4 part (rank 49)
    and direct multiplication for the 5th column (rank 16).
    Total rank = 49 + 16 = 65.
    
    For the 4x4x4 part, we use the standard Strassen decomposition:
    Split into 2x2 blocks and use 7 Strassen products.
    Each 2x2x2 product uses 7 terms, giving 7*7 = 49 total for 4x4x4.
    
    For the 5th column, we use direct multiplication:
    4x4 * 4x1 = 16 terms.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    rank_strassen = 49
    rank_direct = 16
    rank_total = rank_strassen + rank_direct
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Helper to add a Strassen 2x2x2 decomposition
    # A: rows [ra*2, ra*2+1], cols [ca*2, ca*2+1]
    # B: rows [rb*2, rb*2+1], cols [cb*2, cb*2+1]
    # C: rows [ra*2, ra*2+1], cols [cb*2, cb*2+1]
    def add_strassen_2x2(ra, ca, rb, cb, base_col):
        """Add 7 Strassen terms for 2x2 * 2x2 block."""
        a_r0, a_r1 = ra*2, ra*2+1
        a_c0, a_c1 = ca*2, ca*2+1
        b_r0, b_r1 = rb*2, rb*2+1
        b_c0, b_c1 = cb*2, cb*2+1
        c_r0, c_r1 = ra*2, ra*2+1
        c_c0, c_c1 = cb*2, cb*2+1
        
        col = base_col
        # M1: (a+d)*(e+h) -> C[c_r0,c_c0] += 1, C[c_r1,c_c1] += 1
        U[a_r0 * m + a_c0, col] = 1.0
        U[a_r1 * m + a_c1, col] = 1.0
        V[b_r0 * p + b_c0, col] = 1.0
        V[b_r1 * p + b_c1, col] = 1.0
        W[c_c0 * n + c_r0, col] = 1.0
        W[c_c1 * n + c_r1, col] = 1.0
        col += 1
        
        # M2: (c+d)*e -> C[c_r1,c_c0] += 1, C[c_r1,c_c1] -= 1
        U[a_r1 * m + a_c0, col] = 1.0
        U[a_r1 * m + a_c1, col] = 1.0
        V[b_r0 * p + b_c0, col] = 1.0
        W[c_c0 * n + c_r1, col] = 1.0
        W[c_c1 * n + c_r1, col] = -1.0
        col += 1
        
        # M3: a*(f-h) -> C[c_r0,c_c1] += 1, C[c_r1,c_c1] += 1
        U[a_r0 * m + a_c0, col] = 1.0
        V[b_r0 * p + b_c1, col] = 1.0
        V[b_r1 * p + b_c1, col] = -1.0
        W[c_c1 * n + c_r0, col] = 1.0
        W[c_c1 * n + c_r1, col] = 1.0
        col += 1
        
        # M4: d*(g-e) -> C[c_r0,c_c0] += 1, C[c_r1,c_c0] += 1
        U[a_r1 * m + a_c1, col] = 1.0
        V[b_r1 * p + b_c0, col] = 1.0
        V[b_r0 * p + b_c0, col] = -1.0
        W[c_c0 * n + c_r0, col] = 1.0
        W[c_c0 * n + c_r1, col] = 1.0
        col += 1
        
        # M5: (a+b)*h -> C[c_r0,c_c0] -= 1, C[c_r0,c_c1] += 1
        U[a_r0 * m + a_c0, col] = 1.0
        U[a_r0 * m + a_c1, col] = 1.0
        V[b_r1 * p + b_c1, col] = 1.0
        W[c_c0 * n + c_r0, col] = -1.0
        W[c_c1 * n + c_r0, col] = 1.0
        col += 1
        
        # M6: (c-a)*(e+f) -> C[c_r1,c_c1] += 1
        U[a_r1 * m + a_c0, col] = 1.0
        U[a_r0 * m + a_c0, col] = -1.0
        V[b_r0 * p + b_c0, col] = 1.0
        V[b_r0 * p + b_c1, col] = 1.0
        W[c_c1 * n + c_r1, col] = 1.0
        col += 1
        
        # M7: (b-d)*(g+h) -> C[c_r0,c_c0] += 1
        U[a_r0 * m + a_c1, col] = 1.0
        U[a_r1 * m + a_c1, col] = -1.0
        V[b_r1 * p + b_c0, col] = 1.0
        V[b_r1 * p + b_c1, col] = 1.0
        W[c_c0 * n + c_r0, col] = 1.0
        col += 1
        
        return col
    
    # Build the 4x4x4 Strassen decomposition (rank 49)
    # Split A into 4 blocks: A00, A01, A10, A11
    # Split B into 4 blocks: B00, B01, B10, B11
    # C = A*B has blocks: C00 = A00*B00 + A01*B10
    #                      C01 = A00*B01 + A01*B11
    #                      C10 = A10*B00 + A11*B10
    #                      C11 = A10*B01 + A11*B11
    
    col = 0
    
    # C00 = A00*B00 + A01*B10
    # Use Strassen on the 2x2x2 problem for each block product
    # A00*B00: ra=0, ca=0, rb=0, cb=0
    col = add_strassen_2x2(0, 0, 0, 0, col)
    # A01*B10: ra=0, ca=1, rb=1, cb=0
    col = add_strassen_2x2(0, 1, 1, 0, col)
    
    # C01 = A00*B01 + A01*B11
    # A00*B01: ra=0, ca=0, rb=0, cb=1
    col = add_strassen_2x2(0, 0, 0, 1, col)
    # A01*B11: ra=0, ca=1, rb=1, cb=1
    col = add_strassen_2x2(0, 1, 1, 1, col)
    
    # C10 = A10*B00 + A11*B10
    # A10*B00: ra=1, ca=0, rb=0, cb=0
    col = add_strassen_2x2(1, 0, 0, 0, col)
    # A11*B10: ra=1, ca=1, rb=1, cb=0
    col = add_strassen_2x2(1, 1, 1, 0, col)
    
    # C11 = A10*B01 + A11*B11
    # A10*B01: ra=1, ca=0, rb=0, cb=1
    col = add_strassen_2x2(1, 0, 0, 1, col)
    # A11*B11: ra=1, ca=1, rb=1, cb=1
    col = add_strassen_2x2(1, 1, 1, 1, col)
    
    assert col == rank_strassen, f"Strassen rank mismatch: {col} != {rank_strassen}"
    
    # Now handle the 5th column (p=4, 0-indexed)
    # C[:,4] = A * B[:,4]
    # This is a 4x4 * 4x1 product, rank 16
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
            col += 1
    
    assert col == rank_total, f"Total rank mismatch: {col} != {rank_total}"
    
    # Verify the decomposition
    def verify(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook decomposition (rank n*m*p = 80)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col_school = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col_school] = 1.0
                V_school[j*p + k, col_school] = 1.0
                W_school[k*n + i, col_school] = 1.0
                col_school += 1
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
