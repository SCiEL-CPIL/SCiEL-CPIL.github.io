"""Matrix multiplication 4x4x5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy: Block decomposition.
    A (4x4) split into 2x2 blocks.
    B (4x5) split into:
      - B0 (4x2): Cols 0,1
      - B1 (4x2): Cols 2,3
      - B2 (4x1): Col 4
    
    C (4x5) is formed by:
      C[:, 0:2] = A @ B0
      C[:, 2:4] = A @ B1
      C[:, 4:5] = A @ B2
    
    For C[:, 0:2] and C[:, 2:4]:
      Each is a 4x2 result from 4x4 * 4x2.
      This can be seen as two 2x2 block column multiplications.
      e.g., C00 = A00*B00 + A01*B10 (2x2 result)
      We use Strassen (rank 7) for each 2x2*2x2 product.
      There are 4 such products for the first 2 cols, 4 for the next 2 cols.
      Total 8 Strassen blocks -> Rank 56.
    
    For C[:, 4:5]:
      This is a 4x1 result from 4x4 * 4x1.
      C0 = A00*B02 + A01*B12 (2x1 result)
      C1 = A10*B02 + A11*B12 (2x1 result)
      Each 2x2*2x1 product has rank 4.
      Total 4 such products -> Rank 16.
    
    Total Rank = 56 + 16 = 72.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    # Use a list to hold the mutable column offset for nested functions
    state = [0]
    
    rank_total = 72
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    def add_strassen_2x2(ra, ca, rb, cb):
        """Add 7 Strassen terms for 2x2 * 2x2."""
        # ra, ca: A block row, col (0 or 1)
        # rb, cb: B block row, col (0 or 1)
        # Note: cb here refers to the 2-column block index of B.
        # B block 0 covers cols 0,1. B block 1 covers cols 2,3.
        
        col = state[0]
        
        # M1: (a+d)*(e+h)
        U[2*ra * m + 2*ca, col] = 1.0
        U[(2*ra+1) * m + (2*ca+1), col] = 1.0
        V[2*rb * p + 2*cb, col] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col] = 1.0
        W[2*cb * n + 2*ra, col] = 1.0    # C11
        W[(2*cb+1) * n + (2*ra+1), col] = 1.0  # C22
        state[0] += 1
        
        # M2: (c+d)*e
        col = state[0]
        U[(2*ra+1) * m + 2*ca, col] = 1.0
        U[(2*ra+1) * m + (2*ca+1), col] = 1.0
        V[2*rb * p + 2*cb, col] = 1.0
        W[2*cb * n + (2*ra+1), col] = 1.0    # C21
        W[(2*cb+1) * n + (2*ra+1), col] = -1.0  # C22
        state[0] += 1
        
        # M3: a*(f-h)
        col = state[0]
        U[2*ra * m + 2*ca, col] = 1.0
        V[2*rb * p + (2*cb+1), col] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col] = -1.0
        W[(2*cb+1) * n + 2*ra, col] = 1.0    # C12
        W[(2*cb+1) * n + (2*ra+1), col] = 1.0  # C22
        state[0] += 1
        
        # M4: d*(g-e)
        col = state[0]
        U[(2*ra+1) * m + (2*ca+1), col] = 1.0
        V[(2*rb+1) * p + 2*cb, col] = 1.0
        V[2*rb * p + 2*cb, col] = -1.0
        W[2*cb * n + 2*ra, col] = 1.0    # C11
        W[2*cb * n + (2*ra+1), col] = 1.0  # C21
        state[0] += 1
        
        # M5: (a+b)*h
        col = state[0]
        U[2*ra * m + 2*ca, col] = 1.0
        U[2*ra * m + (2*ca+1), col] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col] = 1.0
        W[2*cb * n + 2*ra, col] = -1.0    # C11
        W[(2*cb+1) * n + 2*ra, col] = 1.0  # C12
        state[0] += 1
        
        # M6: (c-a)*(e+f)
        col = state[0]
        U[(2*ra+1) * m + 2*ca, col] = 1.0
        U[2*ra * m + 2*ca, col] = -1.0
        V[2*rb * p + 2*cb, col] = 1.0
        V[2*rb * p + (2*cb+1), col] = 1.0
        W[(2*cb+1) * n + (2*ra+1), col] = 1.0  # C22
        state[0] += 1
        
        # M7: (b-d)*(g+h)
        col = state[0]
        U[2*ra * m + (2*ca+1), col] = 1.0
        U[(2*ra+1) * m + (2*ca+1), col] = -1.0
        V[(2*rb+1) * p + 2*cb, col] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col] = 1.0
        W[2*cb * n + 2*ra, col] = 1.0    # C11
        state[0] += 1

    def add_direct_2x2_2x1(ra, ca, rb, cb_col):
        """Add 4 direct terms for 2x2 * 2x1.
        cb_col is the actual global column index of the single B column.
        """
        # A block (2x2) at ra, ca
        # B block (2x1) at rb, cb_col
        # Result (2x1) at ra, cb_col
        
        for i in range(2):
            for j in range(2):
                col = state[0]
                U[(2*ra + i) * m + (2*ca + j), col] = 1.0
                V[(2*rb + j) * p + cb_col, col] = 1.0
                W[cb_col * n + (2*ra + i), col] = 1.0
                state[0] += 1
    
    # Block 0: B cols 0,1 (cb=0)
    add_strassen_2x2(0, 0, 0, 0)  # A00 * B00
    add_strassen_2x2(0, 1, 1, 0)  # A01 * B10
    add_strassen_2x2(1, 0, 0, 0)  # A10 * B00
    add_strassen_2x2(1, 1, 1, 0)  # A11 * B10
    
    # Block 1: B cols 2,3 (cb=1)
    add_strassen_2x2(0, 0, 0, 1)  # A00 * B01
    add_strassen_2x2(0, 1, 1, 1)  # A01 * B11
    add_strassen_2x2(1, 0, 0, 1)  # A10 * B01
    add_strassen_2x2(1, 1, 1, 1)  # A11 * B11
    
    # Block 2: B col 4 (cb_col=4)
    add_direct_2x2_2x1(0, 0, 0, 4)  # A00 * B02 (B02 is col 4)
    add_direct_2x2_2x1(0, 1, 1, 4)  # A01 * B12 (B12 is col 4)
    add_direct_2x2_2x1(1, 0, 0, 4)  # A10 * B02
    add_direct_2x2_2x1(1, 1, 1, 4)  # A11 * B12
    
    used_rank = state[0]
    
    U = U[:, :used_rank]
    V = V[:, :used_rank]
    W = W[:, :used_rank]
    
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
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
