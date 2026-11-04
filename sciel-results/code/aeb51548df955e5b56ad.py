"""Matrix multiplication 4x4x5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy: Block decomposition using Strassen for 2x2x2 blocks.
    A (4x4) split into 2x2 blocks.
    B (4x5) split into:
      - B0 (4x2): Cols 0,1
      - B1 (4x2): Cols 2,3
      - B2 (4x1): Col 4
    
    For each 2x2x2 sub-product, use Strassen (rank 7).
    For each 2x2x1 sub-product, use direct (rank 4).
    
    Total Rank = 8*7 + 4*4 = 56 + 16 = 72.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    state = [0]
    
    rank_total = 72
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    def add_strassen_2x2(ra, ca, rb, cb):
        """Add 7 Strassen terms for 2x2 * 2x2.
        A block: rows [2*ra, 2*ra+1], cols [2*ca, 2*ca+1]
        B block: rows [2*rb, 2*rb+1], cols [2*cb*2, 2*cb*2+1] (cb is 0 or 1 for 2-col blocks)
        C block: rows [2*ra, 2*ra+1], cols [2*cb*2, 2*cb*2+1]
        """
        # Global indices
        a_r0, a_r1 = 2*ra, 2*ra+1
        a_c0, a_c1 = 2*ca, 2*ca+1
        b_r0, b_r1 = 2*rb, 2*rb+1
        b_c0, b_c1 = 2*cb, 2*cb+1
        c_r0, c_r1 = 2*ra, 2*ra+1
        c_c0, c_c1 = 2*cb, 2*cb+1
        
        col = state[0]
        # M1: (a+d)*(e+h) -> C[c_r0,c_c0] += 1, C[c_r1,c_c1] += 1
        U[a_r0 * m + a_c0, col] = 1.0
        U[a_r1 * m + a_c1, col] = 1.0
        V[b_r0 * p + b_c0, col] = 1.0
        V[b_r1 * p + b_c1, col] = 1.0
        W[c_c0 * n + c_r0, col] = 1.0
        W[c_c1 * n + c_r1, col] = 1.0
        state[0] += 1
        
        col = state[0]
        # M2: (c+d)*e -> C[c_r1,c_c0] += 1, C[c_r1,c_c1] -= 1
        U[a_r1 * m + a_c0, col] = 1.0
        U[a_r1 * m + a_c1, col] = 1.0
        V[b_r0 * p + b_c0, col] = 1.0
        W[c_c0 * n + c_r1, col] = 1.0
        W[c_c1 * n + c_r1, col] = -1.0
        state[0] += 1
        
        col = state[0]
        # M3: a*(f-h) -> C[c_r0,c_c1] += 1, C[c_r1,c_c1] += 1
        U[a_r0 * m + a_c0, col] = 1.0
        V[b_r0 * p + b_c1, col] = 1.0
        V[b_r1 * p + b_c1, col] = -1.0
        W[c_c1 * n + c_r0, col] = 1.0
        W[c_c1 * n + c_r1, col] = 1.0
        state[0] += 1
        
        col = state[0]
        # M4: d*(g-e) -> C[c_r0,c_c0] += 1, C[c_r1,c_c0] += 1
        U[a_r1 * m + a_c1, col] = 1.0
        V[b_r1 * p + b_c0, col] = 1.0
        V[b_r0 * p + b_c0, col] = -1.0
        W[c_c0 * n + c_r0, col] = 1.0
        W[c_c0 * n + c_r1, col] = 1.0
        state[0] += 1
        
        col = state[0]
        # M5: (a+b)*h -> C[c_r0,c_c0] -= 1, C[c_r0,c_c1] += 1
        U[a_r0 * m + a_c0, col] = 1.0
        U[a_r0 * m + a_c1, col] = 1.0
        V[b_r1 * p + b_c1, col] = 1.0
        W[c_c0 * n + c_r0, col] = -1.0
        W[c_c1 * n + c_r0, col] = 1.0
        state[0] += 1
        
        col = state[0]
        # M6: (c-a)*(e+f) -> C[c_r1,c_c1] += 1
        U[a_r1 * m + a_c0, col] = 1.0
        U[a_r0 * m + a_c0, col] = -1.0
        V[b_r0 * p + b_c0, col] = 1.0
        V[b_r0 * p + b_c1, col] = 1.0
        W[c_c1 * n + c_r1, col] = 1.0
        state[0] += 1
        
        col = state[0]
        # M7: (b-d)*(g+h) -> C[c_r0,c_c0] += 1
        U[a_r0 * m + a_c1, col] = 1.0
        U[a_r1 * m + a_c1, col] = -1.0
        V[b_r1 * p + b_c0, col] = 1.0
        V[b_r1 * p + b_c1, col] = 1.0
        W[c_c0 * n + c_r0, col] = 1.0
        state[0] += 1

    def add_direct_2x2_2x1(ra, ca, rb, cb_col):
        """Add 4 direct terms for 2x2 * 2x1.
        A block: rows [2*ra, 2*ra+1], cols [2*ca, 2*ca+1]
        B block: rows [2*rb, 2*rb+1], col [cb_col]
        C block: rows [2*ra, 2*ra+1], col [cb_col]
        """
        for i in range(2):
            for j in range(2):
                col = state[0]
                U[(2*ra + i) * m + (2*ca + j), col] = 1.0
                V[(2*rb + j) * p + cb_col, col] = 1.0
                W[cb_col * n + (2*ra + i), col] = 1.0
                state[0] += 1
    
    # Block 0: B cols 0,1 (cb=0)
    add_strassen_2x2(0, 0, 0, 0)  # A00 * B00 -> C00
    add_strassen_2x2(0, 1, 1, 0)  # A01 * B10 -> C00
    add_strassen_2x2(1, 0, 0, 0)  # A10 * B00 -> C10
    add_strassen_2x2(1, 1, 1, 0)  # A11 * B10 -> C10
    
    # Block 1: B cols 2,3 (cb=1)
    add_strassen_2x2(0, 0, 0, 1)  # A00 * B01 -> C01
    add_strassen_2x2(0, 1, 1, 1)  # A01 * B11 -> C01
    add_strassen_2x2(1, 0, 0, 1)  # A10 * B01 -> C10
    add_strassen_2x2(1, 1, 1, 1)  # A11 * B11 -> C11
    
    # Block 2: B col 4
    add_direct_2x2_2x1(0, 0, 0, 4)  # A00 * B02 -> C04
    add_direct_2x2_2x1(0, 1, 1, 4)  # A01 * B12 -> C04
    add_direct_2x2_2x1(1, 0, 0, 4)  # A10 * B02 -> C14
    add_direct_2x2_2x1(1, 1, 1, 4)  # A11 * B12 -> C14
    
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
