"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    This implementation uses a known rank-26 decomposition.
    It combines a 7-term Strassen decomposition for the 2x2 sub-block 
    with 19 explicit rank-1 terms for the remaining border elements.
    
    The decomposition T = sum_r U[:, :, r] * V[:, :, r] * W[:, :, r] 
    must hold for the multiplication tensor.
    """
    start_time = time.time()
    deadline = 145.0 # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    R = 26
    
    # Use int8 for memory efficiency, sufficient for values -1, 0, 1
    U = np.zeros((n * m, R), dtype=np.int8)
    V = np.zeros((m * p, R), dtype=np.int8)
    W = np.zeros((p * n, R), dtype=np.int8)
    
    # Helper to get flattened indices
    # U is (n*m, R), index is i*m + j
    # V is (m*p, R), index is j*p + k
    # W is (p*n, R), index is k*n + i
    
    def idx_u(i, j):
        return i * m + j
        
    def idx_v(j, k):
        return j * p + k
        
    def idx_w(k, i):
        return k * n + i
        
    # --- Strassen Part (7 terms) ---
    # These handle the 2x2 multiplication of the top-left block (indices 0,1).
    # The 2x2 block corresponds to i,j,k in {0, 1}.
    
    # m1 = (a00 + a11)(b00 + b11)
    r = 0
    U[idx_u(0, 0), r] = 1
    U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1  # c00 += m1
    W[idx_w(1, 1), r] = 1  # c11 += m1
    
    # m2 = (a10 + a11)b00
    r = 1
    U[idx_u(1, 0), r] = 1
    U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    W[idx_w(0, 1), r] = 1  # c10 += m2
    W[idx_w(1, 1), r] = -1 # c11 -= m2
    
    # m3 = a00(b01 - b11)
    r = 2
    U[idx_u(0, 0), r] = 1
    V[idx_v(0, 1), r] = 1
    V[idx_v(1, 1), r] = -1
    W[idx_w(1, 0), r] = 1  # c01 += m3
    W[idx_w(1, 1), r] = 1  # c11 += m3
    
    # m4 = a11(b10 - b00)
    r = 3
    U[idx_u(1, 1), r] = 1
    V[idx_v(1, 0), r] = 1
    V[idx_v(0, 0), r] = -1
    W[idx_w(0, 0), r] = 1  # c00 += m4
    W[idx_w(0, 1), r] = 1  # c10 += m4
    
    # m5 = (a00 + a01)b11
    r = 4
    U[idx_u(0, 0), r] = 1
    U[idx_u(0, 1), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = -1 # c00 -= m5
    W[idx_w(1, 0), r] = 1  # c01 += m5
    
    # m6 = (a10 - a00)(b00 + b01)
    r = 5
    U[idx_u(1, 0), r] = 1
    U[idx_u(0, 0), r] = -1
    V[idx_v(0, 0), r] = 1
    V[idx_v(0, 1), r] = 1
    W[idx_w(1, 1), r] = 1  # c11 += m6
    
    # m7 = (a01 - a11)(b10 + b11)
    r = 6
    U[idx_u(0, 1), r] = 1
    U[idx_u(1, 1), r] = -1
    V[idx_v(1, 0), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1  # c00 += m7
    
    # --- Standard Part (19 terms) ---
    # These handle all (i, j, k) where at least one index is 2.
    # The Strassen part covers all (i, j, k) with i, j, k in {0, 1}.
    # The remaining terms are rank-1 tensors e_{i,j} \otimes e_{j,k} \otimes e_{k,i}.
    
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if all indices are in {0, 1} (handled by Strassen)
                if i < 2 and j < 2 and k < 2:
                    continue
                
                # Check time budget, though this loop is tiny (19 iterations)
                if time.time() > deadline:
                    # If we hit the deadline, we return what we have. 
                    # Since we are in the middle of filling 19 terms, 
                    # and the first 7 are done, returning an incomplete tensor 
                    # would be invalid. 
                    # However, 19 iterations take microseconds. 
                    # This guard is for safety per directive.
                    pass 
                
                # Add a rank-1 term
                U[idx_u(i, j), col] = 1
                V[idx_v(j, k), col] = 1
                W[idx_w(k, i), col] = 1
                col += 1
                
    # Verify rank
    assert col == R, f"Rank mismatch: {col} vs {R}"

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
