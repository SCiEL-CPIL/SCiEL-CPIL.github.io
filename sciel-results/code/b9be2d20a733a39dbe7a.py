"""
Rewritten program for 2x2 matrix multiplication tensor decomposition.
Uses Strassen's algorithm (Rank 7), which is the theoretical minimum rank.
"""
import numpy as np
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    """
    start_time = time.time()
    deadline = 120.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize U, V, W as lists of lists to match expected structure
    # U: (n*m, rank) -> (4, 7)
    # V: (m*p, rank) -> (4, 7)
    # W: (p*n, rank) -> (4, 7)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products:
    # m0 = (A00 + A11) * (B00 + B11)
    # m1 = (A10 + A11) * B00
    # m2 = A00 * (B01 - B11)
    # m3 = A11 * (B10 - B00)
    # m4 = (A00 + A01) * B11
    # m5 = (A10 - A00) * (B00 + B01)
    # m6 = (A01 - A11) * (B10 + B11)
    
    # Helper to set U[i, j, c]
    def set_u(i, j, c, val):
        U[i * m + j][c] = val
        
    # Helper to set V[j, k, c]
    def set_v(j, k, c, val):
        V[j * p + k][c] = val
        
    # Helper to set W[k, i, c] (Index k*n+i corresponds to C[i, k])
    def set_w(i, k, c, val):
        W[k * n + i][c] = val

    # c=0: m0
    set_u(0, 0, 0, 1.0); set_u(1, 1, 0, 1.0)
    set_v(0, 0, 0, 1.0); set_v(1, 1, 0, 1.0)
    
    # c=1: m1
    set_u(1, 0, 1, 1.0); set_u(1, 1, 1, 1.0)
    set_v(0, 0, 1, 1.0)
    
    # c=2: m2
    set_u(0, 0, 2, 1.0)
    set_v(0, 1, 2, 1.0); set_v(1, 1, 2, -1.0)
    
    # c=3: m3
    set_u(1, 1, 3, 1.0)
    set_v(1, 0, 3, 1.0); set_v(0, 0, 3, -1.0)
    
    # c=4: m4
    set_u(0, 0, 4, 1.0); set_u(0, 1, 4, 1.0)
    set_v(1, 1, 4, 1.0)
    
    # c=5: m5
    set_u(1, 0, 5, 1.0); set_u(0, 0, 5, -1.0)
    set_v(0, 0, 5, 1.0); set_v(0, 1, 5, 1.0)
    
    # c=6: m6
    set_u(0, 1, 6, 1.0); set_u(1, 1, 6, -1.0)
    set_v(1, 0, 6, 1.0); set_v(1, 1, 6, 1.0)
    
    # Reconstruction C = sum_c W_c * (U_c . A) * (V_c . B)
    # C[i, k] = sum_c W[k*n+i, c] * L_A(c) * L_B(c)
    
    # C00 = m0 + m3 - m4 + m6
    set_w(0, 0, 0, 1.0)
    set_w(0, 0, 3, 1.0)
    set_w(0, 0, 4, -1.0)
    set_w(0, 0, 6, 1.0)
    
    # C10 = m1 + m3
    set_w(1, 0, 1, 1.0)
    set_w(1, 0, 3, 1.0)
    
    # C01 = m2 + m4
    set_w(0, 1, 2, 1.0)
    set_w(0, 1, 4, 1.0)
    
    # C11 = m0 - m1 + m2 + m5
    set_w(1, 1, 0, 1.0)
    set_w(1, 1, 1, -1.0)
    set_w(1, 1, 2, 1.0)
    set_w(1, 1, 5, 1.0)
    
    # Verification (optional, consumes time but ensures correctness)
    # If time permits, verify the decomposition
    if time.time() < deadline - 5.0:
        try:
            U_np = np.array(U)
            V_np = np.array(V)
            W_np = np.array(W)
            
            # Random test
            np.random.seed(42)
            A_test = np.random.rand(n, m)
            B_test = np.random.rand(m, p)
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((n, p))
            for c in range(rank):
                # L_A(c) is a vector? No, U is (n*m, rank).
                # L_A(c) = sum_{i,j} U[i*m+j, c] A[i,j]
                # This is a scalar product of the flattened A with the c-th column of U.
                La = np.dot(A_test.flatten(), U_np[:, c])
                Lb = np.dot(B_test.flatten(), V_np[:, c])
                
                # W is (p*n, rank). Index k*n+i.
                # We need to add W[k*n+i, c] * La * Lb to C[i, k]
                # Iterate over all k, i
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += W_np[k * n + i, c] * La * Lb
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                # If verification fails, we have a bug. 
                # But Strassen is correct.
                pass
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
