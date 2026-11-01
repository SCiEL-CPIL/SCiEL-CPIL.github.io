import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    
    # Strassen 7 terms for 2x2x2
    # U vectors (coeffs for A00, A01, A10, A11)
    # V vectors (coeffs for B00, B01, B10, B11)
    # W vectors (coeffs for C00, C10, C01, C11) -> Note: W index is (k,i)
    # In local 2x2 block:
    # C00 is i=0, k=0. Index 0.
    # C10 is i=1, k=0. Index 1.
    # C01 is i=0, k=1. Index 2.
    # C11 is i=1, k=1. Index 3.
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
    # M3 = A00(B10-B11)       -> C01 += M3, C10 += M3
    # M4 = (A00+A01)B11       -> C00 -= M4, C10 -= M4
    # M5 = (A01+A11)(B10+B11) -> C10 += M5, C11 += M5
    # M6 = (A00-A10)(B01+B11) -> C00 += M6, C01 -= M6
    # M7 = (A01-A11)(B00+B10) -> C00 += M7, C01 += M7
    
    # Let's verify C00: M1 - M4 + M6 + M7
    # C00 = (A00+A11)(B00+B11) - (A00+A01)B11 + (A00-A10)(B01+B11) + (A01-A11)(B00+B10)
    # = A00B00 + A00B11 + A11B00 + A11B11 
    #   - A00B11 - A01B11 
    #   + A00B01 + A00B11 - A10B01 - A10B11 
    #   + A01B00 + A01B10 - A11B00 - A11B10
    # Terms:
    # A00B00: 1
    # A00B11: 1 - 1 + 1 = 1? No.
    # Let's re-calculate carefully.
    # M1: A00B00, A00B11, A11B00, A11B11
    # M4: A00B11, A01B11
    # M6: A00B01, A00B11, -A10B01, -A10B11
    # M7: A01B00, A01B10, -A11B00, -A11B10
    
    # Sum for C00:
    # A00B00: 1 (from M1)
    # A00B01: 1 (from M6)
    # A00B10: 0
    # A00B11: 1 (M1) - 1 (M4) + 1 (M6) = 1. Correct.
    # A01B00: 1 (from M7)
    # A01B01: 0
    # A01B10: 1 (from M7)
    # A01B11: -1 (from M4)
    # A10B00: 0
    # A10B01: -1 (from M6)
    # A10B10: 0
    # A10B11: -1 (from M6)
    # A11B00: 1 (M1) - 1 (M7) = 0. Correct.
    # A11B01: 0
    # A11B10: -1 (from M7)
    # A11B11: 1 (from M1)
    
    # Wait, C00 should be A00B00 + A01B10.
    # My sum has A00B11, A01B00, A01B10, A01B11, A10B01, A10B11, A11B11.
    # This is wrong. The standard Strassen formulas I listed might be slightly off or my expansion is.
    
    # Let's use the verified terms from the current program which scored 1.1429 (valid).
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    R = 35
    U = np.zeros((4, R), dtype=float)
    V = np.zeros((20, R), dtype=float)
    W = np.zeros((20, R), dtype=float)
    
    col = 0
    for q in range(5):
        for a_vec, b_vec, c_vec in terms:
            # U
            U[0, col] = a_vec[0]
            U[1, col] = a_vec[1]
            U[2, col] = a_vec[2]
            U[3, col] = a_vec[3]
            
            # V
            V[0*10 + 2*q, col] = b_vec[0]
            V[0*10 + 2*q+1, col] = b_vec[1]
            V[1*10 + 2*q, col] = b_vec[2]
            V[1*10 + 2*q+1, col] = b_vec[3]
            
            # W
            W[4*q + 0, col] = c_vec[0]
            W[4*q + 1, col] = c_vec[1]
            W[4*q + 2, col] = c_vec[2]
            W[4*q + 3, col] = c_vec[3]
            
            col += 1
            
    # Verification
    # Check if sum_r W[k,i,r] U[i',j',r] V[j'',k'',r] == delta(i,i') delta(k,k'') delta(j',j'')
    # We can check this by computing C = A B and comparing with decomposition.
    # A is 2x2, B is 2x10.
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Decomposition:
    # C_decomp[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_j' V[j',k,r] B[j',k]) W[k,i,r]
    # Note: The product of the two sums is a scalar for each r.
    # Let P_r = (sum_j U[i,j,r] A[i,j]) * (sum_j' V[j',k,r] B[j',k])
    # Then C_decomp[i,k] = sum_r P_r W[k,i,r]
    
    # Let's test with a random A and B.
    np.random.seed(42)
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 10)
    C_true = A @ B
    
    C_decomp = np.zeros((2, 10))
    for i in range(2):
        for k in range(10):
            s = 0.0
            for r in range(R):
                # sum_j U[i,j,r] A[i,j]
                u_val = 0.0
                for j in range(2):
                    u_val += U[i*2 + j, r] * A[i, j]
                    
                # sum_j' V[j',k,r] B[j',k]
                v_val = 0.0
                for jp in range(2):
                    v_val += V[jp*10 + k, r] * B[jp, k]
                    
                s += u_val * v_val * W[k*2 + i, r]
            C_decomp[i, k] = s
            
    if not np.allclose(C_true, C_decomp):
        # Fallback to schoolbook if verification fails (should not happen)
        R = 40
        U = np.zeros((4, R), dtype=float)
        V = np.zeros((20, R), dtype=float)
        W = np.zeros((20, R), dtype=float)
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(10):
                    U[i*2 + j, col] = 1.0
                    V[j*10 + k, col] = 1.0
                    W[k*2 + i, col] = 1.0
                    col += 1
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
