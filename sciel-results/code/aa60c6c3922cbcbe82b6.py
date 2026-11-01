import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Safety margin before 120s limit

def _time_remaining():
    return _DEADLINE - time.time()

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1
    - Use Strassen's rank-7 algorithm for columns 2,3  
    - Use Strassen's rank-7 algorithm for columns 4,5
    - Use schoolbook (rank 4) for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    
    Indexing conventions:
    - U[i*m + j, r] where A[i,j] -> index i*2+j (0:A00, 1:A01, 2:A10, 3:A11)
    - V[j*p + k, r] where B[j,k] -> index j*7+k (0:B00, ..., 6:B06, 7:B10, ..., 13:B16)
    - W[k*n + i, r] where C[i,k] -> index k*2+i (0:C00, 1:C10, 2:C01, 3:C11, ...)
    """
    if _time_remaining() < 5.0:
        return _schoolbook_decomposition()
    
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(col_start, k_start):
        """
        Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1.
        
        A = [[A00, A01], [A10, A11]]
        B_block = [[B0,ks, B0,ks+1], [B1,ks, B1,ks+1]]
        C_block = [[C0,ks, C0,ks+1], [C1,ks, C1,ks+1]]
        
        Strassen:
        M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        M2 = (A10 + A11) * B0,ks
        M3 = A00 * (B1,ks+1 - B0,ks+1)
        M4 = A11 * (B1,ks - B0,ks)
        M5 = (A00 + A01) * B1,ks+1
        M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        
        C0,ks   = M1 + M4 - M5 + M7
        C0,ks+1 = M3 + M5
        C1,ks   = M2 + M6
        C1,ks+1 = M1 - M2 + M3 + M4
        """
        ks = k_start
        ks1 = k_start + 1
        
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        # Contributes: +1 to C0,ks and C1,ks+1
        U[0, col_start+0] += 1   # A00
        U[3, col_start+0] += 1   # A11
        V[7*0 + ks, col_start+0] += 1      # B0,ks
        V[7*1 + ks1, col_start+0] += 1     # B1,ks+1
        W[2*ks + 0, col_start+0] += 1      # C0,ks
        W[2*ks1 + 1, col_start+0] += 1     # C1,ks+1
        
        # M2 = (A10 + A11) * B0,ks
        # Contributes: +1 to C1,ks, -1 to C1,ks+1
        U[2, col_start+1] += 1   # A10
        U[3, col_start+1] += 1   # A11
        V[7*0 + ks, col_start+1] += 1      # B0,ks
        W[2*ks + 1, col_start+1] += 1      # C1,ks
        W[2*ks1 + 1, col_start+1] -= 1     # C1,ks+1
        
        # M3 = A00 * (B1,ks+1 - B0,ks+1)
        # Contributes: +1 to C0,ks+1 and C1,ks+1
        U[0, col_start+2] += 1   # A00
        V[7*1 + ks1, col_start+2] += 1     # B1,ks+1
        V[7*0 + ks1, col_start+2] -= 1     # B0,ks+1
        W[2*ks1 + 0, col_start+2] += 1     # C0,ks+1
        W[2*ks1 + 1, col_start+2] += 1     # C1,ks+1
        
        # M4 = A11 * (B1,ks - B0,ks)
        # Contributes: +1 to C0,ks and C1,ks+1
        U[3, col_start+3] += 1   # A11
        V[7*1 + ks, col_start+3] += 1      # B1,ks
        V[7*0 + ks, col_start+3] -= 1      # B0,ks
        W[2*ks + 0, col_start+3] += 1      # C0,ks
        W[2*ks1 + 1, col_start+3] += 1     # C1,ks+1
        
        # M5 = (A00 + A01) * B1,ks+1
        # Contributes: -1 to C0,ks, +1 to C0,ks+1
        U[0, col_start+4] += 1   # A00
        U[1, col_start+4] += 1   # A01
        V[7*1 + ks1, col_start+4] += 1     # B1,ks+1
        W[2*ks + 0, col_start+4] -= 1      # C0,ks
        W[2*ks1 + 0, col_start+4] += 1     # C0,ks+1
        
        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        # Contributes: +1 to C1,ks
        U[2, col_start+5] += 1   # A10
        U[0, col_start+5] -= 1   # A00
        V[7*0 + ks, col_start+5] += 1      # B0,ks
        V[7*0 + ks1, col_start+5] += 1     # B0,ks+1
        W[2*ks + 1, col_start+5] += 1      # C1,ks
        
        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        # Contributes: +1 to C0,ks
        U[1, col_start+6] += 1   # A01
        U[3, col_start+6] -= 1   # A11
        V[7*1 + ks, col_start+6] += 1      # B1,ks
        V[7*1 + ks1, col_start+6] += 1     # B1,ks+1
        W[2*ks + 0, col_start+6] += 1      # C0,ks
        
    # Add Strassen for columns 0,1
    add_strassen(0, 0)
    
    # Add Strassen for columns 2,3
    add_strassen(7, 2)
    
    # Add Strassen for columns 4,5
    add_strassen(14, 4)
    
    # Add schoolbook for column 6 (4 terms)
    col_start = 21
    # A00 * B0,6 -> C0,6
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    
    # A01 * B1,6 -> C0,6
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    
    # A10 * B0,6 -> C1,6
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    
    # A11 * B1,6 -> C1,6
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = n * m * p  # 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Test with multiple random matrices
    np.random.seed(42)
    all_pass = True
    for trial in range(100):
        A = np.random.rand(2, 2)
        B = np.random.rand(2, 7)
        C_expected = A @ B
        
        # Reconstruct C from tensor decomposition
        # C[i,k] = sum_r (sum_{i',j} U[i',j,r] * A[i',j]) * (sum_{j',k'} V[j',k',r] * B[j',k']) * W[k*n+i, r]
        # Wait, that's not right. The tensor decomposition is:
        # T = sum_r U_r ⊗ V_r ⊗ W_r
        # where T[i,j,k] is the coefficient such that C_k = sum_{i,j} T[i,j,k] A_i B_j
        # In our flattened representation:
        # C_flat[k] = sum_r (sum_i U[i,r] * A_flat[i]) * (sum_j V[j,r] * B_flat[j]) * W[k,r]
        # No, that's a rank-1 tensor in a different sense.
        
        # Actually, the standard matrix multiplication tensor decomposition is:
        # For each r, we have a bilinear form: M_r(A,B) = (U_r^T A_vec) * (V_r^T B_vec)
        # and C_vec = sum_r M_r(A,B) * W_r
        
        A_flat = A.flatten()  # [A00, A01, A10, A11]
        B_flat = B.flatten()  # [B00, B01, ..., B06, B10, B11, ..., B16]
        
        C_flat = np.zeros(14)
        for r in range(U.shape[1]):
            u_val = np.dot(U[:, r], A_flat)
            v_val = np.dot(V[:, r], B_flat)
            C_flat += u_val * v_val * W[:, r]
        
        # C_flat is indexed by k*n + i = 2*k + i
        C_reconstructed = np.zeros((2, 7))
        for k in range(7):
            for i in range(2):
                C_reconstructed[i, k] = C_flat[2*k + i]
        
        if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
            print(f"Trial {trial} FAILED")
            print(f"A = {A}")
            print(f"B = {B}")
            print(f"C_expected = {C_expected}")
            print(f"C_reconstructed = {C_reconstructed}")
            all_pass = False
            break
    
    if all_pass:
        print("All 100 trials PASSED")
    else:
        print("Verification FAILED")
