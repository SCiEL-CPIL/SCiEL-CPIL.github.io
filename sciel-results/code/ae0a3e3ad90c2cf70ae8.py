"""Program for matrix multiplication 2x2 by 2x8 tensor decomposition."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2_block(base_col, n, m, p, U, V, W):
    """
    Add 7 Strassen terms for a 2x2 * 2x2 block starting at column base_col.
    U rows: i*m+j (i in 0..1, j in 0..1) -> rows 0..3
    V rows: j*p+k (j in 0..1, k in 0..1) -> rows 0..3
    W rows: k*n+i (k in 0..1, i in 0..1) -> rows 0..3
    
    Strassen for A(2x2) * B(2x2) = C(2x2):
      M1 = (A00+A11)*(B00+B11)
      M2 = (A10+A11)*B00
      M3 = A00*(B01-B11)
      M4 = A11*(B10-B00)
      M5 = (A00+A01)*B11
      M6 = (A10-A00)*(B00+B01)
      M7 = (A01-A11)*(B10+B11)
      C00 = M1+M4-M5+M7
      C01 = M3+M5
      C10 = M2+M4
      C11 = M1-M2+M3+M6
    """
    # For each Strassen term t (0..6), we add a rank-1 tensor.
    # U contributes to A factors, V to B factors, W to C reconstruction.
    
    # A indices: A[i,j] -> U row i*m+j = i*2+j
    # B indices: B[j,k] -> V row j*p+k = j*2+k
    # C indices: C[i,k] -> W row k*n+i = k*2+i
    
    # M1 = (A00+A11)*(B00+B11)
    # A00: U[0,0], A11: U[1,1]
    # B00: V[0,0], B11: V[1,1]
    # C00: W[0,0], C01: W[1,0], C10: W[0,1], C11: W[1,1]
    # C00 += M1, C01 += 0, C10 += 0, C11 += M1
    # Wait, let me recheck: C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    
    terms = []
    # Each term: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict from (i,j) to coeff
    # V_coeffs: dict from (j,k) to coeff
    # W_coeffs: dict from (i,k) to coeff
    
    # M1 = (A00+A11)*(B00+B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00 += 1, C11 += 1 (from C00=M1+... and C11=M1-...)
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    
    # M2 = (A10+A11)*B00
    # U: A10=1, A11=1
    # V: B00=1
    # W: C10 += 1, C11 -= 1
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}))
    
    # M3 = A00*(B01-B11)
    # U: A00=1
    # V: B01=1, B11=-1
    # W: C01 += 1, C11 += 1
    terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}))
    
    # M4 = A11*(B10-B00)
    # U: A11=1
    # V: B10=1, B00=-1
    # W: C00 += 1, C10 += 1
    terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}))
    
    # M5 = (A00+A01)*B11
    # U: A00=1, A01=1
    # V: B11=1
    # W: C00 -= 1, C01 += 1
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}))
    
    # M6 = (A10-A00)*(B00+B01)
    # U: A10=1, A00=-1
    # V: B00=1, B01=1
    # W: C11 += 1
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
    
    # M7 = (A01-A11)*(B10+B11)
    # U: A01=1, A11=-1
    # V: B10=1, B11=1
    # W: C00 += 1
    terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
    
    for u_coeffs, v_coeffs, w_coeffs in terms:
        for (i, j), u_val in u_coeffs.items():
            U[i * m + j][base_col] += u_val
        for (j, k), v_val in v_coeffs.items():
            V[j * p + k][base_col] += v_val
        for (i, k), w_val in w_coeffs.items():
            W[k * n + i][base_col] += w_val
        base_col += 1
    
    return base_col

def _verify(U, V, W, n, m, p, num_trials=20):
    """Verify the decomposition exactly using integer arithmetic."""
    rank = len(U[0])
    U_np = np.array(U, dtype=object)
    V_np = np.array(V, dtype=object)
    W_np = np.array(W, dtype=object)
    
    for _ in range(num_trials):
        A = np.random.randint(-5, 6, size=(n, m))
        B = np.random.randint(-5, 6, size=(m, p))
        C_expected = A @ B
        
        # Compute C via tensor decomposition: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # But that's not how it works. The tensor decomposition means:
        # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        # Wait no. Let me think again.
        
        # Actually the standard tensor decomposition for matmul:
        # C[i,k] = sum_c U_{i,j,c} * A[i,j] * V_{j,k,c} ... no.
        
        # The correct interpretation:
        # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        # This is wrong too. Let me think more carefully.
        
        # For matrix multiplication C = A*B where A is n x m and B is m x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # A rank-r tensor decomposition expresses this as:
        # C[i,k] = sum_{c=0}^{r-1} (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        
        # Hmm, that doesn't look right dimensionally either. Let me reconsider.
        
        # Actually, the standard formulation is:
        # C[i,k] = sum_c alpha_c * (sum_j U_{c,i,j} * A[i,j]) * (sum_j V_{c,j,k} * B[j,k])
        
        # But in our convention, U has shape (n*m, rank), so U[i*m+j, c].
        # V has shape (m*p, rank), so V[j*p+k, c].
        # W has shape (p*n, rank), so W[k*n+i, c].
        
        # The correct formula should be:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        
        # Wait, that gives C[i,k] = sum_c W[k*n+i,c] * (U^T A)_c * (B V^T)_c
        # where (U^T A)_c = sum_{i,j} U[i*m+j, c] * A[i,j]... but that sums over i too, which is wrong.
        
        # Let me reconsider. I think the correct interpretation is:
        # For each c, we form:
        #   S_A_c[i] = sum_j U[i*m+j, c] * A[i,j]  (for each i)
        #   S_B_c[k] = sum_j V[j*p+k, c] * B[j,k]  (for each k)
        #   C[i,k] += W[k*n+i, c] * S_A_c[i] * S_B_c[k]
        
        # Let me verify with schoolbook: U[i*m+j, c]=1 when c corresponds to (i,j,k)
        # V[j*p+k, c]=1 when same
        # W[k*n+i, c]=1 when same
        # Then S_A_c[i] = sum_j U[i*m+j,c]*A[i,j] = A[i,j_c] (only one j nonzero)
        # S_B_c[k] = sum_j V[j*p+k,c]*B[j,k] = B[j_c, k] (only one j nonzero, and j_c = j)
        # C[i,k] += 1 * A[i,j] * B[j,k] which is correct!
        
        # So the formula is:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        
        C_computed = np.zeros((n, p), dtype=object)
        for c in range(rank):
            # S_A_c[i] = sum_j U[i*m+j, c] * A[i,j]
            S_A = np.zeros(n, dtype=object)
            for i in range(n):
                for j in range(m):
                    S_A[i] += U_np[i * m + j, c] * A[i, j]
            
            # S_B_c[k] = sum_j V[j*p+k, c] * B[j,k]
            S_B = np.zeros(p, dtype=object)
            for k in range(p):
                for j in range(m):
                    S_B[k] += V_np[j * p + k, c] * B[j, k]
            
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W_np[k * n + i, c] * S_A[i] * S_B[k]
        
        if not np.array_equal(C_computed, C_expected):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Group 8 columns into 4 blocks of 2
    # Each block is a 2x2 * 2x2 multiplication, using Strassen (rank 7)
    # Total rank = 4 * 7 = 28
    
    rank = 4 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    base_col = 0
    for block in range(4):
        # For block b, columns 2*b and 2*b+1 of B
        # V row index: j*p + k where k is the global column
        # We need to offset V indices for each block
        # But our _strassen_2x2_block assumes local indices...
        # Let me handle the offset manually.
        
        # For block b, the B columns are 2*b and 2*b+1
        # V[j*p + k, c] where k = 2*b + local_k
        # So V row = j*8 + 2*b + local_k
        
        # Let me just build it directly for each block.
        terms = []
        # Same Strassen terms as before
        terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
        terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}))
        terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}))
        terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}))
        terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}))
        terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
        terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
        
        for u_coeffs, v_coeffs, w_coeffs in terms:
            for (i, j), u_val in u_coeffs.items():
                U[i * m + j][base_col] += u_val
            for (j, lk), v_val in v_coeffs.items():
                # lk is local k (0 or 1), global k = 2*block + lk
                V[j * p + 2 * block + lk][base_col] += v_val
            for (i, lk), w_val in w_coeffs.items():
                # lk is local k, global k = 2*block + lk
                W[(2 * block + lk) * n + i][base_col] += w_val
            base_col += 1
    
    # Verify
    if not _verify(U, V, W, n, m, p, num_trials=5):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
