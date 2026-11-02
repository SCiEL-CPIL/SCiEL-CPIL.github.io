"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Rank-42 Strassen decomposition for 2x2 * 2x12 matrix multiplication.
    
    Partition 12 columns into 6 groups of 2. For each 2x2 * 2x2 block,
    use Strassen's rank-7 decomposition.
    Total rank = 6 * 7 = 42.
    
    Indexing conventions:
      U[i*m+j, r]: row = i*m+j (i in 0..n-1, j in 0..m-1) for A[i][j]
      V[j*p+k, r]: row = j*p+k (j in 0..m-1, k in 0..p-1) for B[j][k]
      W[k*n+i, r]: row = k*n+i (k in 0..p-1, i in 0..n-1) for C[i][k] (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_blocks = p // 2  # 6
    rank = num_blocks * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strassen 7-product decomposition for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[C00,C01],[C10,C11]]
    # Local indices:
    # A: 0=a, 1=b, 2=c, 3=d
    # B: 0=e, 1=f, 2=g, 3=h
    # C: 0=C00, 1=C01, 2=C10, 3=C11
    
    # The 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define coefficients for each term (M1..M7)
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (c-a)(e+f)
        ({0: -1, 2: 1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for t in range(num_blocks):
        if time.time() > deadline:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        for s in range(7):
            col = t * 7 + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
            
            # U: A indices are global (0..3), no offset
            for a_idx, val in u_coeffs.items():
                U[a_idx, col] = val
            
            # V: B indices are local to the block
            # Local B index 0 -> B[0, 2t] -> global index 0*p + 2t = 2t
            # Local B index 1 -> B[0, 2t+1] -> global index 0*p + 2t+1 = 2t+1
            # Local B index 2 -> B[1, 2t] -> global index 1*p + 2t = 12+2t
            # Local B index 3 -> B[1, 2t+1] -> global index 1*p + 2t+1 = 12+2t+1
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 2 * t
                elif local_b == 1:
                    global_b = 2 * t + 1
                elif local_b == 2:
                    global_b = m * p // 2 + 2 * t  # 12 + 2t
                else:  # local_b == 3
                    global_b = m * p // 2 + 2 * t + 1  # 12 + 2t + 1
                V[global_b, col] = val
            
            # W: C indices are local to the block
            # Local C index 0 -> C[0, 2t] -> W index k*n+i = (2t)*2 + 0 = 4t
            # Local C index 1 -> C[0, 2t+1] -> W index (2t+1)*2 + 0 = 4t+2
            # Local C index 2 -> C[1, 2t] -> W index (2t)*2 + 1 = 4t+1
            # Local C index 3 -> C[1, 2t+1] -> W index (2t+1)*2 + 1 = 4t+3
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0
                elif local_c == 2:
                    global_w = (2 * t) * n + 1
                else:  # local_c == 3
                    global_w = (2 * t + 1) * n + 1
                W[global_w, col] = val
    
    # Verify correctness before returning
    # Construct random test matrices and check
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute C from decomposition: C[i,k] = sum_r U[i*m+?, r] * V[?, r] * W[k*n+i, r]
    # Actually: C[i,k] = sum_r A_vec[i*m + j] * B_vec[j*p + k] contribution
    # The decomposition is: C = U^T * diag(1) * V^T rearranged...
    # More precisely: C[i,k] = sum_r (sum_j U[i*m+j, r]) * (sum_? ...)
    # No, the tensor decomposition means:
    # C[i,k] = sum_r A[i,0]*U[i*m+0,r] + A[i,1]*U[i*m+1,r]) * ... 
    # Actually for matrix multiplication tensor:
    # T[i,j,k,l] where T = sum_r U1[i,j,r]*U2[k,l,r]*U3[i,l,r]
    # Here: C[i,k] = sum_{j} A[i,j]*B[j,k]
    # = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_? ...)
    # The correct formula: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * ... 
    # No. For the tensor T[i,j,k] = delta (i.e., C[i,k] = sum_j A[i,j] B[j,k]):
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]... 
    # Wait, the factor matrices are:
    # U has shape (n*m, rank) -> U[i*m+j, r]
    # V has shape (m*p, rank) -> V[j*p+k, r]  
    # W has shape (p*n, rank) -> W[k*n+i, r]
    # The tensor identity is: delta[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where delta is such that C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's verify: C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j sum_r (U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]) * ... 
    # No, that's not right either. The factorization is:
    # The multiplication tensor M has M[i,j,k] = 1 if the index maps correctly
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T[i,j,k] where T[i,j,k] contributes A[i,j]*B[j,k] to C[i,k]
    # So T[i,j,k] = delta (it's just the identity in j linking A and B)
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Verify with random values
    C_from_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_from_decomp[i, k] = total
    
    # Check if it matches
    max_err = np.max(np.abs(C_from_decomp - C_expected))
    
    if max_err > 1e-10:
        # If verification fails, return zeros (should not happen)
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
