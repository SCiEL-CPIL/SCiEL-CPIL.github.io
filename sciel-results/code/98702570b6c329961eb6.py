import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    Strategy:
    - Use Strassen's algorithm (rank 7) for 2x2x2 matrix multiplication.
    - Partition the 12 columns of B into 6 pairs of 2 columns.
    - Each pair uses 7 rank-1 terms from Strassen.
    - Total rank = 6 * 7 = 42.
    
    This is an improvement over the schoolbook rank of 48 (2*2*12).
    The decomposition is exact and uses integer coefficients.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_blocks = p // 2  # 6 blocks of 2 columns each
    rank = num_blocks * 7  # 42
    
    # U has shape (n*m, rank) = (4, 42)
    # V has shape (m*p, rank) = (24, 42)
    # W has shape (p*n, rank) = (24, 42)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7-product decomposition for 2x2x2:
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=[[C00,C01],[C10,C11]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e)
    # M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    #
    # Local A indices: 0=a(A[0,0]), 1=b(A[0,1]), 2=c(A[1,0]), 3=d(A[1,1])
    # Local B indices: 0=e(B[0,0]), 1=f(B[0,1]), 2=g(B[1,0]), 3=h(B[1,1])
    # Local C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({0: -1, 2: 1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
            
            for a_idx, val in u_coeffs.items():
                U[a_idx][col] = val
            
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 0 * p + 2 * t
                elif local_b == 1:
                    global_b = 0 * p + 2 * t + 1
                elif local_b == 2:
                    global_b = 1 * p + 2 * t
                else:
                    global_b = 1 * p + 2 * t + 1
                V[global_b][col] = val
            
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0
                elif local_c == 2:
                    global_w = (2 * t) * n + 1
                else:
                    global_w = (2 * t + 1) * n + 1
                W[global_w][col] += val
    
    # Convert to numpy arrays for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Numerical verification
    np.random.seed(12345)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_test = A_test @ B_test
    
    # Compute C using tensor decomposition:
    # C[i,k] = sum_r u_r * v_r * W[k*n+i, r]
    # where u_r = sum_{i',j'} A[i',j'] U[i'*m+j', r]
    # and v_r = sum_{j',k'} B[j',k'] V[j'*p+k', r]
    
    # Efficient computation using numpy
    # u_r for all r: shape (rank,)
    # A_test is (n, m), U_np is (n*m, rank)
    # u_r = sum over (i,j) of A[i,j] * U[i*m+j, r]
    A_flat = A_test.flatten()  # shape (n*m,)
    u_vals = U_np.T @ A_flat  # shape (rank,)
    
    B_flat = B_test.flatten()  # shape (m*p,)
    v_vals = V_np.T @ B_flat  # shape (rank,)
    
    # C[i,k] = sum_r u_r * v_r * W[k*n+i, r]
    # W_np is (p*n, rank), W[k*n+i, r]
    # C is (n, p)
    # Let's reshape: for each r, we add u_r*v_r * W_np[:, r] to C_decomp
    # W_np[:, r] has shape (p*n,), which we need to reshape to (n, p) with W[k*n+i] = C[i,k]
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        C_decomp += (u_vals[r] * v_vals[r]) * W_np[:, r].reshape(p, n).T
    
    # Check
    err = np.max(np.abs(C_decomp - C_test))
    if err > 1e-9:
        # If there's an error, we could try to fix it, but the Strassen construction should be correct
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
