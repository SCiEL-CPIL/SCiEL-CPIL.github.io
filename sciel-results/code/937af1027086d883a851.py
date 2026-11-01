import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strassen-based decomposition: partition p=12 columns into 6 pairs
    # Each pair uses 7 rank-1 terms (Strassen for 2x2x2 matrix mult)
    # Total rank = 6 * 7 = 42 (improvement over schoolbook rank 48)
    
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 42)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # shape (24, 42)
    
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
        # (u: dict of A_idx->coeff, v: dict of local_B_idx->coeff, w: dict of local_C_idx->coeff)
        {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1},       # M1
        {2: 1, 3: 1}, {0: 1},        {2: 1, 3: -1},      # M2
        {0: 1},       {1: 1, 3: -1}, {1: 1, 3: 1},       # M3
        {3: 1},       {2: 1, 0: -1}, {0: 1, 2: 1},       # M4
        {0: 1, 1: 1}, {3: 1},        {0: -1, 1: 1},      # M5
        {0: -1, 2: 1},{0: 1, 1: 1},  {3: 1},             # M6
        {1: 1, 3: -1},{2: 1, 3: 1},  {0: 1},             # M7
    ]
    
    for t in range(num_blocks):
        if time.time() - start_time > deadline:
            break
        for s in range(7):
            col = t * 7 + s
            u_coeffs = strassen_terms[s * 3]
            v_coeffs = strassen_terms[s * 3 + 1]
            w_coeffs = strassen_terms[s * 3 + 2]
            
            # Set U (A indices are global, no offset needed)
            for a_idx, val in u_coeffs.items():
                U[a_idx][col] = val
            
            # Set V (map local B index to global)
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 2 * t        # B[0, 2t]
                elif local_b == 1:
                    global_b = 2 * t + 1    # B[0, 2t+1]
                elif local_b == 2:
                    global_b = m * p // 2 + 2 * t  # B[1, 2t] = 12+2t
                else:  # local_b == 3
                    global_b = m * p // 2 + 2 * t + 1  # B[1, 2t+1]
                V[global_b][col] = val
            
            # Set W (map local C index to global W index: W[k*n+i] where k=col, i=row)
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0       # C[0, 2t]
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0   # C[0, 2t+1]
                elif local_c == 2:
                    global_w = (2 * t) * n + 1       # C[1, 2t]
                else:  # local_c == 3
                    global_w = (2 * t + 1) * n + 1   # C[1, 2t+1]
                W[global_w][col] = val
    
    # Verify correctness with a random test
    if time.time() - start_time < deadline - 5:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            # Reshape to tensor form
            # U: (n*m, rank) -> (n, m, rank)
            U_tensor = U_np.reshape(n, m, rank)
            # V: (m*p, rank) -> (m, p, rank)
            V_tensor = V_np.reshape(m, p, rank)
            # W: (p*n, rank) -> (p, n, rank)
            W_tensor = W_np.reshape(p, n, rank)
            
            # Compute tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # Wait, the decomposition is: C[i,k] = sum_j A[i,j] * B[j,k]
            # Tensor form: T[i,j,k] = A[i,j] * B[j,k]
            # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]?
            # Actually, the standard form is:
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # But here U has shape (n*m, rank), so U[i*m+j, r]
            # V has shape (m*p, rank), so V[j*p+k, r]
            # W has shape (p*n, rank), so W[k*n+i, r]
            
            # Let me verify: for each (i,j,k), T[i,j,k] should equal A[i,j]*B[j,k]
            # And sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal that
            
            # Test with specific values
            A_test = np.array([[1, 2], [3, 4]], dtype=float)
            B_test = np.zeros((m, p), dtype=float)
            B_test[0, :] = range(p)
            B_test[1, :] = range(p, 2*p)
            
            C_expected = A_test @ B_test  # (n, p)
            
            # Compute via decomposition
            C_computed = np.zeros((n, p), dtype=float)
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        C_computed[i, k] += U[i * m + (0), r] * 0  # This isn't right
            
            # Actually, let me just check the tensor identity directly
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This should equal delta... no, it should be part of the matrix mult tensor
            
            # The matrix multiplication tensor T has T[i,j,k] = 1 if we think of it as 
            # the structure tensor, but actually for the bilinear map:
            # C = A*B, the tensor is T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
            # So T[i,j,k] = 1 for all valid (i,j,k)
            
            # Wait no. The tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
            # is T in R^{n x m x p} where T[i,j,k] = 1 for all i,j,k.
            # The decomposition says: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # for all i, j, k.
            
            # Let's verify this
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err > 1e-10:
                # Fallback: use schoolbook decomposition
                rank_school = n * m * p  # 48
                U_s = [[0] * rank_school for _ in range(n * m)]
                V_s = [[0] * rank_school for _ in range(m * p)]
                W_s = [[0] * rank_school for _ in range(p * n)]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i * m * p + j * p + k
                            U_s[i * m + j][idx] = 1
                            V_s[j * p + k][idx] = 1
                            W_s[k * n + i][idx] = 1
                return {"U": U_s, "V": V_s, "W": W_s}
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
