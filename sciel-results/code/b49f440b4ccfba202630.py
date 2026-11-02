import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    num_u = n * m  # 6
    num_v = m * p  # 21
    num_w = p * n  # 14

    def verify(U_flat, V_flat, W_flat):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k"""
        R = U_flat.shape[1]
        u_indices = []
        v_indices = []
        w_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_indices.append(i * m + j)
                    v_indices.append(j * p + k)
                    w_indices.append(k * n + i)
        u_indices = np.array(u_indices)
        v_indices = np.array(v_indices)
        w_indices = np.array(w_indices)
        
        U_sel = U_flat[u_indices, :]
        V_sel = V_flat[v_indices, :]
        W_sel = W_flat[w_indices, :]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.max(np.abs(vals - 1.0))
    
    # Try to construct rank-39 solution
    # Split m=3 into first 2 rows (j=0,1) and last 1 row (j=2)
    # For j=0,1: split p=7 into 2+2+3
    #   k=0,1: Strassen 2x2x2 (rank 7)
    #   k=2,3: Strassen 2x2x2 (rank 7)
    #   k=4,5,6: 2x2x3 (rank 11)
    # For j=2: schoolbook 2x1x7 (rank 14)
    # Total: 7+7+11+14 = 39
    
    R = 39
    # Initialize as 3D arrays for easier indexing in construction
    U = np.zeros((n, m, R))
    V = np.zeros((m, p, R))
    W = np.zeros((p, n, R))
    
    # Strassen 2x2x2 coefficients
    # For sub-problem with i,j in {0,1}, k in {0,1}
    # m1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # m2 = (A[1,0]+A[1,1])*B[0,0]
    # m3 = A[0,0]*(B[0,1]-B[1,1])
    # m4 = A[1,1]*(B[1,0]-B[0,0])
    # m5 = (A[0,0]+A[0,1])*B[1,1]
    # m6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # m7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    def add_strassen(U, V, W, i_offset, j_offset, k_offset, col_start):
        """Add Strassen 2x2x2 terms for i in {i_offset, i_offset+1}, j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}"""
        # alpha_s[i,j] for s=1..7, i,j in {0,1}
        alphas = [
            [[1, 0], [0, 1]],   # s=1
            [[0, 0], [1, 1]],   # s=2
            [[1, 0], [0, 0]],   # s=3
            [[0, 0], [0, 1]],   # s=4
            [[1, 1], [0, 0]],   # s=5
            [[-1, 0], [1, 0]],  # s=6
            [[0, 1], [-1, 0]],  # s=7
        ]
        # beta_s[j,k] for s=1..7, j,k in {0,1}
        betas = [
            [[1, 1], [1, 1]],   # s=1
            [[1, 0], [0, 0]],   # s=2
            [[0, 1], [-1, 1]],  # s=3
            [[0, -1], [1, 0]],  # s=4
            [[0, 0], [0, 1]],   # s=5
            [[1, 1], [0, 0]],   # s=6
            [[0, 1], [1, 1]],   # s=7
        ]
        # gamma_s[i,k] for s=1..7, i,k in {0,1}
        # Note: W index is [k, i] in the 3D array W[k, i, col]
        # The previous code had W[k_offset + kk, i_offset + ii, col] = gammas[s][ii][kk]
        # This means gamma_s[i, k] maps to W[k, i]
        gammas = [
            [[1, 0], [0, 1]],   # s=1: C[0,0]+=1, C[1,1]+=1 -> W[0,0]=1, W[1,1]=1
            [[0, 0], [1, -1]],  # s=2: C[1,0]+=1, C[1,1]+=-1 -> W[0,1]=1, W[1,1]=-1
            [[0, 1], [1, 1]],   # s=3: C[0,1]+=1, C[1,1]+=1 -> W[1,0]=1, W[1,1]=1
            [[1, 0], [1, 0]],   # s=4: C[0,0]+=1, C[1,0]+=1 -> W[0,0]=1, W[0,1]=1
            [[-1, 1], [0, 0]],  # s=5: C[0,0]+=-1, C[0,1]+=1 -> W[0,0]=-1, W[1,0]=1
            [[0, 0], [0, 1]],   # s=6: C[1,1]+=1 -> W[1,1]=1
            [[1, 0], [0, 0]],   # s=7: C[0,0]+=1 -> W[0,0]=1
        ]
        
        for s in range(7):
            col = col_start + s
            for ii in range(2):
                for jj in range(2):
                    U[i_offset + ii, j_offset + jj, col] = alphas[s][ii][jj]
            for jj in range(2):
                for kk in range(2):
                    V[j_offset + jj, k_offset + kk, col] = betas[s][jj][kk]
            for kk in range(2):
                for ii in range(2):
                    # W is indexed [k, i, col]
                    # gamma[s][ii][kk] corresponds to coefficient for C[ii, kk]
                    # So W[kk, ii, col] should be gamma[s][ii][kk]
                    W[k_offset + kk, i_offset + ii, col] = gammas[s][ii][kk]
        return col_start + 7
    
    col = 0
    
    # Part 1: j=0,1, k=0,1 -> Strassen 2x2x2
    col = add_strassen(U, V, W, i_offset=0, j_offset=0, k_offset=0, col_start=col)
    # col is now 7
    
    # Part 2: j=0,1, k=2,3 -> Strassen 2x2x2
    col = add_strassen(U, V, W, i_offset=0, j_offset=0, k_offset=2, col_start=col)
    # col is now 14
    
    # Part 3: j=0,1, k=4,5,6 -> 2x2x3 (Strassen for k=4,5 + schoolbook for k=6)
    # First: Strassen for k=4,5
    col = add_strassen(U, V, W, i_offset=0, j_offset=0, k_offset=4, col_start=col)
    # col is now 21
    
    # Then: schoolbook for k=6 (2x2x1 = 4 terms)
    for ii in range(2):
        for jj in range(2):
            U[ii, jj, col] = 1
            V[jj, 6, col] = 1
            W[6, ii, col] = 1
            col += 1
    # col is now 25
    
    # Part 4: j=2, k=0..6 -> schoolbook 2x1x7 (14 terms)
    for ii in range(2):
        for kk in range(7):
            U[ii, 2, col] = 1
            V[2, kk, col] = 1
            W[kk, ii, col] = 1
            col += 1
    # col should be 39
    
    assert col == R, f"Expected {R} columns, got {col}"
    
    # Flatten to 2D for output
    # U: (n, m, R) -> (n*m, R) with index i*m+j
    # V: (m, p, R) -> (m*p, R) with index j*p+k
    # W: (p, n, R) -> (p*n, R) with index k*n+i
    U_flat = U.reshape(num_u, R)
    V_flat = V.reshape(num_v, R)
    W_flat = W.reshape(num_w, R)
    
    # Verify
    err = verify(U_flat, V_flat, W_flat)
    if err < 1e-10:
        return {"U": U_flat.tolist(), "V": V_flat.tolist(), "W": W_flat.tolist()}
    
    # If verification fails, try to fix or fall back
    print(f"Verification error: {err}")
    
    # Fallback: schoolbook rank 42
    R42 = n * m * p
    U42 = np.zeros((num_u, R42))
    V42 = np.zeros((num_v, R42))
    W42 = np.zeros((num_w, R42))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U42[i * m + j, c] = 1
                V42[j * p + k, c] = 1
                W42[k * n + i, c] = 1
                c += 1
    return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
