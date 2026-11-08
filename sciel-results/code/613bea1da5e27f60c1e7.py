import numpy as np
import time

DIMENSIONS = [2, 3, 9]
DEADLINE = 110.0  # 10s buffer

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We aim for Rank 50 initially
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def verify_np(U, V, W, n, m, p):
        """Verify decomposition using NumPy: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # We need to check T_{ijk} = sum_r U_{im+j, r} V_{jp+k, r} W_{kn+i, r}
        
        # Reshape U to (n, m, R)
        U_reshaped = U_np.reshape(n, m, -1)
        # Reshape V to (m, p, R)
        V_reshaped = V_np.reshape(m, p, -1)
        # Reshape W to (p, n, R)
        W_reshaped = W_np.reshape(p, n, -1)
        
        # We want to compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] for all i,j,k
        # This is a tensor contraction.
        # Let's iterate over j to keep it simple and fast enough for R=50
        valid = True
        for j in range(m):
            # U[j] is (n, R) -> actually U_reshaped[:, j, :] is (n, R)
            U_j = U_reshaped[:, j, :] # (n, R)
            # V[j] is (p, R) -> actually V_reshaped[j, :, :] is (p, R)
            V_j = V_reshaped[j, :, :] # (p, R)
            
            # We need to check for all i, k:
            # sum_r U_j[i, r] * V_j[k, r] * W_reshaped[k, i, r] == 1
            
            # W_reshaped is (p, n, R). We need W_reshaped[k, i, r].
            # Let's transpose W_reshaped to (n, p, R) to access W[i, k, r] easily?
            # W_reshaped.T is (R, n, p). 
            # W_reshaped.transpose(1, 0, 2) is (n, p, R).
            W_reshaped_T = W_reshaped.transpose(1, 0, 2) # (n, p, R)
            
            # For each i:
            for i in range(n):
                U_i = U_j[i, :] # (R,)
                W_i = W_reshaped_T[i, :, :] # (p, R)
                
                # We need sum_r U_i[r] * V_j[k, r] * W_i[k, r] for all k
                # This is element-wise product of U_i (broadcast to (p, R)) with V_j (p, R) and W_i (p, R), then sum over R.
                # V_j is (p, R). U_i is (R,).
                # prod = V_j * W_i * U_i
                # sum over axis 1
                # result is (p,)
                # Check if all are 1.
                
                # V_j * U_i broadcasts U_i to (p, R)
                term = V_j * W_i * U_i
                sums = np.sum(term, axis=1)
                
                if not np.allclose(sums, 1.0, atol=1e-9):
                    return False
        return True

    col = 0
    
    # Part 1: A_L (2x2) * B_T (2x9)
    # A_L indices: 0, 1, 3, 4 (A_{00}, A_{01}, A_{10}, A_{11})
    # B_T indices: 0..8, 9..17 (B_{0k}, B_{1k})
    
    # We process 4 blocks of 2 columns of B
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices for this block
        v_e = 0 * p + k0 # B_{0, k0}
        v_f = 0 * p + k1 # B_{0, k1}
        v_g = 1 * p + k0 # B_{1, k0}
        v_h = 1 * p + k1 # B_{1, k1}
        
        # A indices
        u_a = 0 * m + 0 # A_{00}
        u_b = 0 * m + 1 # A_{01}
        u_c = 1 * m + 0 # A_{10}
        u_d = 1 * m + 1 # A_{11}
        
        # W indices: k*n+i
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # Strassen-like decomposition for 2x2x2 block
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        
        # M2 = (c+d)e
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        W[w_10][col+1] += 1
        W[w_11][col+1] -= 1
        
        # M3 = a(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        W[w_01][col+2] += 1
        W[w_11][col+2] += 1
        
        # M4 = d(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        W[w_00][col+3] += 1
        W[w_10][col+3] += 1
        
        # M5 = (a+b)h
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        W[w_00][col+4] -= 1
        W[w_01][col+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        W[w_11][col+5] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        W[w_00][col+6] += 1
        
        col += 7
        
    # Part 2: Remaining column k=8 for A_L * B_T
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    # Direct multiplication for 2x2x1 (rank 4)
    # C08 = a*e + b*g
    U[u_a][col] += 1
    V[v_e][col] += 1
    W[w_08][col] += 1
    
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    W[w_08][col+1] += 1
    
    # C18 = c*e + d*g
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    W[w_18][col+2] += 1
    
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # Part 3: A_R (2x1) * B_B (1x9)
    # A_R indices: 2, 5 (A_{02}, A_{12})
    # B_B indices: 18..26 (B_{2k})
    
    u_ar0 = 0 * m + 2
    u_ar1 = 1 * m + 2
    
    for k in range(9):
        v_b = 2 * p + k
        
        # C_{k, 0} += A_{02} * B_{2k}
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        # C_{k, 1} += A_{12} * B_{2k}
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2

    # Verify the decomposition
    if not verify_np(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    # Try to reduce rank by removing redundant columns if we have time
    current_time = time.time()
    if current_time - start_time < DEADLINE - 5:
        for col_to_remove in range(rank):
            U_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
            V_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
            W_new = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
            if verify_np(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                break
    
    return {"U": U, "V": V, "W": W}
