# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    
    # Naive decomposition (Rank 40)
    rank_naive = n * m * p
    U_naive = np.zeros((n*m, rank_naive))
    V_naive = np.zeros((m*p, rank_naive))
    W_naive = np.zeros((p*n, rank_naive))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m + j, col] = 1.0
                V_naive[j*p + k, col] = 1.0
                W_naive[k*n + i, col] = 1.0
                col += 1
    
    # Try Strassen Block Decomposition (Rank 36)
    # We'll build it and verify. If it fails, return Naive.
    
    rank_strassen = 36
    U_s = np.zeros((n*m, rank_strassen))
    V_s = np.zeros((m*p, rank_strassen))
    W_s = np.zeros((p*n, rank_strassen))
    
    def u_idx(i, j): return i*m + j
    def v_idx(j, k): return j*p + k
    def w_idx(k, i): return k*n + i
    
    col_s = 0
    
    def add_strassen_2x2x2(ai0, ai1, aj0, aj1, bk0, bk1):
        nonlocal col_s
        # aj0, aj1 are the inner indices (rows of B)
        # Note: In previous code, bj0, bj1 were passed but assumed to be aj0, aj1.
        # Here we assume the block of B uses rows aj0, aj1.
        
        a00 = u_idx(ai0, aj0)
        a01 = u_idx(ai0, aj1)
        a10 = u_idx(ai1, aj0)
        a11 = u_idx(ai1, aj1)
        
        b00 = v_idx(aj0, bk0)
        b01 = v_idx(aj0, bk1)
        b10 = v_idx(aj1, bk0)
        b11 = v_idx(aj1, bk1)
        
        c00 = w_idx(bk0, ai0)
        c01 = w_idx(bk1, ai0)
        c10 = w_idx(bk0, ai1)
        c11 = w_idx(bk1, ai1)
        
        # M1: (A00+A11)(B00+B11)
        U_s[a00, col_s] += 1; U_s[a11, col_s] += 1
        V_s[b00, col_s] += 1; V_s[b11, col_s] += 1
        W_s[c00, col_s] += 1; W_s[c11, col_s] += 1
        col_s += 1
        
        # M2: (A10+A11)B00
        U_s[a10, col_s] += 1; U_s[a11, col_s] += 1
        V_s[b00, col_s] += 1
        W_s[c10, col_s] += 1; W_s[c11, col_s] -= 1
        col_s += 1
        
        # M3: A00(B01-B11)
        U_s[a00, col_s] += 1
        V_s[b01, col_s] += 1; V_s[b11, col_s] -= 1
        W_s[c01, col_s] += 1; W_s[c11, col_s] += 1
        col_s += 1
        
        # M4: A11(B10-B00)
        U_s[a11, col_s] += 1
        V_s[b10, col_s] += 1; V_s[b00, col_s] -= 1
        W_s[c00, col_s] -= 1; W_s[c10, col_s] += 1 # Fixed sign for C00
        col_s += 1
        
        # M5: (A00+A01)B11
        U_s[a00, col_s] += 1; U_s[a01, col_s] += 1
        V_s[b11, col_s] += 1
        W_s[c00, col_s] += 1; W_s[c01, col_s] += 1 # Fixed sign for C00
        col_s += 1
        
        # M6: (A10-A00)(B00+B01)
        U_s[a10, col_s] += 1; U_s[a00, col_s] -= 1
        V_s[b00, col_s] += 1; V_s[b01, col_s] += 1
        W_s[c11, col_s] -= 1 # Fixed sign for C11
        col_s += 1
        
        # M7: (A01-A11)(B10+B11)
        U_s[a01, col_s] += 1; U_s[a11, col_s] -= 1
        V_s[b10, col_s] += 1; V_s[b11, col_s] += 1
        W_s[c00, col_s] -= 1 # Fixed sign for C00
        col_s += 1

    def add_vec_mat_2x2x1(ai0, ai1, aj0, aj1, bk):
        nonlocal col_s
        a00 = u_idx(ai0, aj0)
        a01 = u_idx(ai0, aj1)
        a10 = u_idx(ai1, aj0)
        a11 = u_idx(ai1, aj1)
        
        b0 = v_idx(aj0, bk)
        b1 = v_idx(aj1, bk)
        
        c0 = w_idx(bk, ai0)
        c1 = w_idx(bk, ai1)
        
        U_s[a00, col_s] = 1; V_s[b0, col_s] = 1; W_s[c0, col_s] += 1; col_s += 1
        U_s[a01, col_s] = 1; V_s[b1, col_s] = 1; W_s[c0, col_s] += 1; col_s += 1
        U_s[a10, col_s] = 1; V_s[b0, col_s] = 1; W_s[c1, col_s] += 1; col_s += 1
        U_s[a11, col_s] = 1; V_s[b1, col_s] = 1; W_s[c1, col_s] += 1; col_s += 1

    # Blocks
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1) # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen_2x2x2(0, 1, 0, 1, 2, 3) # A[0:2,0:2] * B[0:2,2:4] -> C[2:4,0:2]
    add_vec_mat_2x2x1(0, 1, 0, 1, 4)     # A[0:2,0:2] * B[0:2,4]   -> C[4,0:2]
    add_strassen_2x2x2(0, 1, 2, 3, 0, 1) # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    add_strassen_2x2x2(0, 1, 2, 3, 2, 3) # A[0:2,2:4] * B[2:4,2:4] -> C[2:4,0:2]
    add_vec_mat_2x2x1(0, 1, 2, 3, 4)     # A[0:2,2:4] * B[2:4,4]   -> C[4,0:2]
    
    # Verify
    def verify_decomp(U, V, W, rank):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((p, n))
        for l in range(rank):
            u_prod = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    u_prod[i] += U[i*m+j, l] * A[i, j]
            v_prod = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    v_prod[k] += V[j*p+k, l] * B[j, k]
            for k in range(p):
                for i in range(n):
                    C_computed[k, i] += W[k*n+i, l] * u_prod[i] * v_prod[k]
                    
        return np.max(np.abs(C_expected.T - C_computed))
    
    err_s = verify_decomp(U_s, V_s, W_s, col_s)
    
    if err_s < 1e-10:
        return {"U": U_s[:, :col_s], "V": V_s[:, :col_s], "W": W_s[:, :col_s]}
    else:
        return {"U": U_naive, "V": V_naive, "W": W_naive}
# EVOLVE-BLOCK-END
