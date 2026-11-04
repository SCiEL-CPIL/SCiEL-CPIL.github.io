"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s margin for 150s limit
    
    # Schoolbook decomposition (rank 40) as fallback
    U_school = np.zeros((n * m, n * m * p))
    V_school = np.zeros((m * p, n * m * p))
    W_school = np.zeros((p * n, n * m * p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U_school[idx_u, col] = 1.0
                V_school[idx_v, col] = 1.0
                W_school[idx_w, col] = 1.0
                col += 1
                
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        # Check all 40 one-positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for c in range(r):
                        val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        
        # Check that zero positions are zero
        for idx_u in range(n * m):
            i, j = divmod(idx_u, m)
            for idx_v in range(m * p):
                j2, k = divmod(idx_v, p)
                if j2 != j:
                    continue
                for idx_w in range(p * n):
                    k2, i2 = divmod(idx_w, n)
                    if k2 != k or i2 != i:
                        val = 0.0
                        for c in range(r):
                            val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                        if abs(val) > 1e-10:
                            return False
        return True

    # Strategy: Split m=4 into two blocks of size 2.
    # Block 1: j in {0, 1}
    # Block 2: j in {2, 3}
    
    # For each block, we have two sub-problems based on p:
    # Sub-problem A: k in {0, 1} -> mu_{2,2,2} (Rank 7)
    # Sub-problem B: k in {2, 3, 4} -> mu_{2,2,3} (Rank 12)
    
    # Total Rank = 2 * (7 + 12) = 38
    
    total_rank = 38
    U_new = np.zeros((n * m, total_rank))
    V_new = np.zeros((m * p, total_rank))
    W_new = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # Helper to add Strassen decomposition for mu_{2,2,2}
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global indices: 
    # U: i*4 + (j + j_offset)
    # V: (j + j_offset)*5 + (k + k_offset)
    # W: (k + k_offset)*2 + i
    def add_strassen(j_offset, k_offset):
        nonlocal col_offset
        # Strassen multipliers
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We define the 7 rank-1 terms
        # Each term is (u_vec, v_vec, w_vec)
        
        # Term 1: M1
        # U: A00, A11 -> 1
        # V: B00, B11 -> 1
        # W: C00 -> 1, C11 -> 1
        u1 = np.zeros(n*m); v1 = np.zeros(m*p); w1 = np.zeros(p*n)
        u1[0*4 + (0+j_offset)] = 1.0; u1[1*4 + (1+j_offset)] = 1.0
        v1[(0+j_offset)*5 + (0+k_offset)] = 1.0; v1[(1+j_offset)*5 + (1+k_offset)] = 1.0
        w1[(0+k_offset)*2 + 0] = 1.0; w1[(1+k_offset)*2 + 1] = 1.0
        U_new[:, col_offset] = u1; V_new[:, col_offset] = v1; W_new[:, col_offset] = w1
        col_offset += 1
        
        # Term 2: M2
        # U: A10, A11 -> 1
        # V: B00 -> 1
        # W: C10 -> 1, C11 -> -1
        u2 = np.zeros(n*m); v2 = np.zeros(m*p); w2 = np.zeros(p*n)
        u2[1*4 + (0+j_offset)] = 1.0; u2[1*4 + (1+j_offset)] = 1.0
        v2[(0+j_offset)*5 + (0+k_offset)] = 1.0
        w2[(0+k_offset)*2 + 1] = 1.0; w2[(1+k_offset)*2 + 1] = -1.0
        U_new[:, col_offset] = u2; V_new[:, col_offset] = v2; W_new[:, col_offset] = w2
        col_offset += 1
        
        # Term 3: M3
        # U: A00 -> 1
        # V: B01, B11 -> 1, -1
        # W: C01 -> 1, C11 -> 1
        u3 = np.zeros(n*m); v3 = np.zeros(m*p); w3 = np.zeros(p*n)
        u3[0*4 + (0+j_offset)] = 1.0
        v3[(0+j_offset)*5 + (1+k_offset)] = 1.0; v3[(1+j_offset)*5 + (1+k_offset)] = -1.0
        w3[(1+k_offset)*2 + 0] = 1.0; w3[(1+k_offset)*2 + 1] = 1.0
        U_new[:, col_offset] = u3; V_new[:, col_offset] = v3; W_new[:, col_offset] = w3
        col_offset += 1
        
        # Term 4: M4
        # U: A11 -> 1
        # V: B10, B00 -> 1, -1
        # W: C00 -> 1, C10 -> 1
        u4 = np.zeros(n*m); v4 = np.zeros(m*p); w4 = np.zeros(p*n)
        u4[1*4 + (1+j_offset)] = 1.0
        v4[(1+j_offset)*5 + (0+k_offset)] = 1.0; v4[(0+j_offset)*5 + (0+k_offset)] = -1.0
        w4[(0+k_offset)*2 + 0] = 1.0; w4[(0+k_offset)*2 + 1] = 1.0
        U_new[:, col_offset] = u4; V_new[:, col_offset] = v4; W_new[:, col_offset] = w4
        col_offset += 1
        
        # Term 5: M5
        # U: A00, A01 -> 1
        # V: B11 -> 1
        # W: C00 -> -1, C01 -> 1
        u5 = np.zeros(n*m); v5 = np.zeros(m*p); w5 = np.zeros(p*n)
        u5[0*4 + (0+j_offset)] = 1.0; u5[0*4 + (1+j_offset)] = 1.0
        v5[(1+j_offset)*5 + (1+k_offset)] = 1.0
        w5[(0+k_offset)*2 + 0] = -1.0; w5[(1+k_offset)*2 + 0] = 1.0
        U_new[:, col_offset] = u5; V_new[:, col_offset] = v5; W_new[:, col_offset] = w5
        col_offset += 1
        
        # Term 6: M6
        # U: A10, A00 -> 1, -1
        # V: B00, B01 -> 1
        # W: C11 -> 1
        u6 = np.zeros(n*m); v6 = np.zeros(m*p); w6 = np.zeros(p*n)
        u6[1*4 + (0+j_offset)] = 1.0; u6[0*4 + (0+j_offset)] = -1.0
        v6[(0+j_offset)*5 + (0+k_offset)] = 1.0; v6[(0+j_offset)*5 + (1+k_offset)] = 1.0
        w6[(1+k_offset)*2 + 1] = 1.0
        U_new[:, col_offset] = u6; V_new[:, col_offset] = v6; W_new[:, col_offset] = w6
        col_offset += 1
        
        # Term 7: M7
        # U: A01, A11 -> 1, -1
        # V: B10, B11 -> 1
        # W: C00 -> 1
        u7 = np.zeros(n*m); v7 = np.zeros(m*p); w7 = np.zeros(p*n)
        u7[0*4 + (1+j_offset)] = 1.0; u7[1*4 + (1+j_offset)] = -1.0
        v7[(1+j_offset)*5 + (0+k_offset)] = 1.0; v7[(1+j_offset)*5 + (1+k_offset)] = 1.0
        w7[(0+k_offset)*2 + 0] = 1.0
        U_new[:, col_offset] = u7; V_new[:, col_offset] = v7; W_new[:, col_offset] = w7
        col_offset += 1

    # Helper to add mu_{2,2,3} decomposition (3x mu_{2,2,1})
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1,2} (local k)
    # Global indices:
    # U: i*4 + (j + j_offset)
    # V: (j + j_offset)*5 + (k + k_offset)
    # W: (k + k_offset)*2 + i
    def add_mu_223(j_offset, k_offset):
        nonlocal col_offset
        # For each local k in 0,1,2:
        # T1: A00 B0k C0k
        # T2: A01 B1k C0k
        # T3: A10 B0k C1k
        # T4: A11 B1k C1k
        
        for lk in range(3):
            gk = lk + k_offset
            
            # T1
            u1 = np.zeros(n*m); v1 = np.zeros(m*p); w1 = np.zeros(p*n)
            u1[0*4 + (0+j_offset)] = 1.0
            v1[(0+j_offset)*5 + gk] = 1.0
            w1[gk*2 + 0] = 1.0
            U_new[:, col_offset] = u1; V_new[:, col_offset] = v1; W_new[:, col_offset] = w1
            col_offset += 1
            
            # T2
            u2 = np.zeros(n*m); v2 = np.zeros(m*p); w2 = np.zeros(p*n)
            u2[0*4 + (1+j_offset)] = 1.0
            v2[(1+j_offset)*5 + gk] = 1.0
            w2[gk*2 + 0] = 1.0
            U_new[:, col_offset] = u2; V_new[:, col_offset] = v2; W_new[:, col_offset] = w2
            col_offset += 1
            
            # T3
            u3 = np.zeros(n*m); v3 = np.zeros(m*p); w3 = np.zeros(p*n)
            u3[1*4 + (0+j_offset)] = 1.0
            v3[(0+j_offset)*5 + gk] = 1.0
            w3[gk*2 + 1] = 1.0
            U_new[:, col_offset] = u3; V_new[:, col_offset] = v3; W_new[:, col_offset] = w3
            col_offset += 1
            
            # T4
            u4 = np.zeros(n*m); v4 = np.zeros(m*p); w4 = np.zeros(p*n)
            u4[1*4 + (1+j_offset)] = 1.0
            v4[(1+j_offset)*5 + gk] = 1.0
            w4[gk*2 + 1] = 1.0
            U_new[:, col_offset] = u4; V_new[:, col_offset] = v4; W_new[:, col_offset] = w4
            col_offset += 1

    # Apply to Block 1 (j_offset=0)
    add_strassen(0, 0)       # k in {0,1}
    add_mu_223(0, 2)         # k in {2,3,4}
    
    # Apply to Block 2 (j_offset=2)
    add_strassen(2, 0)       # k in {0,1}
    add_mu_223(2, 2)         # k in {2,3,4}
    
    # Verify new decomposition
    if verify(U_new, V_new, W_new):
        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
