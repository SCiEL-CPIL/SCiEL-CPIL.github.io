import numpy as np

def construct():
    n, m, p = 2, 3, 7
    
    # Helper to add a decomposition term
    # U, V, W are lists of lists
    # We will build numpy arrays and convert at the end
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u, v, w):
        # u: shape (n*m,)
        # v: shape (m*p,)
        # w: shape (p*n,)
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    # 1. Handle j=2 part (Rank 1)
    # A[:, 2] * B[2, :]
    # u: index i*m + j. j=2. i in 0..1.
    # u[0] = 1 (i=0, j=2), u[1] = 0 (i=1, j=2)? No.
    # u is vector of length n*m = 6.
    # u[i*m + j].
    # We want u to be 1 at i=0, j=2 and i=1, j=2?
    # No, the term is A[i,2] * B[2,k].
    # This is not a single rank-1 term in the tensor sense if we consider the sum over i and k.
    # T[i,2,k] = 1.
    # u[i*3+2] = 1 for all i? No.
    # u is fixed for the term.
    # If u[i*3+2] = 1 for i=0,1, then u = [0,0,1, 0,0,1].
    # v[j*7+k]. j=2. v[2*7+k] = 1 for all k?
    # v = [0..0, 1,1,1,1,1,1,1].
    # w[k*2+i].
    # We need w[k*2+i] such that sum u*v*w reconstructs T.
    # T[i,2,k] = u[i*3+2] * v[2*7+k] * w[k*2+i].
    # If u[i*3+2]=1, v[2*7+k]=1, then we need w[k*2+i]=1.
    # So w is all 1s?
    # Let's check: C[i,k] = sum_j A[i,j] B[j,k].
    # For j=2, term is A[i,2] B[2,k].
    # Decomposition term: u * v * w.
    # u[i*3+2] * v[2*7+k] * w[k*2+i] = 1 * 1 * w[k*2+i].
    # We want this to equal 1 (coefficient of A[i,2]B[2,k]).
    # So w must be 1 everywhere.
    # So Rank 1 term:
    u_j2 = np.zeros(6); u_j2[2] = 1; u_j2[5] = 1
    v_j2 = np.zeros(21); v_j2[14:21] = 1
    w_j2 = np.ones(14)
    add_term(u_j2, v_j2, w_j2)

    # 2. Handle j in {0,1} part (2x2x7)
    # Split k into {0,1,2} and {3,4,5,6}
    
    # 2.1 k in {0,1,2} (2x2x3)
    # Split k into {0,1} and {2}
    
    # 2.1.1 k in {0,1} (2x2x2) -> Strassen Rank 7
    # Indices: i in 0..1, j in 0..1, k in 0..1
    # u length 6, v length 21, w length 14
    # We need to map Strassen terms to these indices.
    
    def add_strassen_2x2x2(offset_i=0, offset_j=0, offset_k=0):
        # offset_i, offset_j, offset_k are 0 or 1 (shift in the 2x2 block)
        # Actually, we are in the j={0,1}, k={0,1} subblock.
        # u index: i*3 + j. j is 0 or 1.
        # v index: j*7 + k. j is 0 or 1. k is 0 or 1.
        # w index: k*2 + i. k is 0 or 1. i is 0 or 1.
        
        # Strassen M1..M7
        # M1: (A00+A11)(B00+B11)
        # A00: i=0,j=0. A11: i=1,j=1.
        # B00: j=0,k=0. B11: j=1,k=1.
        
        # Term 1: M1
        u1 = np.zeros(6); u1[0] = 1; u1[1] = 1 # i=0,j=0 and i=1,j=1? No.
        # u corresponds to A.
        # u[i*3+j].
        # A00 is index 0. A11 is index 1*3+1 = 4.
        # u1[0]=1, u1[4]=1.
        v1 = np.zeros(21); v1[0] = 1; v1[8] = 1 # j=0,k=0 (idx 0) and j=1,k=1 (idx 1*7+1=8).
        w1 = np.zeros(14)
        # C00 = M1 + ...
        # C11 = M1 - ...
        # C01 = ...
        # C10 = ...
        # M1 contributes to C00, C11.
        # C00: i=0,k=0. w[0*2+0]=w[0].
        # C11: i=1,k=1. w[1*2+1]=w[3].
        w1[0] = 1; w1[3] = 1
        add_term(u1, v1, w1)
        
        # Term 2: M2 = (A10+A11)B00
        u2 = np.zeros(6); u2[3] = 1; u2[4] = 1 # A10 (i=1,j=0 -> idx 3), A11 (idx 4)
        v2 = np.zeros(21); v2[0] = 1 # B00
        w2 = np.zeros(14)
        # C10 = M2 + ... (i=1,k=0 -> w[0*2+1]=w[1])
        # C11 = -M2 + ... (i=1,k=1 -> w[3])
        w2[1] = 1; w2[3] = -1
        add_term(u2, v2, w2)
        
        # Term 3: M3 = A00(B01-B11)
        u3 = np.zeros(6); u3[0] = 1 # A00
        v3 = np.zeros(21); v3[1] = 1; v3[8] = -1 # B01 (j=0,k=1 -> idx 1), B11 (idx 8)
        w3 = np.zeros(14)
        # C01 = M3 + ... (i=0,k=1 -> w[1*2+0]=w[2])
        # C11 = M3 + ... (i=1,k=1 -> w[3])
        w3[2] = 1; w3[3] = 1
        add_term(u3, v3, w3)
        
        # Term 4: M4 = A11(B10-B00)
        u4 = np.zeros(6); u4[4] = 1 # A11
        v4 = np.zeros(21); v4[7] = 1; v4[0] = -1 # B10 (j=1,k=0 -> idx 7), B00 (idx 0)
        w4 = np.zeros(14)
        # C00 = M4 + ... (w[0])
        # C10 = M4 + ... (w[1])
        w4[0] = 1; w4[1] = 1
        add_term(u4, v4, w4)
        
        # Term 5: M5 = (A00+A01)B11
        u5 = np.zeros(6); u5[0] = 1; u5[1] = 1 # A00, A01 (i=0,j=1 -> idx 1)
        v5 = np.zeros(21); v5[8] = 1 # B11
        w5 = np.zeros(14)
        # C00 = -M5 + ... (w[0])
        # C01 = M5 + ... (w[2])
        w5[0] = -1; w5[2] = 1
        add_term(u5, v5, w5)
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        u6 = np.zeros(6); u6[3] = 1; u6[0] = -1 # A10, A00
        v6 = np.zeros(21); v6[0] = 1; v6[1] = 1 # B00, B01
        w6 = np.zeros(14)
        # C11 = M6 + ... (w[3])
        w6[3] = 1
        add_term(u6, v6, w6)
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        u7 = np.zeros(6); u7[1] = 1; u7[4] = -1 # A01, A11
        v7 = np.zeros(21); v7[7] = 1; v7[8] = 1 # B10, B11
        w7 = np.zeros(14)
        # C00 = M7 + ... (w[0])
        w7[0] = 1
        add_term(u7, v7, w7)

    add_strassen_2x2x2()

    # 2.1.2 k=2 (2x2x1) -> Rank 2
    # C[i,2] = A[i,0]B[0,2] + A[i,1]B[1,2]
    # Term 1: A[i,0] B[0,2]
    u_t1 = np.zeros(6); u_t1[0] = 1; u_t1[3] = 1 # A00, A10
    v_t1 = np.zeros(21); v_t1[2] = 1 # B02 (j=0,k=2 -> idx 2)
    w_t1 = np.zeros(14); w_t1[2*2+0] = 1; w_t1[2*2+1] = 1 # C02, C12
    add_term(u_t1, v_t1, w_t1)
    
    # Term 2: A[i,1] B[1,2]
    u_t2 = np.zeros(6); u_t2[1] = 1; u_t2[4] = 1 # A01, A11
    v_t2 = np.zeros(21); v_t2[9] = 1 # B12 (j=1,k=2 -> idx 9)
    w_t2 = np.zeros(14); w_t2[2*2+0] = 1; w_t2[2*2+1] = 1
    add_term(u_t2, v_t2, w_t2)

    # 2.2 k in {3,4,5,6} (2x2x4)
    # Split into k={3,4} and k={5,6}
    # Both are 2x2x2.
    
    # Shift k by 3.
    # k=3,4 -> offset 3.
    # k=5,6 -> offset 5.
    
    def add_strassen_2x2x2_k_shift(shift):
        # Similar to above, but v and w indices shifted.
        # v index: j*7 + k. k = shift + 0 or shift + 1.
        # w index: k*2 + i. k = shift + 0 or shift + 1.
        
        # M1
        u1 = np.zeros(6); u1[0] = 1; u1[4] = 1
        v1 = np.zeros(21); v1[0*7+shift] = 1; v1[1*7+(shift+1)] = 1
        w1 = np.zeros(14); w1[(shift)*2+0] = 1; w1[(shift+1)*2+1] = 1
        add_term(u1, v1, w1)
        
        # M2
        u2 = np.zeros(6); u2[3] = 1; u2[4] = 1
        v2 = np.zeros(21); v2[0*7+shift] = 1
        w2 = np.zeros(14); w2[(shift)*2+1] = 1; w2[(shift+1)*2+1] = -1
        add_term(u2, v2, w2)
        
        # M3
        u3 = np.zeros(6); u3[0] = 1
        v3 = np.zeros(21); v3[0*7+(shift+1)] = 1; v3[1*7+(shift+1)] = -1
        w3 = np.zeros(14); w3[(shift+1)*2+0] = 1; w3[(shift+1)*2+1] = 1
        add_term(u3, v3, w3)
        
        # M4
        u4 = np.zeros(6); u4[4] = 1
        v4 = np.zeros(21); v4[1*7+shift] = 1; v4[0*7+shift] = -1
        w4 = np.zeros(14); w4[(shift)*2+0] = 1; w4[(shift)*2+1] = 1
        add_term(u4, v4, w4)
        
        # M5
        u5 = np.zeros(6); u5[0] = 1; u5[1] = 1
        v5 = np.zeros(21); v5[1*7+(shift+1)] = 1
        w5 = np.zeros(14); w5[(shift)*2+0] = -1; w5[(shift+1)*2+0] = 1
        add_term(u5, v5, w5)
        
        # M6
        u6 = np.zeros(6); u6[3] = 1; u6[0] = -1
        v6 = np.zeros(21); v6[0*7+shift] = 1; v6[0*7+(shift+1)] = 1
        w6 = np.zeros(14); w6[(shift+1)*2+1] = 1
        add_term(u6, v6, w6)
        
        # M7
        u7 = np.zeros(6); u7[1] = 1; u7[4] = -1
        v7 = np.zeros(21); v7[1*7+shift] = 1; v7[1*7+(shift+1)] = 1
        w7 = np.zeros(14); w7[(shift)*2+0] = 1
        add_term(u7, v7, w7)

    add_strassen_2x2x2_k_shift(3)
    add_strassen_2x2x2_k_shift(5)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
