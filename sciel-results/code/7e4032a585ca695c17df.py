import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # We construct a rank 19 decomposition
    # 1. 2x2x2 Winograd (Rank 7) for A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # 2. 2x2x1 Schoolbook (Rank 4) for A[:, 0:2] * B[0:2, 2:3] -> C[:, 2:3]
    # 3. 2x2x2 Winograd (Rank 7) for A[:, 0:2] * B[0:2, 3:5] -> C[:, 3:5]
    # 4. 2x1x5 Outer Product (Rank 1) for A[:, 2:3] * B[2:3, :] -> C[:, :]
    
    U = []
    V = []
    W = []
    
    def add_factors(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)

    # Helper to create Winograd 2x2x2 factors
    # A is 2x2, B is 2x2, C is 2x2
    # Indices:
    # A: 0, 1, 2, 3 (i*m + j) -> m=2. 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0, 1, 2, 3 (j*p + k) -> p=2. 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0, 1, 2, 3 (k*n + i) -> n=2. 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
    # Note: W index is k*n + i.
    # C_00 is at index 0*2+0=0.
    # C_01 is at index 1*2+0=2.
    # C_10 is at index 0*2+1=1.
    # C_11 is at index 1*2+1=3.
    
    def winograd_222(a_base, b_base, c_base):
        # a_base, b_base, c_base are offsets in the global vectors
        # We will add 7 factors
        # m1 = A00 B00
        u = np.zeros(6); v = np.zeros(6); w = np.zeros(4)
        # A indices: a_base + 0 (0,0), a_base+1 (0,1), a_base+2 (1,0), a_base+3 (1,1)
        # B indices: b_base + 0 (0,0), b_base+1 (0,1), b_base+2 (1,0), b_base+3 (1,1)
        # C indices: c_base + 0 (0,0), c_base+1 (1,0), c_base+2 (0,1), c_base+3 (1,1)
        
        # 1. m1 = A00 B00
        u[a_base] = 1; v[b_base] = 1; w[c_base] = 1
        add_factors(u.copy(), v.copy(), w.copy())
        
        # 2. m2 = A01 B10
        u = np.zeros(6); v = np.zeros(6); w = np.zeros(4)
        u[a_base+1] = 1; v[b_base+2] = 1; w[c_base] = 1
        add_factors(u.copy(), v.copy(), w.copy())
        
        # 3. m3 = A10 B01
        u = np.zeros(6); v = np.zeros(6); w = np.zeros(4)
        u[a_base+2] = 1; v[b_base+1] = 1; w[c_base] = 1
        add_factors(u.copy(), v.copy(), w.copy())
        
        # 4. m4 = A11 B11
        u = np.zeros(6); v = np.zeros(6); w = np.zeros(4)
        u[a_base+3] = 1; v[b_base+3] = 1; w[c_base] = 1
        add_factors(u.copy(), v.copy(), w.copy())
        
        # 5. m5 = (A00+A01)(B00+B10)
        u = np.zeros(6); v = np.zeros(6); w = np.zeros(4)
        u[a_base] = 1; u[a_base+1] = 1
        v[b_base] = 1; v[b_base+2] = 1
        # C00 = A00B00 + A01B10
        # m5 contributes to C00?
        # C00 = m1 + m2 + ...?
        # Standard Winograd:
        # C00 = m1 + m5 - m2 - m6?
        # Let's use the explicit formula:
        # C00 = A00B00 + A01B10
        # C01 = A00B01 + A01B11
        # C10 = A10B00 + A11B10
        # C11 = A10B01 + A11B11
        
        # m1 = A00B00
        # m2 = A01B10
        # m3 = A10B01
        # m4 = A11B11
        # m5 = (A00+A01)(B00+B10) = A00B00 + A00B10 + A01B00 + A01B10
        # m6 = (A00+A10)(B00+B01) = A00B00 + A00B01 + A10B00 + A10B01
        # m7 = (A01+A11)(B10+B11) = A01B10 + A01B11 + A11B10 + A11B11
        
        # C00 = A00B00 + A01B10
        # C00 = m1 + m2
        # This is Rank 4 for C00?
        # We want to share.
        # C00 = m1 + m5 - (A00B10 + A01B00) - m2?
        # A00B10 is part of m5. A01B00 is part of m5.
        # Let's use the standard Winograd 7 mults:
        # m1 = A00B00
        # m2 = A01B10
        # m3 = A10B01
        # m4 = A11B11
        # m5 = (A00+A01)(B00+B10)
        # m6 = (A00+A10)(B00+B01)
        # m7 = (A01+A11)(B10+B11)
        
        # C00 = m1 + m5 - m2 - (A00B10 + A01B00)?
        # No, C00 = A00B00 + A01B10.
        # m1 = A00B00. m2 = A01B10.
        # So C00 = m1 + m2.
        # This doesn't use m5, m6, m7.
        
        # Correct Winograd:
        # m1 = (A00+A01)(B00+B10)
        # m2 = (A00+A10)(B00+B01)
        # m3 = (A01+A11)(B10+B11)
        # m4 = A00B00
        # m5 = A01B10
        # m6 = A10B01
        # m7 = A11B11
        
        # C00 = m4 + m5 + m1 - m2 - m3?
        # m1 = A00B00 + A00B10 + A01B00 + A01B10
        # m2 = A00B00 + A00B01 + A10B00 + A10B01
        # m3 = A01B10 + A01B11 + A11B10 + A11B11
        
        # C00 = A00B00 + A01B10
        # C00 = m4 + m5
        # This is just schoolbook for C00.
        
        # Let's use the known Winograd 7 mults:
        # m1 = A00B00
        # m2 = A01B10
        # m3 = A10B01
        # m4 = A11B11
        # m5 = (A00+A01)(B00+B10)
        # m6 = (A00+A10)(B00+B01)
        # m7 = (A01+A11)(B10+B11)
        
        # C00 = m1 + m5 - m2 - (A00B10 + A01B00)
        # This is not linear in m's.
        
        # I will use the 7 mults from a reliable source.
        # m1 = (a00+a01)(b00+b10)
        # m2 = (a00+a10)(b00+b01)
        # m3 = (a01+a11)(b10+b11)
        # m4 = a00b00
        # m5 = a01b10
        # m6 = a10b01
        # m7 = a11b11
        
        # c00 = m1 + m4 - m2 - m3 + m5?
        # c00 = a00b00 + a01b10
        # m1 = a00b00 + a00b10 + a01b00 + a01b10
        # m2 = a00b00 + a00b01 + a10b00 + a10b01
        # m3 = a01b10 + a01b11 + a11b10 + a11b11
        
        # c00 = m4 + m5
        # c01 = m6 + m7? No.
        # c01 = a00b01 + a01b11
        # c10 = a10b00 + a11b10
        # c11 = a10b01 + a11b11
        
        # This decomposition is tricky to get right quickly.
        # I will use Schoolbook for 2x2x2 to be safe?
        # Rank 4.
        # Total Rank = 4 + 4 + 4 + 1 = 13.
        # 13 < 30.
        # This is a very safe and significant improvement.
        
        pass

    # I will implement Schoolbook for all parts to ensure correctness, then optimize.
    # Rank 13 is much better than 30.
    
    # Part 1: 2x2x2 (A[:,0:2], B[0:2,0:2] -> C[:,0:2])
    # Part 2: 2x2x1 (A[:,0:2], B[0:2,2:3] -> C[:,2:3])
    # Part 3: 2x2x2 (A[:,0:2], B[0:2,3:5] -> C[:,3:5])
    # Part 4: 2x1x5 (A[:,2:3], B[2:3,:] -> C[:,:])
    
    # Schoolbook for 2x2x2 (Rank 4)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u = np.zeros(6); v = np.zeros(6); w = np.zeros(4)
                u[i*2+j] = 1
                v[j*2+k] = 1
                w[k*2+i] = 1
                add_factors(u, v, w)
                
    # Schoolbook for 2x2x1 (Rank 4)
    for i in range(2):
        for j in range(2):
            u = np.zeros(6); v = np.zeros(6); w = np.zeros(4)
            u[i*2+j] = 1
            v[j*2+2] = 1 # k=2 in local, but global k=2
            w[2*2+i] = 1 # k=2, i
            add_factors(u, v, w)

    # Schoolbook for 2x2x2 (Rank 4)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u = np.zeros(6); v = np.zeros(6); w = np.zeros(4)
                u[i*2+j] = 1
                v[j*2+(k+3)] = 1 # k=3,4
                w[(k+3)*2+i] = 1
                add_factors(u, v, w)

    # Outer Product for 2x1x5 (Rank 1)
    u = np.zeros(6); v = np.zeros(6); w = np.zeros(4)
    u[2] = 1; u[3] = 1 # A[:, 2] -> indices 2, 3
    v[2] = 1 # B[2, :] -> row 2. Indices 2,3,4,5,6?
    # B is 3x5. Index j*5+k.
    # j=2. k=0..4. Indices 10, 11, 12, 13, 14.
    # Wait, my V vector size is 6?
    # m=3, p=5. V size 15.
    # U size 6. W size 10.
    
    # I need to fix the sizes.
    # U: n*m = 6.
    # V: m*p = 15.
    # W: p*n = 10.
    
    # Restart construction with correct sizes.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_factors_correct(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    # Part 1: 2x2x2 (A[:,0:2], B[0:2,0:2] -> C[:,0:2])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u = np.zeros(6); v = np.zeros(15); w = np.zeros(10)
                u[i*3+j] = 1 # A index i*3+j
                v[j*5+k] = 1 # B index j*5+k
                w[k*2+i] = 1 # C index k*2+i
                add_factors_correct(u, v, w)

    # Part 2: 2x2x1 (A[:,0:2], B[0:2,2:3] -> C[:,2:3])
    for i in range(2):
        for j in range(2):
            u = np.zeros(6); v = np.zeros(15); w = np.zeros(10)
            u[i*3+j] = 1
            v[j*5+2] = 1 # k=2
            w[2*2+i] = 1
            add_factors_correct(u, v, w)

    # Part 3: 2x2x2 (A[:,0:2], B[0:2,3:5] -> C[:,3:5])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u = np.zeros(6); v = np.zeros(15); w = np.zeros(10)
                u[i*3+j] = 1
                v[j*5+(k+3)] = 1
                w[(k+3)*2+i] = 1
                add_factors_correct(u, v, w)

    # Part 4: 2x1x5 (A[:,2:3], B[2:3,:] -> C[:,:])
    # C_ik += A_i2 B_2k
    # This is Rank 1?
    # C_ik = A_i2 B_2k
    # u_i = A_i2. v_j = B_jk?
    # B index is j*5+k.
    # We need v to be 1 for j=2, all k.
    # But v is a single vector.
    # If we use Rank 1:
    # u = [0,0,1, 0,0,1] (A_i2)
    # v = [0,0,0, 0,0,0, 1,1,1,1,1, 0,0,0,0,0]?
    # No, v must be constant for all k?
    # C_ik = u_i v_k w?
    # v is indexed by j,k.
    # If we use Rank 1, v_{j,k} must be separable?
    # v_{j,k} = 1 if j=2 else 0.
    # Then C_ik = sum_c u_{i,c} v_{j,c} w_{k,c}.
    # If c=1: u_i = A_i2. v_{j,1} = delta_{j,2}. w_{k,1} = 1.
    # C_ik = A_i2 * delta_{j,2} * 1.
    # This gives C_ik = A_i2 for all k?
    # No, w_{k,1} should be B_2k.
    # But w is a vector. It cannot depend on k in a way that matches B_2k for all k unless B_2k is constant.
    # So Rank 1 is NOT sufficient for 2x1x5.
    # Rank 5 is sufficient (Schoolbook).
    
    for k in range(5):
        u = np.zeros(6); v = np.zeros(15); w = np.zeros(10)
        u[2] = 1; u[5] = 1 # A_02, A_12
        v[2*5+k] = 1 # B_2k
        w[k*2+0] = 1; w[k*2+1] = 1 # C_0k, C_1k
        add_factors_correct(u, v, w)

    U_mat = np.column_stack(U_list)
    V_mat = np.column_stack(V_list)
    W_mat = np.column_stack(W_list)
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
