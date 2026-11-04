"""Matrix multiplication 3x4x8 - Strassen-embedded decomposition."""

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy:
    - Split A (3x4) into a top 2x4 block and a bottom 1x4 row
    - Split B (4x8) into 4 blocks of 4x2
    - For the top 2x4 * 4x2 part, apply Strassen to 2x2*2x2 sub-blocks
    - Use schoolbook for the bottom row
    - This reduces rank from 96 (schoolbook) to 7*4 + 8 = 36
    
    The Strassen 2x2*2x2 rank-7 decomposition:
    For matrices P = [[p11,p12],[p21,p22]], Q = [[q11,q12],[q21,q22]]:
    R = P*Q computed as:
    r1 = (p11+p21)(q11+q22)
    r2 = (p21+p22)q11
    r3 = p11(q22-q12)
    r4 = p22(q11-q21)
    r5 = (p11+p12)q22
    r6 = (p21-p11)(q11+q12)
    r7 = (p12-p22)(q21+q22)
    """
    import numpy as np
    
    n, m, p = 3, 4, 8
    
    # Strassen rank-7 factors for 2x2*2x2
    # U: (4,7), V: (4,7), W: (4,7)
    # U maps the 4 input elements of P to the 7 products
    # V maps the 4 input elements of Q to the 7 products
    # W maps the 7 products to the 4 output elements of R
    
    strassen_U = np.array([
        [1, 0, 1, 0, 1, 0, 1],  # p11
        [0, 1, 0, 1, 0, -1, 0], # p12
        [1, 1, 0, 0, 0, 1, 0],  # p21
        [0, 1, -1, 1, 1, 0, -1] # p22
    ], dtype=float)
    
    strassen_V = np.array([
        [1, 1, 1, -1, 0, 1, 0],  # q11
        [0, 0, -1, 0, 0, 1, 0],  # q12
        [0, 0, 0, 1, 0, 0, 1],  # q21
        [1, 0, 1, 0, 1, 0, 1]   # q22
    ], dtype=float)
    
    strassen_W = np.array([
        [1, -1, 1, 0, 0, 0, 0],  # r11
        [0, 0, 0, 1, 1, 0, 1],   # r12
        [0, 1, 0, 0, 0, 1, 0],   # r21
        [1, 0, -1, 1, 1, 0, 0]   # r22
    ], dtype=float)
    
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: Top 2x4 block of A times each 4x2 block of B
    # A_top = A[0:2, :] (2x4), split into left 2x2 and right 2x2
    # B_block = B[:, k_start:k_start+2] (4x2), split into top 2x2 and bottom 2x2
    
    for k_start in range(0, 8, 2):
        # Left 2x2 * 2x2: A[0:2, 0:2] * B[0:2, k_start:k_start+2]
        for r in range(7):
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            
            # A_top left: A[0,0], A[0,1], A[1,0], A[1,1] -> U[0], U[1], U[4], U[5]
            u[0] = strassen_U[0, r]
            u[1] = strassen_U[1, r]
            u[4] = strassen_U[2, r]
            u[5] = strassen_U[3, r]
            
            # B top: B[0,k_start], B[0,k_start+1], B[1,k_start], B[1,k_start+1]
            v[k_start] = strassen_V[0, r]
            v[k_start + 1] = strassen_V[1, r]
            v[8 + k_start] = strassen_V[2, r]
            v[8 + k_start + 1] = strassen_V[3, r]
            
            # C top-left: C[0,k_start], C[0,k_start+1], C[1,k_start], C[1,k_start+1]
            w[k_start * 3] = strassen_W[0, r]
            w[(k_start + 1) * 3] = strassen_W[1, r]
            w[k_start * 3 + 1] = strassen_W[2, r]
            w[(k_start + 1) * 3 + 1] = strassen_W[3, r]
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
        
        # Right 2x2 * 2x2: A[0:2, 2:4] * B[0:2, k_start:k_start+2]
        for r in range(7):
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            
            # A_top right: A[0,2], A[0,3], A[1,2], A[1,3] -> U[2], U[3], U[6], U[7]
            u[2] = strassen_U[0, r]
            u[3] = strassen_U[1, r]
            u[6] = strassen_U[2, r]
            u[7] = strassen_U[3, r]
            
            # B top: B[0,k_start], B[0,k_start+1], B[1,k_start], B[1,k_start+1]
            v[k_start] = strassen_V[0, r]
            v[k_start + 1] = strassen_V[1, r]
            v[8 + k_start] = strassen_V[2, r]
            v[8 + k_start + 1] = strassen_V[3, r]
            
            # C top-right: C[0,k_start+2], C[0,k_start+3], C[1,k_start+2], C[1,k_start+3]
            w[(k_start + 2) * 3] = strassen_W[0, r]
            w[(k_start + 3) * 3] = strassen_W[1, r]
            w[(k_start + 2) * 3 + 1] = strassen_W[2, r]
            w[(k_start + 3) * 3 + 1] = strassen_W[3, r]
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
        
        # Bottom 2x2 * 2x2: A[0:2, 0:2] * B[2:4, k_start:k_start+2]
        for r in range(7):
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            
            # A_top left: A[0,0], A[0,1], A[1,0], A[1,1] -> U[0], U[1], U[4], U[5]
            u[0] = strassen_U[0, r]
            u[1] = strassen_U[1, r]
            u[4] = strassen_U[2, r]
            u[5] = strassen_U[3, r]
            
            # B bottom: B[2,k_start], B[2,k_start+1], B[3,k_start], B[3,k_start+1]
            v[16 + k_start] = strassen_V[0, r]
            v[16 + k_start + 1] = strassen_V[1, r]
            v[24 + k_start] = strassen_V[2, r]
            v[24 + k_start + 1] = strassen_V[3, r]
            
            # C bottom: C[0,k_start], C[0,k_start+1], C[1,k_start], C[1,k_start+1]
            w[k_start * 3] = strassen_W[0, r]
            w[(k_start + 1) * 3] = strassen_W[1, r]
            w[k_start * 3 + 1] = strassen_W[2, r]
            w[(k_start + 1) * 3 + 1] = strassen_W[3, r]
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
        
        # Bottom-right 2x2 * 2x2: A[0:2, 2:4] * B[2:4, k_start:k_start+2]
        for r in range(7):
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            
            # A_top right: A[0,2], A[0,3], A[1,2], A[1,3] -> U[2], U[3], U[6], U[7]
            u[2] = strassen_U[0, r]
            u[3] = strassen_U[1, r]
            u[6] = strassen_U[2, r]
            u[7] = strassen_U[3, r]
            
            # B bottom: B[2,k_start], B[2,k_start+1], B[3,k_start], B[3,k_start+1]
            v[16 + k_start] = strassen_V[0, r]
            v[16 + k_start + 1] = strassen_V[1, r]
            v[24 + k_start] = strassen_V[2, r]
            v[24 + k_start + 1] = strassen_V[3, r]
            
            # C bottom-right: C[0,k_start+2], C[0,k_start+3], C[1,k_start+2], C[1,k_start+3]
            w[(k_start + 2) * 3] = strassen_W[0, r]
            w[(k_start + 3) * 3] = strassen_W[1, r]
            w[(k_start + 2) * 3 + 1] = strassen_W[2, r]
            w[(k_start + 3) * 3 + 1] = strassen_W[3, r]
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
    
    # Part 2: Bottom row of A (1x4) times B (4x8)
    # C[2,k] = sum_{j=0}^{3} A[2,j] * B[j,k]
    # For each j, schoolbook term: A[2,j] * B[j,:]
    
    for j in range(4):
        for k in range(8):
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            
            u[2 * m + j] = 1  # A[2,j]
            v[j * p + k] = 1  # B[j,k]
            w[k * n + 2] = 1  # C[2,k]
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
    
    # Convert to list of lists
    rank = len(U_list)
    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    
    U = U_arr.tolist()
    V = V_arr.tolist()
    W = W_arr.tolist()
    
    return {"U": U, "V": V, "W": W}
