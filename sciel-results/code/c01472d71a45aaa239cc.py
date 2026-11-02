"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We use the decomposition T_{2,3,8} = T_{2,3,2} x T_{1,1,4}
    # Rank(T_{2,3,2}) = 11
    # Rank(T_{1,1,4}) = 4
    # Total Rank = 44
    
    # 1. Define T_{2,3,2} decomposition (11 terms)
    # A is 2x3, B is 3x2, C is 2x2
    # u: length 6 (i*3+j)
    # v: length 6 (j*2+k)
    # w: length 4 (k*2+i)
    
    terms_232 = []
    
    # Part 1: j=0 (index 0 in j) -> T_{2,1,2}
    # A indices: (0,0), (1,0) -> 0, 3
    # B indices: (0,0), (0,1) -> 0, 1
    # C indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    
    # Term 1: A[0,0] B[0,0] C[0,0]
    u1 = np.zeros(6); u1[0] = 1
    v1 = np.zeros(6); v1[0] = 1
    w1 = np.zeros(4); w1[0] = 1
    terms_232.append((u1, v1, w1))
    
    # Term 2: A[0,0] B[0,1] C[0,1]
    u2 = np.zeros(6); u2[0] = 1
    v2 = np.zeros(6); v2[1] = 1
    w2 = np.zeros(4); w2[1] = 1
    terms_232.append((u2, v2, w2))
    
    # Term 3: A[1,0] B[0,0] C[1,0]
    u3 = np.zeros(6); u3[3] = 1
    v3 = np.zeros(6); v3[0] = 1
    w3 = np.zeros(4); w3[2] = 1
    terms_232.append((u3, v3, w3))
    
    # Term 4: A[1,0] B[0,1] C[1,1]
    u4 = np.zeros(6); u4[3] = 1
    v4 = np.zeros(6); v4[1] = 1
    w4 = np.zeros(4); w4[3] = 1
    terms_232.append((u4, v4, w4))
    
    # Part 2: j=1,2 -> T_{2,2,2} Strassen
    # A indices: (0,1), (0,2), (1,1), (1,2) -> 1, 2, 4, 5
    # B indices: (1,0), (1,1), (2,0), (2,1) -> 2, 3, 4, 5
    # C indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    
    # Strassen products for A_sub (2x2) and B_sub (2x2)
    # A_sub = [[A[0,1], A[0,2]], [A[1,1], A[1,2]]]
    # B_sub = [[B[1,0], B[1,1]], [B[2,0], B[2,1]]]
    
    # P1 = (A00+A11)(B00+B11)
    u_p1 = np.zeros(6); u_p1[1] = 1; u_p1[5] = 1
    v_p1 = np.zeros(6); v_p1[2] = 1; v_p1[5] = 1
    w_p1 = np.zeros(4); w_p1[0] = 1; w_p1[3] = 1 # C00 += P1, C11 += P1
    terms_232.append((u_p1, v_p1, w_p1))
    
    # P2 = (A10+A11)B00
    u_p2 = np.zeros(6); u_p2[4] = 1; u_p2[5] = 1
    v_p2 = np.zeros(6); v_p2[2] = 1
    w_p2 = np.zeros(4); w_p2[2] = 1; w_p2[3] = -1 # C10 += P2, C11 -= P2
    terms_232.append((u_p2, v_p2, w_p2))
    
    # P3 = A00(B10-B00)
    u_p3 = np.zeros(6); u_p3[1] = 1
    v_p3 = np.zeros(6); v_p3[5] = 1; v_p3[2] = -1
    w_p3 = np.zeros(4); w_p3[1] = 1; w_p3[3] = 1 # C01 += P3, C11 += P3
    terms_232.append((u_p3, v_p3, w_p3))
    
    # P4 = A11(B01-B10)
    u_p4 = np.zeros(6); u_p4[5] = 1
    v_p4 = np.zeros(6); v_p4[3] = 1; v_p4[4] = -1
    w_p4 = np.zeros(4); w_p4[0] = 1; w_p4[2] = 1 # C00 += P4, C10 += P4
    terms_232.append((u_p4, v_p4, w_p4))
    
    # P5 = (A00+A01)B11
    u_p5 = np.zeros(6); u_p5[1] = 1; u_p5[2] = 1
    v_p5 = np.zeros(6); v_p5[3] = 1
    w_p5 = np.zeros(4); w_p5[1] = 1; w_p5[0] = -1 # C01 += P5, C00 -= P5
    terms_232.append((u_p5, v_p5, w_p5))
    
    # P6 = (A10-A00)(B00+B01)
    u_p6 = np.zeros(6); u_p6[4] = 1; u_p6[1] = -1
    v_p6 = np.zeros(6); v_p6[2] = 1; v_p6[3] = 1
    w_p6 = np.zeros(4); w_p6[3] = 1 # C11 += P6
    terms_232.append((u_p6, v_p6, w_p6))
    
    # P7 = (A01-A11)(B10+B11)
    u_p7 = np.zeros(6); u_p7[2] = 1; u_p7[5] = -1
    v_p7 = np.zeros(6); v_p7[4] = 1; v_p7[5] = 1
    w_p7 = np.zeros(4); w_p7[0] = 1 # C00 += P7
    terms_232.append((u_p7, v_p7, w_p7))
    
    # 2. Define T_{1,1,4} decomposition (4 terms)
    # A is 1x1, B is 1x4, C is 1x4
    # u: length 1
    # v: length 4
    # w: length 4
    
    terms_114 = []
    for k in range(4):
        u_t = np.zeros(1); u_t[0] = 1
        v_t = np.zeros(4); v_t[k] = 1
        w_t = np.zeros(4); w_t[k] = 1
        terms_114.append((u_t, v_t, w_t))
        
    # 3. Combine
    rank = len(terms_232) * len(terms_114)
    U = np.zeros((6, rank))
    V = np.zeros((24, rank))
    W = np.zeros((16, rank))
    
    col = 0
    for u1, v1, w1 in terms_232:
        for u2, v2, w2 in terms_114:
            # U: Kronecker product of u1 (len 6) and u2 (len 1)
            U[:, col] = np.kron(u1, u2)
            
            # V: Kronecker product of v1 (len 6) and v2 (len 4)
            V[:, col] = np.kron(v1, v2)
            
            # W: Custom permutation
            # w1 indices: k' * 2 + i
            # w2 indices: k''
            # Target index: (k' * 4 + k'') * 2 + i
            W_vec = np.zeros(16)
            for idx1 in range(4):
                if w1[idx1] != 0:
                    k_prime = idx1 // 2
                    i = idx1 % 2
                    for idx2 in range(4):
                        if w2[idx2] != 0:
                            k_double_prime = idx2
                            target_idx = (k_prime * 4 + k_double_prime) * 2 + i
                            W_vec[target_idx] += w1[idx1] * w2[idx2]
            W[:, col] = W_vec
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
