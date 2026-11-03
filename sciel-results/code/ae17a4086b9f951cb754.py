import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    # Step 1: Construct rank-11 decomposition for 2x2 by 2x3
    # It is the direct sum of T_2,2,1 (rank 4) and T_2,2,2 (rank 7)
    
    # Part 1: T_2,2,1 (Rank 4)
    # A (2x2), b (2x1)
    # C = A b
    # C_0 = a*e + b*h
    # C_1 = c*e + d*h
    # Indices:
    # U: (i, j) -> i*m+j
    # V: (j, k) -> j*p+k (p=1)
    # W: (k, i) -> k*n+i (n=2)
    
    U1 = np.zeros((4, 4)) # 2*2
    V1 = np.zeros((2, 4)) # 2*1
    W1 = np.zeros((2, 4)) # 1*2
    
    # Term 1: a*e
    U1[0, 0] = 1 # i=0, j=0
    V1[0, 0] = 1 # j=0, k=0
    W1[0, 0] = 1 # k=0, i=0
    
    # Term 2: b*h
    U1[1, 1] = 1 # i=0, j=1
    V1[1, 1] = 1 # j=1, k=0
    W1[0, 1] = 1 # k=0, i=0
    
    # Term 3: c*e
    U1[2, 2] = 1 # i=1, j=0
    V1[0, 2] = 1 # j=0, k=0
    W1[1, 2] = 1 # k=0, i=1
    
    # Term 4: d*h
    U1[3, 3] = 1 # i=1, j=1
    V1[1, 3] = 1 # j=1, k=0
    W1[1, 3] = 1 # k=0, i=1
    
    # Part 2: T_2,2,2 (Rank 7) - Strassen
    # A (2x2), B (2x2)
    # We need to append these to U1, V1, W1
    # But V and W indices for the second part are shifted.
    # V1 corresponds to k=0.
    # The second part uses k=1,2 in the 2x3 context?
    # No, the second part is a 2x2 multiplication.
    # Let's define it separately and then combine.
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7)) # 2*2
    W2 = np.zeros((4, 7)) # 2*2
    
    # Strassen for A * B
    # P1 = (a+b)(e+i)
    # P2 = (c+d)(e+i)
    # P3 = a(g-i)
    # P4 = d(j-f)
    # P5 = (a+b)j
    # P6 = (c-a)(e+i)
    # P7 = (b-d)(g+i)
    
    # U indices: 0:00, 1:01, 2:10, 3:11
    # V indices: 0:00, 1:01, 2:10, 3:11
    # W indices: 0:00, 1:01, 2:10, 3:11
    
    # P1
    U2[0, 0] = 1; U2[1, 0] = 1
    V2[0, 0] = 1; V2[2, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C11 += P1, C22 += P1
    
    # P2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1; V2[2, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C21 += P2, C22 -= P2
    
    # P3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[2, 2] = 1 # C12 += P3, C21 += P3
    
    # P4
    U2[3, 3] = 1
    V2[3, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C11 += P4, C21 += P4
    
    # P5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C11 -= P5, C12 += P5
    
    # P6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[2, 5] = 1
    W2[1, 5] = 1; W2[2, 5] = 1 # C12 += P6, C21 += P6
    
    # P7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[1, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = -1 # C11 += P7, C22 -= P7
    
    # Combine U1, V1, W1 and U2, V2, W2 into U_2x2x3, V_2x2x3, W_2x2x3
    # U is same size (4 rows). Concatenate columns.
    U_small = np.hstack([U1, U2])
    
    # V: 
    # V1 is 2x4. Corresponds to k=0.
    # V2 is 4x7. Corresponds to k=0,1 of the 2x2 block.
    # In the 2x3 context, the 2x2 block uses the last two columns (k=1,2).
    # So we need to shift V2's k indices by 1.
    # V2 indices j*p+k. p=2 for the sub-block.
    # In the full 2x3, p=3.
    # Sub-block B' is columns 1 and 2 (0-indexed 1,2).
    # So k_sub=0 -> k_full=1. k_sub=1 -> k_full=2.
    # V2[j, k_sub] -> V_full[j, k_sub + 1]
    
    V_small = np.zeros((6, 11)) # 2*3 rows
    # Copy V1
    # V1[j, r] -> V_small[j*3 + 0, r]
    for j in range(2):
        for r in range(4):
            V_small[j*3 + 0, r] = V1[j, r]
            
    # Copy V2
    for j in range(2):
        for k_sub in range(2):
            for r in range(7):
                V_small[j*3 + (k_sub + 1), 4 + r] = V2[j*2 + k_sub, r]
                
    # W:
    # W1 is 2x4. k=0, i=0,1.
    # W2 is 4x7. k_sub=0,1, i=0,1.
    # W_full[k_full, i, r]
    # W1: k_full=0.
    # W2: k_full = k_sub + 1.
    
    W_small = np.zeros((6, 11)) # 3*2 rows
    # Copy W1
    # W1[k, i, r] -> W_small[0*2 + i, r]
    for k in range(1):
        for i in range(2):
            for r in range(4):
                W_small[k*2 + i, r] = W1[k, i, r]
                
    # Copy W2
    for k_sub in range(2):
        for i in range(2):
            for r in range(7):
                W_small[(k_sub + 1)*2 + i, 4 + r] = W2[k_sub*2 + i, r]
                
    # Now we have decomposition for 2x2 by 2x3 of rank 11.
    # U_small (4, 11), V_small (6, 11), W_small (6, 11)
    
    # Step 2: Kronecker product to get 4x4 by 4x9
    # U_total = U_small (Kron) U_small
    # V_total = V_small (Kron) V_small
    # W_total = W_small (Kron) W_small
    
    # Shapes:
    # U_small: 4x11. U_total: 16x121.
    # V_small: 6x11. V_total: 36x121.
    # W_small: 6x11. W_total: 36x121.
    
    U_total = np.kron(U_small, U_small)
    V_total = np.kron(V_small, V_small)
    W_total = np.kron(W_small, W_small)
    
    # Convert to lists of lists as required?
    # The interface expects {'U', 'V', 'W'} with integer/half-integer entries.
    # Numpy arrays are usually accepted if they are numeric.
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries".
    # It doesn't explicitly forbid numpy arrays, but lists are safer if "integer/half-integer" implies types.
    # However, usually numpy is fine.
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
