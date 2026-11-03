import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Rank 56 decomposition
    # Structure: 4 blocks of 2x2x2 Strassen multiplications
    # Each block handles 2 rows of A, 2 cols of A, 2 rows of B, 2 cols of B?
    # No, the previous logic was:
    # j in 0..3 (col of A / row of B pair)
    # s in 0..1 (row of A pair? No, s was row group of A?)
    
    # Let's re-verify the mapping from the previous code.
    # U: (8, 56)
    # V: (32, 56)
    # W: (16, 56)
    
    # Strassen 2x2x2
    # U_l, V_l, W_l are 4x7 matrices.
    
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)
    
    # M1
    U_s[0,0]=1; U_s[3,0]=1
    V_s[0,0]=1; V_s[3,0]=1
    W_s[0,0]=1; W_s[3,0]=1
    # M2
    U_s[2,1]=1; U_s[3,1]=1
    V_s[0,1]=1
    W_s[2,1]=1; W_s[3,1]=-1
    # M3
    U_s[0,2]=1
    V_s[1,2]=1; V_s[3,2]=-1
    W_s[1,2]=1; W_s[3,2]=1
    # M4
    U_s[3,3]=1
    V_s[2,3]=1; V_s[0,3]=-1
    W_s[0,3]=1; W_s[2,3]=1
    # M5
    U_s[0,4]=1; U_s[1,4]=1
    V_s[3,4]=1
    W_s[0,4]=-1; W_s[1,4]=1
    # M6
    U_s[2,5]=1; U_s[0,5]=-1
    V_s[0,5]=1; V_s[1,5]=1
    W_s[3,5]=1
    # M7
    U_s[1,6]=1; U_s[3,6]=-1
    V_s[2,6]=1; V_s[3,6]=1
    W_s[0,6]=1
    
    R = 56
    U = np.zeros((n*m, R), dtype=int)
    V = np.zeros((m*p, R), dtype=int)
    W = np.zeros((p*n, R), dtype=int)
    
    # Loop over 4 column-pairs of A (j=0,1,2,3)
    # and 2 row-pairs of A? No, A is 2x4.
    # The previous code iterated j in 0..3 and s in 0..1.
    # s=0: A cols 0,1. B rows 0,1.
    # s=1: A cols 2,3. B rows 2,3.
    # Wait, the previous code had:
    # if s==0: a_map = [0,1,4,5] (A[0,0], A[0,1], A[1,0], A[1,1])
    # if s==1: a_map = [2,3,6,7] (A[0,2], A[0,3], A[1,2], A[1,3])
    # This covers all of A.
    # For B:
    # if s==0: b_map = [0,1,8,9] for j=0?
    # b_map = [0*8+2*j, 0*8+2*j+1, 1*8+2*j, 1*8+2*j+1]
    # For j=0: B[0,0], B[0,1], B[1,0], B[1,1].
    # For j=1: B[0,2], B[0,3], B[1,2], B[1,3].
    # This covers B rows 0,1.
    # For s==1:
    # b_map = [2*8+2*j, ...]
    # For j=0: B[2,0], B[2,1], B[3,0], B[3,1].
    # This covers B rows 2,3.
    
    # So for each j (0..3), we have 2 Strassen blocks (s=0,1).
    # Total 8 blocks.
    # Each block rank 7. Total 56.
    
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            # A indices
            if s == 0:
                a_idx = [0, 1, 4, 5]
            else:
                a_idx = [2, 3, 6, 7]
            
            # B indices
            if s == 0:
                b_idx = [0*8 + 2*j, 0*8 + 2*j + 1, 1*8 + 2*j, 1*8 + 2*j + 1]
            else:
                b_idx = [2*8 + 2*j, 2*8 + 2*j + 1, 3*8 + 2*j, 3*8 + 2*j + 1]
            
            # C indices
            # C[i, k] index = k*2 + i
            # k = 2*j, 2*j+1
            # i = 0, 1
            # c_idx order: [C(0, 2j), C(1, 2j), C(0, 2j+1), C(1, 2j+1)]?
            # Previous c_map: [4*j + 0, 4*j + 2, 4*j + 1, 4*j + 3]
            # 4*j = 2*(2*j).
            # C(0, 2j) = (2j)*2 + 0 = 4j.
            # C(1, 2j) = (2j)*2 + 1 = 4j+1.
            # C(0, 2j+1) = (2j+1)*2 + 0 = 4j+2.
            # C(1, 2j+1) = (2j+1)*2 + 1 = 4j+3.
            # Previous map: [4j, 4j+2, 4j+1, 4j+3].
            # This corresponds to:
            # 0: C(0, 2j)
            # 1: C(0, 2j+1)
            # 2: C(1, 2j)
            # 3: C(1, 2j+1)
            # Wait, Strassen W_local indices:
            # 0: C00
            # 1: C01
            # 2: C10
            # 3: C11
            # So c_map should be [C(0,2j), C(0,2j+1), C(1,2j), C(1,2j+1)]
            # Which is [4j, 4j+2, 4j+1, 4j+3]. Correct.
            
            c_idx = [4*j, 4*j+2, 4*j+1, 4*j+3]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_s[la, l]
                    if val != 0:
                        U[a_idx[la], r] += val
                for lb in range(4):
                    val = V_s[lb, l]
                    if val != 0:
                        V[b_idx[lb], r] += val
                for lc in range(4):
                    val = W_s[lc, l]
                    if val != 0:
                        W[c_idx[lc], r] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
