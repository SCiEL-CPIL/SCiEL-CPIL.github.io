import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Rank 56 decomposition using 8 independent Strassen 2x2x2 blocks.
    # Structure:
    # A (2x4) split into A_left (2x2), A_right (2x2)
    # B (4x8) split into B_top (2x8), B_bottom (2x8)
    # B_top split into 4 blocks of 2x2
    # B_bottom split into 4 blocks of 2x2
    # Total 8 blocks of 2x2x2.
    # Each 2x2x2 has rank 7.
    # Total rank 56.
    
    # Strassen 2x2x2 components (4x7 matrices)
    # U_s, V_s, W_s
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11)
    U_s[0,0]=1; U_s[3,0]=1
    V_s[0,0]=1; V_s[3,0]=1
    W_s[0,0]=1; W_s[3,0]=1
    # M2: (A10+A11)B00
    U_s[2,1]=1; U_s[3,1]=1
    V_s[0,1]=1
    W_s[2,1]=1; W_s[3,1]=-1
    # M3: A00(B10-B11)
    U_s[0,2]=1
    V_s[1,2]=1; V_s[3,2]=-1
    W_s[1,2]=1; W_s[3,2]=1
    # M4: A11(B01-B00)
    U_s[3,3]=1
    V_s[2,3]=1; V_s[0,3]=-1
    W_s[0,3]=1; W_s[2,3]=1
    # M5: (A00+A01)B11
    U_s[0,4]=1; U_s[1,4]=1
    V_s[3,4]=1
    W_s[0,4]=-1; W_s[1,4]=1
    # M6: (A10-A00)(B00+B01)
    U_s[2,5]=1; U_s[0,5]=-1
    V_s[0,5]=1; V_s[1,5]=1
    W_s[3,5]=1
    # M7: (A01-A11)(B10+B11)
    U_s[1,6]=1; U_s[3,6]=-1
    V_s[2,6]=1; V_s[3,6]=1
    W_s[0,6]=1

    R = 56
    U = np.zeros((n*m, R), dtype=int)
    V = np.zeros((m*p, R), dtype=int)
    W = np.zeros((p*n, R), dtype=int)
    
    # Iterate over the 8 blocks
    # s=0: A_left (cols 0,1), B_top (rows 0,1)
    # s=1: A_right (cols 2,3), B_bottom (rows 2,3)
    # k_pair: 0..3 for the 4 column pairs in B
    
    for s in range(2):
        for k_pair in range(4):
            base_r = (s * 4 + k_pair) * 7
            
            # A indices (flattened row-major: i*4+j)
            if s == 0:
                # A[0,0], A[0,1], A[1,0], A[1,1] -> 0, 1, 4, 5
                a_idx = [0, 1, 4, 5]
            else:
                # A[0,2], A[0,3], A[1,2], A[1,3] -> 2, 3, 6, 7
                a_idx = [2, 3, 6, 7]
            
            # B indices (flattened row-major: j*8+k)
            if s == 0:
                # Rows 0,1. Cols 2*k_pair, 2*k_pair+1
                # B[0, 2k], B[0, 2k+1], B[1, 2k], B[1, 2k+1]
                b_idx = [0*8 + 2*k_pair, 0*8 + 2*k_pair + 1, 1*8 + 2*k_pair, 1*8 + 2*k_pair + 1]
            else:
                # Rows 2,3. Cols 2*k_pair, 2*k_pair+1
                # B[2, 2k], B[2, 2k+1], B[3, 2k], B[3, 2k+1]
                b_idx = [2*8 + 2*k_pair, 2*8 + 2*k_pair + 1, 3*8 + 2*k_pair, 3*8 + 2*k_pair + 1]
            
            # C indices (Trace convention: k*n+i = 2k+i)
            # Block output is 2x2.
            # Local 0: C(0, 2k_pair) -> 2*(2k_pair) + 0 = 4*k_pair
            # Local 1: C(0, 2k_pair+1) -> 2*(2k_pair+1) + 0 = 4*k_pair + 2
            # Local 2: C(1, 2k_pair) -> 2*(2k_pair) + 1 = 4*k_pair + 1
            # Local 3: C(1, 2k_pair+1) -> 2*(2k_pair+1) + 1 = 4*k_pair + 3
            c_idx = [4*k_pair, 4*k_pair + 2, 4*k_pair + 1, 4*k_pair + 3]
            
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
                        
    if time.time() > deadline:
        # This should never happen as the loop is very fast, but just in case
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
