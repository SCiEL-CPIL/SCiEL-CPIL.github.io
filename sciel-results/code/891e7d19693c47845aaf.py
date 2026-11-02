import numpy as np
import time

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 14 (Two independent Strassen 2x2x2)
    # If 14 fails, fallback to 16.
    
    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [i, j]]
    # U vectors (for A)
    U_strassen = [
        [1, 0, 0, 1], # a+d
        [0, 0, 1, 1], # c+d
        [1, 0, 0, 0], # a
        [0, 0, 0, 1], # d
        [1, 1, 0, 0], # a+b
        [-1, 0, 1, 0], # c-a
        [0, 1, 0, -1]  # b-d
    ]
    
    # V vectors (for B)
    V_strassen_left = [
        [1, 0, 1, 0], # e+i
        [1, 0, 0, 0], # e
        [0, 1, 0, -1], # f-i
        [0, -1, 1, 0], # i-e
        [0, 0, 1, 0], # i
        [1, 1, 0, 0], # e+f
        [0, 0, 1, 1]  # i+j
    ]
    
    # W vectors (for C)
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5 + p7
    # C10 = p2 + p4 - p6 + p7
    # C11 = p1 - p3 + p6 - p2
    W_strassen = [
        [1, 0, 0, 0], # p1 -> C00
        [0, 0, 1, 0], # p2 -> C10
        [0, 1, 0, 0], # p3 -> C01
        [1, 0, 1, 0], # p4 -> C00, C10
        [-1, 1, 0, 0], # p5 -> C00, C01
        [0, 0, -1, 1], # p6 -> C10, C11
        [1, 1, 1, 0]  # p7 -> C00, C01, C10
    ]
    # Note: The W vectors above are rows of the W matrix for the 2x2 case?
    # W shape is (p*n, rank). p=2, n=2.
    # W[k*n+i, r].
    # C00 is k=0, i=0 -> index 0.
    # C01 is k=1, i=0 -> index 1.
    # C10 is k=0, i=1 -> index 2.
    # C11 is k=1, i=1 -> index 3.
    
    # Let's re-verify W_strassen indices.
    # p1 contributes to C00. W[0, 0] = 1.
    # p2 contributes to C10. W[2, 1] = 1.
    # p3 contributes to C01. W[1, 2] = 1.
    # p4 contributes to C00, C10. W[0, 3]=1, W[2, 3]=1.
    # p5 contributes to C00, C01. W[0, 4]=-1, W[1, 4]=1.
    # p6 contributes to C10, C11. W[2, 5]=-1, W[3, 5]=1.
    # p7 contributes to C00, C01, C10. W[0, 6]=1, W[1, 6]=1, W[2, 6]=1.
    
    W_strassen_corrected = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 1],  # C01
        [0, 1, 0, 1, 0, -1, 1], # C10
        [0, 0, 0, 0, 0, 1, 0]   # C11
    ]
    # Transpose to get (rank, 4) then transpose back to (4, rank)?
    # W is (p*n, rank) = (4, 7).
    
    U_mat = np.array(U_strassen).T # (4, 7)
    V_mat_left = np.array(V_strassen_left).T # (4, 7)
    W_mat = np.array(W_strassen_corrected).T # (4, 7)
    
    # Build Rank 14 matrices for 2x2x4
    # U is (4, 14)
    # V is (8, 14)
    # W is (8, 14)
    
    U_14 = np.zeros((4, 14), dtype=int)
    V_14 = np.zeros((8, 14), dtype=int)
    W_14 = np.zeros((8, 14), dtype=int)
    
    # First 7 terms: Left block (cols 0,1)
    U_14[:, 0:7] = U_mat
    V_14[0:4, 0:7] = V_mat_left
    # V_14[4:8, 0:7] is 0
    
    # W for left block: C00, C01, C10, C11 correspond to k=0,1.
    # W index: k*n + i.
    # k=0: C00 (i=0) -> 0, C10 (i=1) -> 1.
    # k=1: C01 (i=0) -> 2, C11 (i=1) -> 3.
    # So W_14[0:4, 0:7] should be W_mat.
    W_14[0:4, 0:7] = W_mat
    # W_14[4:8, 0:7] is 0
    
    # Next 7 terms: Right block (cols 2,3)
    U_14[:, 7:14] = U_mat
    # V for right block: B cols 2,3.
    # V index: j*p + k.
    # j=0: k=2 -> 2, k=3 -> 3.
    # j=1: k=2 -> 6, k=3 -> 7.
    # V_mat_left is for B[[e,f],[i,j]].
    # We map e->g, f->h, i->k, j->l.
    # V_mat_left[0] (e) -> V[2]
    # V_mat_left[1] (f) -> V[3]
    # V_mat_left[2] (i) -> V[6]
    # V_mat_left[3] (j) -> V[7]
    
    V_right = np.zeros((8, 7), dtype=int)
    V_right[2, :] = V_mat_left[0, :]
    V_right[3, :] = V_mat_left[1, :]
    V_right[6, :] = V_mat_left[2, :]
    V_right[7, :] = V_mat_left[3, :]
    
    V_14[:, 7:14] = V_right
    
    # W for right block: C02, C03, C12, C13.
    # k=2: C02 (i=0) -> 2*2+0=4, C12 (i=1) -> 2*2+1=5.
    # k=3: C03 (i=0) -> 3*2+0=6, C13 (i=1) -> 3*2+1=7.
    W_14[4:8, 7:14] = W_mat
    
    # Verify decomposition
    # T_ijk = sum_r U_{im+j, r} V_{jp+k, r} W_{kn+i, r}
    # Check if U V W reconstruction matches identity tensor.
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(14):
                    val += U_14[i*m+j, r] * V_14[j*p+k, r] * W_14[k*n+i, r]
                expected = 1
                if val != expected:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        return {
            "U": U_14.tolist(),
            "V": V_14.tolist(),
            "W": W_14.tolist()
        }
    else:
        # Fallback to Schoolbook Rank 16
        rank = 16
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(8)]
        W = [[0] * rank for _ in range(8)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    U[i*2+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
