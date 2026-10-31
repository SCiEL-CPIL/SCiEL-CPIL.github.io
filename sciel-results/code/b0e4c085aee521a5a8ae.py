import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 4, 4, 4
    
    # Define 2x2 Strassen factors
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Rows of U2, V2 correspond to flattened A, B (row-major)
    # Rows of W2 correspond to flattened C (row-major) but with trace convention?
    # Task: "output factor follows the official trace convention W[k*n+i,r]"
    # For 2x2, n=2. W index is k*2+i.
    # C_ik = sum_r W[k*2+i, r] M_r
    # My W2 construction above used W2[k*2+i, r].
    # C_00 (k=0, i=0) -> idx 0
    # C_01 (k=1, i=0) -> idx 1
    # C_10 (k=0, i=1) -> idx 2
    # C_11 (k=1, i=1) -> idx 3
    # This matches the standard row-major flattening of C if we consider C as 2x2.
    # Wait, C_ik is element at row i, col k.
    # Row-major flattening of C: C_00, C_01, C_10, C_11.
    # Indices: 0, 1, 2, 3.
    # W[k*n+i] for n=2:
    # C_00: k=0, i=0 -> 0*2+0 = 0.
    # C_01: k=1, i=0 -> 1*2+0 = 2.
    # C_10: k=0, i=1 -> 0*2+1 = 1.
    # C_11: k=1, i=1 -> 1*2+1 = 3.
    # So the indices are:
    # C_00 -> 0
    # C_10 -> 1
    # C_01 -> 2
    # C_11 -> 3
    
    # My previous W2 mapping:
    # W2[0] -> C_00
    # W2[1] -> C_01 (I used idx 1 for C_01)
    # W2[2] -> C_10 (I used idx 2 for C_10)
    # W2[3] -> C_11
    
    # Let's re-evaluate the W2 indices based on "W[k*n+i]".
    # C_00: k=0, i=0 -> idx 0.
    # C_01: k=1, i=0 -> idx 2.
    # C_10: k=0, i=1 -> idx 1.
    # C_11: k=1, i=1 -> idx 3.
    
    # So:
    # W2[0] corresponds to C_00.
    # W2[1] corresponds to C_10.
    # W2[2] corresponds to C_01.
    # W2[3] corresponds to C_11.
    
    # Let's re-map the coefficients.
    # M1: C_00 += 1, C_11 += 1. -> W2[0, 0]=1, W2[3, 0]=1.
    # M2: C_10 += 1. -> W2[1, 1]=1. (Since C_10 is idx 1)
    # M3: C_01 += 1, C_11 -= 1. -> W2[2, 2]=1, W2[3, 2]=-1. (Since C_01 is idx 2)
    # M4: C_10 += 1, C_00 -= 1, C_11 -= 1. -> W2[1, 3]=1, W2[0, 3]=-1, W2[3, 3]=-1.
    # M5: C_01 += 1, C_11 += 1. -> W2[2, 4]=1, W2[3, 4]=1.
    # M6: C_00 += 1, C_10 += 1. -> W2[0, 5]=1, W2[1, 5]=1.
    # M7: C_00 -= 1, C_11 += 1. -> W2[0, 6]=-1, W2[3, 6]=1.
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1 # C_10
    
    # M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1 # C_01
    W2[3][2] = -1 # C_11
    
    # M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[1][3] = 1 # C_10
    W2[0][3] = -1 # C_00
    W2[3][3] = -1 # C_11
    
    # M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[2][4] = 1 # C_01
    W2[3][4] = 1 # C_11
    
    # M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[0][5] = 1 # C_00
    W2[1][5] = 1 # C_10
    
    # M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = -1 # C_00
    W2[3][6] = 1 # C_11

    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Kronecker Product Construction
    for r in range(7):
        for s in range(7):
            c = 7*r + s
            # U
            for i_r in range(4):
                if U2[i_r][r] == 0: continue
                for i_s in range(4):
                    if U2[i_s][s] == 0: continue
                    i = 4*i_r + i_s
                    U[i][c] += U2[i_r][r] * U2[i_s][s]
            # V
            for j_r in range(4):
                if V2[j_r][r] == 0: continue
                for j_s in range(4):
                    if V2[j_s][s] == 0: continue
                    j = 4*j_r + j_s
                    V[j][c] += V2[j_r][r] * V2[j_s][s]
            # W
            for k_r in range(4):
                if W2[k_r][r] == 0: continue
                for k_s in range(4):
                    if W2[k_s][s] == 0: continue
                    k = 4*k_r + k_s
                    W[k][c] += W2[k_r][r] * W2[k_s][s]
                    
    # Verification
    # C_ik = sum_r W[k*n+i, r] (sum_j U[i*n+j, r] A_ij) (sum_l V[j*n+l, r] B_jl)
    # Note: U index is i*n+j. V index is j*n+l. W index is k*n+i.
    # In our construction:
    # U row i (0..15) corresponds to i_r, i_s. i = 4 i_r + i_s.
    # This is row-major flattening of A (4x4). A[i] = A[i_r, i_s].
    # V row j (0..15) corresponds to j_r, j_s. j = 4 j_r + j_s.
    # This is row-major flattening of B (4x4). B[j] = B[j_r, j_s].
    # W row k (0..15) corresponds to k_r, k_s. k = 4 k_r + k_s.
    # This is row-major flattening of C (4x4). C[k] = C[k_r, k_s].
    # BUT the task requires W[k*n+i, r].
    # My W construction used W2 indices which were mapped to C_ik with trace convention.
    # Let's check if the Kronecker product of W2 (with trace indices) yields W (with trace indices).
    # W2 row index was k_r*2 + i_r? No, W2 row index was just 0..3 corresponding to C_00, C_10, C_01, C_11.
    # C_00 (k=0, i=0) -> idx 0.
    # C_10 (k=0, i=1) -> idx 1.
    # C_01 (k=1, i=0) -> idx 2.
    # C_11 (k=1, i=1) -> idx 3.
    # So W2 row index is k_r*2 + i_r.
    # In the Kronecker product, W row index is k*4 + i?
    # My loop: k = 4*k_r + k_s.
    # This assumes W row index is k (flattened C).
    # But task requires W[k*n+i].
    # If C is 4x4, n=4. W index is k*4 + i.
    # My loop sets W[k][c] where k = 4*k_r + k_s.
    # This k is the flattened index of C.
    # So W[flattened_C][c] is what I have.
    # The task expects W[k*n+i, r].
    # Flattened C index for C[k,i] is k*4 + i.
    # So my W matrix is already in the correct format if I interpret the row index as k*4+i.
    # Let's verify the indices.
    # C[k,i] is at row k*4+i.
    # My W row index is k (from 0..15).
    # So W[row, c] contributes to C[row].
    # C[row] is C[k,i] where row = 4k+i.
    # So yes, it matches.

    # Quick Check with Random Matrices
    if time.time() < deadline - 5.0:
        A = np.random.randint(0, 10, (4, 4))
        B = np.random.randint(0, 10, (4, 4))
        C_direct = A @ B
        
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # A_flat: 16x1
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        # M_r = sum_j U[j, r] A[j] * sum_l V[l, r] B[l]
        # U is 16x49. A_flat is 16.
        # U.T @ A_flat -> 49x1
        M_A = U_np.T @ A_flat
        M_B = V_np.T @ B_flat
        M = M_A * M_B
        
        # C_flat = sum_r W[k, r] M[r]
        # W is 16x49.
        # W @ M -> 16x1
        C_decomp_flat = W_np @ M
        C_decomp = C_decomp_flat.reshape(4, 4)
        
        if not np.allclose(C_direct, C_decomp):
            # Fallback to schoolbook if error (should not happen)
            rank = 64
            U = [[0]*rank for _ in range(16)]
            V = [[0]*rank for _ in range(16)]
            W = [[0]*rank for _ in range(16)]
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        r = i*16 + j*4 + k
                        U[i*4+j][r] = 1
                        V[j*4+k][r] = 1
                        W[k*4+i][r] = 1

    return {"U": U, "V": V, "W": W}
