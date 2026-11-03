import time
import sys
import numpy as np
import scipy.linalg

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8

    # Strategy: Split into 2 independent (2x2) x (2x4) multiplications
    # For each row i of A (2 rows), compute A[i,:] * B (2x4 * 4x8)
    # This is equivalent to 2x2 * 2x4 for each half of B's columns
    # 
    # Actually, let's think more carefully:
    # A is 2x4, B is 4x8, C is 2x8
    # Split A into A = [A0 | A1] where A0, A1 are 2x2
    # Split B into B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = A0*B00 + A1*B10 for columns 0-1, etc.
    #
    # Better: Use the fact that 2x4 * 4x8 = (2x2 * 2x4) applied twice
    # Split B into two 2x4 blocks: B = [B_left | B_right], each 2x4
    # Then C_left = A * B_left (2x4 * 4x4 -> 2x4)
    # C_right = A * B_right (2x4 * 4x4 -> 2x4)
    #
    # Even better: 2x4 * 4x4 can be decomposed as 2x2 * 2x2 applied 2 times
    # 
    # Let me use the Kronecker product approach:
    # 2x2 * 2x4 multiplication has rank 14 (Strassen rank 7 for 2x2*2x2, 
    # but 2x2*2x4 is different)
    #
    # Actually, the cleanest approach:
    # 2x4 * 4x8: Think of this as computing 8 output elements per row.
    # 
    # Use the fact that matrix multiplication is bilinear and decomposable.
    # 
    # Key insight: 2x4 * 4x8 can be seen as 2 independent 2x4 * 4x4 multiplications
    # (splitting B's 8 columns into two groups of 4)
    # Each 2x4 * 4x4 has rank at most 16 (schoolbook) or less.
    #
    # Even better: 2x4 * 4x4 = (2x2 * 2x2) * 2 (Kronecker structure)
    # No, that's not quite right either.
    #
    # Let me use the direct approach with Strassen on blocks:
    # 
    # A (2x4) = [A00 A01; A10 A11] where each is 2x2... no, A is 2x4
    # A = [a0 a1 a2 a3; a4 a5 a6 a7]
    # 
    # Split into: A_left = [a0 a1; a4 a5] (2x2), A_right = [a2 a3; a6 a7] (2x2)
    # B (4x8) = [B_top; B_bottom] where B_top is 2x8, B_bottom is 2x8
    # Split B_top into B_top_left (2x4) and B_top_right (2x4)
    # Split B_bottom into B_bot_left (2x4) and B_bot_right (2x4)
    #
    # C = A_left * B_top_left + A_right * B_bot_left for left 4 cols
    # C = A_left * B_top_right + A_right * B_bot_right for right 4 cols
    #
    # This doesn't directly help with rank reduction.
    #
    # Let me try the simplest correct approach: schoolbook with rank 64,
    # then try to reduce using linear dependencies.
    #
    # Actually, the best known approach for 2x4 * 4x8:
    # Use 2x2 Strassen blocks. Split the 4x8 into four 2x2 blocks.
    # A (2x4) = [A0 | A1] where A0, A1 are 2x2
    # B (4x8) = [[B00, B01, B02, B03], [B10, B11, B12, B13]] each Bij is 2x2
    # 
    # C (2x8) = A0*B00 + A1*B10 for cols 0-1
    #          A0*B01 + A1*B11 for cols 2-3
    #          A0*B02 + A1*B12 for cols 4-5
    #          A0*B03 + A1*B13 for cols 6-7
    #
    # Each 2x2*2x2 product can use Strassen (rank 7), but we share A0 and A1.
    # 
    # For each of the 4 column-pairs, we need A0*Bjk + A1*B(j+1)k
    # This is a 2x2 * 4x2 -> 2x2 multiplication... 
    #
    # Let me just use the rank-56 approach from the seed and try to find
    # linear dependencies to reduce it.
    #
    # Strassen decomposition for 2x2 (rank 7):
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]
    
    # r=0 (M1): (a00+a11)(b00+b11)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1  # C00
    W_local[3][0] = 1  # C11
    
    # r=1 (M2): (a10+a11)b00
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1   # C10
    W_local[3][1] = -1  # C11
    
    # r=2 (M3): a00(b01-b11)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1   # C01
    W_local[3][2] = 1   # C11
    
    # r=3 (M4): a11(b10-b00)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1   # C00
    W_local[2][3] = 1   # C10
    
    # r=4 (M5): (a00+a01)b11
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1  # C00
    W_local[1][4] = 1   # C01
    
    # r=5 (M6): (a10-a00)(b00+b01)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1   # C11
    
    # r=6 (M7): (a01-a11)(b10+b11)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1   # C00

    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    # Decompose 2x4 x 4x8 into 8 blocks of 2x2 x 2x2
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            a_map = [
                0 * 4 + (2 * s + 0),
                0 * 4 + (2 * s + 1),
                1 * 4 + (2 * s + 0),
                1 * 4 + (2 * s + 1),
            ]
            
            b_map = [
                (2 * s + 0) * 8 + (2 * j + 0),
                (2 * s + 0) * 8 + (2 * j + 1),
                (2 * s + 1) * 8 + (2 * j + 0),
                (2 * s + 1) * 8 + (2 * j + 1),
            ]
            
            c_map = [
                (2 * j + 0) * 2 + 0,
                (2 * j + 1) * 2 + 0,
                (2 * j + 0) * 2 + 1,
                (2 * j + 1) * 2 + 1,
            ]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U[a_map[la]][r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V[b_map[lb]][r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W[c_map[lc]][r] += val
                        
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build tensor matrix: M[i*m*p + j*p + k, r] = U[i,r] * V[j,r] * W[k,r]
    M = np.zeros((n * m * p, total_rank))
    for r in range(total_rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i * m * p + j * p + k, r] = U_np[i, r] * V_np[j, r] * W_np[k, r]
    
    # Find null space to reduce rank
    try:
        Q, R, P = scipy.linalg.qr(M, pivoting=True)
        
        tol = 1e-10
        diag_R = np.abs(np.diag(R))
        
        rank_m = 0
        for i in range(len(diag_R)):
            if diag_R[i] > tol:
                rank_m = i + 1
            else:
                break
        
        if rank_m < total_rank:
            keep_indices = P[:rank_m]
            
            U_new = U_np[:, keep_indices]
            V_new = V_np[:, keep_indices]
            W_new = W_np[:, keep_indices]
            
            new_rank = len(keep_indices)
            
            # Verify correctness
            tensor_recon = np.zeros((n, m, p))
            for r in range(new_rank):
                u_r = U_new[:, r]
                v_r = V_new[:, r]
                w_r = W_new[:, r]
                tensor_recon += np.einsum('i,j,k->ijk', u_r, v_r, w_r)
            
            max_err = np.max(np.abs(tensor_recon - 1.0))
            
            if max_err < 1e-8:
                U = U_new.tolist()
                V = V_new.tolist()
                W = W_new.tolist()
    except Exception:
        pass

    if time.time() - start_time > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
