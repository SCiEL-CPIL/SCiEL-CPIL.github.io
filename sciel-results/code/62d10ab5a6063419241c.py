import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    start_time = time.time()
    deadline = start_time + 110

    def verify_solution(U, V, W, r, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        T = np.einsum('ijt,jkt,kit->ijk', U_3d, V_3d, W_3d)
        target = np.ones((n, m, p))
        err = T - target
        if np.max(np.abs(err)) < 1e-8:
            # Check integer/half-integer
            if np.all(np.abs(2*U - np.round(2*U)) < 1e-9) and \
               np.all(np.abs(2*V - np.round(2*V)) < 1e-9) and \
               np.all(np.abs(2*W - np.round(2*W)) < 1e-9):
                return True
        return False

    # Rank-32 structured rectangular Strassen decomposition
    # Decompose A (3x3) into:
    #   A_top = A[0:2, :] (2x3) - rows 0,1
    #   A_bot = A[2, :] (1x3) - row 2
    # Decompose B (3x8) into 4 blocks of width 2:
    #   B_l = B[:, 2l:2l+2] for l=0,1,2,3
    #
    # For each block l, compute C_l = A * B_l (3x2)
    # C_l[0:2, :] = A_top * B_l (2x3 * 3x2)
    # C_l[2, :] = A_bot * B_l (1x3 * 3x2)
    #
    # For A_top * B_l (2x3 * 3x2):
    #   Decompose A_top into A_tl (2x2) and A_tr (2x1)
    #   Decompose B_l into B_lt (2x2) and B_lb (1x2)
    #   A_top * B_l = A_tl * B_lt + A_tr * B_lb
    #   A_tl * B_lt: 2x2*2x2 -> rank 7 (Strassen)
    #   A_tr * B_lb: 2x1*1x2 -> rank 4 (outer product, but we can do better)
    #   Actually 2x1*1x2 = (a0,b0) * (c0,d0; c1,d1) which is rank 4
    #   Wait: A_tr is 2x1, B_lb is 1x2, so A_tr * B_lb is a 2x2 outer product = rank 4
    #   Total for A_top * B_l: rank 7 + 4 = 11? That's too high.
    #
    # Let me reconsider. The total rank should be 32.
    # 4 blocks * 8 per block = 32.
    #
    # For each 3x3 * 3x2 multiplication:
    # We need rank 8 per block.
    #
    # Alternative: Use the fact that 3x3*3x2 can be done in rank 8.
    # This is a known result. Let me construct it.
    #
    # Actually, let me use a different decomposition:
    # For 3x3 * 3x2:
    # Split A into A1 (2x3) and A2 (1x3)
    # Split B into B1 (2x2) and B2 (1x2)
    # A*B = A1*B1 + A1*B2 + A2*B1 + A2*B2
    # A1*B1: 2x3*2x2... wait, dimensions don't match for direct Strassen.
    #
    # Let me use a cleaner approach:
    # 3x3 * 3x2:
    # Write A as [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]
    # Write B as [[B11, B12], [B21, B22], [B31, B32]]
    #
    # C11 = A11*B11 + A12*B21 + A13*B31
    # C12 = A11*B12 + A12*B22 + A13*B32
    # C21 = A21*B11 + A22*B21 + A23*B31
    # C22 = A21*B12 + A22*B22 + A23*B32
    # C31 = A31*B11 + A32*B21 + A33*B31
    # C32 = A31*B12 + A32*B22 + A33*B32
    #
    # Use Strassen on the 2x2 subproblem (rows 0,1 of A, cols 0,1 of B):
    # M1 = (A11+A21)(B11+B21)
    # M2 = A21*(B21-B11)
    # M3 = (A11+A12)(B31)  -- no, this doesn't work directly
    #
    # Let me just construct a rank-8 decomposition for 3x3*3x2 directly.
    # 
    # Known: 3x3*3x3 has rank 19 (Strassen variant)
    # 3x3*3x2 should have rank <= 13 or so.
    #
    # Let me try a different approach: use the existing rank-10 local decomposition
    # but optimize it.
    #
    # Actually, let me just verify the current rank-40 solution works and see if we
    # can find a better one through optimization.

    # Let me try to construct a rank-32 solution using block structure.
    # For each pair of columns (2l, 2l+1), we compute A * B[:, 2l:2l+2]
    # This is a 3x3 * 3x2 multiplication.
    
    # Rank-8 decomposition for 3x3 * 3x2:
    # U: 3x3x8, V: 3x2x8, W: 2x3x8
    
    U_33x32 = np.zeros((3, 3, 8))
    V_33x32 = np.zeros((3, 2, 8))
    W_33x32 = np.zeros((2, 3, 8))
    
    # t=0: (A00+A10)(B00+B10)
    U_33x32[0,0,0] = 1; U_33x32[1,0,0] = 1
    V_33x32[0,0,0] = 1; V_33x32[1,0,0] = 1
    W_33x32[0,0,0] = 1; W_33x32[0,1,0] = 1; W_33x32[1,0,0] = 1; W_33x32[1,1,0] = 1
    
    # t=1: A10*(B10-B00)
    U_33x32[1,0,1] = 1
    V_33x32[1,0,1] = 1; V_33x32[0,0,1] = -1
    W_33x32[1,0,1] = 1; W_33x32[1,1,1] = 1
    
    # t=2: (A00+A01)(B20)
    U_33x32[0,0,2] = 1; U_33x32[0,1,2] = 1
    V_33x32[2,0,2] = 1
    W_33x32[0,0,2] = 1; W_33x32[1,0,2] = 1
    
    # t=3: A01*(B21-B20)
    U_33x32[0,1,3] = 1
    V_33x32[2,1,3] = 1; V_33x32[2,0,3] = -1
    W_33x32[0,1,3] = 1; W_33x32[1,1,3] = 1
    
    # t=4: (A10+A11)(B21)
    U_33x32[1,0,4] = 1; U_33x32[1,1,4] = 1
    V_33x32[2,1,4] = 1
    W_33x32[0,1,4] = 1; W_33x32[1,1,4] = 1
    
    # t=5: A20*(B00+B10)
    U_33x32[2,0,5] = 1
    V_33x32[0,0,5] = 1; V_33x32[1,0,5] = 1
    W_33x32[0,0,5] = 1; W_33x32[1,0,5] = 1
    
    # t=6: A21*(B01+B11)
    U_33x32[2,1,6] = 1
    V_33x32[0,1,6] = 1; V_33x32[1,1,6] = 1
    W_33x32[0,1,6] = 1; W_33x32[1,1,6] = 1
    
    # t=7: A22*(B00+B10) + ... hmm, this is getting complicated.
    # Let me just use a simpler approach: rank 9 for 3x3*3x2
    # and total rank 36. Or try to find a working rank-8.
    
    # Actually, let me just verify the current solution and return it if valid.
    # The current rank-40 solution should work.
    
    # Reconstruct the rank-40 solution from the current code
    U_local = np.zeros((3, 3, 10))
    V_local = np.zeros((3, 2, 10))
    W_local = np.zeros((2, 3, 10))
    
    # Part 1: Strassen for 2x2x2 (i,j in {0,1}, k in {0,1})
    U_local[0,0,0] = 1; U_local[1,1,0] = 1
    V_local[0,0,0] = 1; V_local[1,1,0] = 1
    W_local[0,0,0] = 1; W_local[1,1,0] = 1
    
    U_local[1,0,1] = 1; U_local[1,1,1] = 1
    V_local[0,0,1] = 1
    W_local[1,0,1] = 1
    
    U_local[0,0,2] = 1
    V_local[1,0,2] = 1; V_local[1,1,2] = -1
    W_local[0,1,2] = 1; W_local[1,1,2] = 1
    
    U_local[1,1,3] = 1
    V_local[0,0,3] = 1; V_local[0,1,3] = -1
    W_local[0,0,3] = 1; W_local[1,0,3] = 1
    
    U_local[0,0,4] = 1; U_local[1,0,4] = 1
    V_local[1,1,4] = 1
    W_local[0,0,4] = -1; W_local[0,1,4] = 1
    
    U_local[1,0,5] = 1; U_local[0,0,5] = -1
    V_local[0,0,5] = 1; V_local[0,1,5] = 1
    W_local[1,1,5] = 1
    
    U_local[1,1,6] = 1; U_local[0,1,6] = -1
    V_local[0,1,6] = 1; V_local[1,1,6] = 1
    W_local[0,0,6] = 1; W_local[1,1,6] = 1
    
    # Part 2: A12 * Bbot (i in {0,1}, j=2, k in {0,1})
    U_local[0,2,7] = 1; U_local[1,2,7] = 1
    V_local[2,0,7] = 1; V_local[2,1,7] = 1
    W_local[0,0,7] = 1; W_local[0,1,7] = 1
    W_local[1,0,7] = 1; W_local[1,1,7] = 1
    
    # Part 3: A21 * Btop (i=2, j in {0,1}, k in {0,1})
    U_local[2,0,8] = 1; U_local[2,1,8] = 1
    V_local[0,0,8] = 1; V_local[0,1,8] = 1
    V_local[1,0,8] = 1; V_local[1,1,8] = 1
    W_local[0,2,8] = 1; W_local[1,2,8] = 1
    
    # Part 4: A22 * Bbot (i=2, j=2, k in {0,1})
    U_local[2,2,9] = 1
    V_local[2,0,9] = 1; V_local[2,1,9] = 1
    W_local[0,2,9] = 1; W_local[1,2,9] = 1

    # Construct full U, V, W for M_{3,3,8}
    r_total = 40
    U = np.zeros((9, r_total))
    V = np.zeros((24, r_total))
    W = np.zeros((24, r_total))
    
    for l in range(4):
        offset = l * 10
        U[:, offset:offset+10] = U_local.reshape(9, 10)
        
        for j in range(3):
            for k_loc in range(2):
                k_glob = 2*l + k_loc
                V[j*8 + k_glob, offset:offset+10] = V_local[j, k_loc, :]
        
        for k_loc in range(2):
            k_glob = 2*l + k_loc
            for i in range(3):
                W[k_glob*3 + i, offset:offset+10] = W_local[k_loc, i, :]

    # Verify
    if verify_solution(U, V, W, r_total, n, m, p):
        U_list = [[int(x) if x == int(x) else x for x in row] for row in U]
        V_list = [[int(x) if x == int(x) else x for x in row] for row in V]
        W_list = [[int(x) if x == int(x) else x for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback
    return {"U": U_school, "V": V_school, "W": W_school}
