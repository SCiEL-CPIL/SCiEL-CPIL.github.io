"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 5

    # We decompose 4x4x5 into 4x4x4 (rank 49 via Strassen) and 4x4x1 (rank 16 via Schoolbook).
    # Total rank = 65.

    # Helper function to generate Strassen decomposition for N x N x N
    def strassen_uv_w(n):
        if n == 1:
            return np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])
        
        half = n // 2
        U_sub, V_sub, W_sub = strassen_uv_w(half)
        R_sub = U_sub.shape[1]
        
        # We need to construct U, V, W of size (n*n, 7*R_sub)
        U = np.zeros((n*n, 7*R_sub))
        V = np.zeros((n*n, 7*R_sub))
        W = np.zeros((n*n, 7*R_sub))
        
        # Strassen steps definitions
        # Each step has A_map, B_map, C_map
        # A_map: list of (bi, bj, coeff) where bi, bj are block indices (0 or 1)
        # B_map: list of (bj, bk, coeff)
        # C_map: list of (bi, bk, coeff)
        
        steps = [
            # M1 = (A00 + A11) * (B00 + B11)
            ( [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)] ),
            # M2 = (A10 + A11) * B00
            ( [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)] ),
            # M3 = A00 * (B01 - B11)
            ( [(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)] ),
            # M4 = A11 * (B10 - B00)
            ( [(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)] ),
            # M5 = (A00 + A01) * B11
            ( [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)] ),
            # M6 = (A10 - A00) * (B00 + B01)
            ( [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)] ),
            # M7 = (A01 - A11) * (B10 + B11)
            ( [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)] ),
        ]
        
        for s in range(7):
            A_map, B_map, C_map = steps[s]
            offset = s * R_sub
            
            # Fill U
            for i_loc in range(half):
                for j_loc in range(half):
                    for r in range(R_sub):
                        val = U_sub[i_loc * half + j_loc, r]
                        if val == 0: continue
                        for bi, bj, coeff in A_map:
                            i_glob = bi * half + i_loc
                            j_glob = bj * half + j_loc
                            U[i_glob * n + j_glob, offset + r] += coeff * val
                            
            # Fill V
            for j_loc in range(half):
                for k_loc in range(half):
                    for r in range(R_sub):
                        val = V_sub[j_loc * half + k_loc, r]
                        if val == 0: continue
                        for bj, bk, coeff in B_map:
                            j_glob = bj * half + j_loc
                            k_glob = bk * half + k_loc
                            V[j_glob * n + k_glob, offset + r] += coeff * val
                            
            # Fill W
            for k_loc in range(half):
                for i_loc in range(half):
                    for r in range(R_sub):
                        val = W_sub[k_loc * half + i_loc, r]
                        if val == 0: continue
                        for bi, bk, coeff in C_map:
                            i_glob = bi * half + i_loc
                            k_glob = bk * half + k_loc
                            W[k_glob * n + i_glob, offset + r] += coeff * val
                            
        return U, V, W

    # Get Strassen decomposition for 4x4x4
    U4, V4, W4 = strassen_uv_w(4)
    R4 = U4.shape[1] # Should be 49
    
    # Get Schoolbook decomposition for 4x4x1
    # A is 4x4, B2 is 4x1 (column 4 of B), C2 is 4x1 (column 4 of C)
    R1 = 16
    U1 = np.zeros((16, R1))
    V1 = np.zeros((20, R1)) # m*p = 4*5 = 20
    W1 = np.zeros((20, R1)) # p*n = 5*4 = 20
    
    col = 0
    for i in range(4):
        for j in range(4):
            # U[i*m+j, col] = 1
            U1[i * 4 + j, col] = 1
            # V[j*p+k, col] = 1 where k=4
            V1[j * 5 + 4, col] = 1
            # W[k*n+i, col] = 1 where k=4
            W1[4 * 4 + i, col] = 1
            col += 1
            
    # Combine
    R_total = R4 + R1
    U = np.hstack([U4, U1])
    V = np.hstack([V4, V1])
    W = np.hstack([W4, W1])
    
    # Optional: Verify validity
    # def check_valid(U, V, W, R):
    #     for i in range(n):
    #         for j in range(m):
    #             for k in range(p):
    #                 val = 0
    #                 for r in range(R):
    #                     val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    #                 if abs(val - 1) > 1e-6:
    #                     return False
    #     return True
    # if not check_valid(U, V, W, R_total):
    #     # Fallback to schoolbook if verification fails (should not happen)
    #     pass

    if time.time() - start_time > DEADLINE - 5:
        # Fallback to schoolbook if time is running out (unlikely)
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
