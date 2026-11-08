"""Rewritten program: attempt a lower-rank decomposition for 2x4 * 4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS
    # n=2, m=4, p=8

    # The schoolbook rank is n*m*p = 64.
    # We can potentially do better by exploiting the small dimensions.
    # Let's try to find a decomposition with rank < 64.
    
    # One approach: use the fact that n=2 is small.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can try to use a rank-7 Strassen-like decomposition for parts of it,
    # or find a specific low-rank decomposition.
    
    # For now, let's implement a robust schoolbook decomposition as a baseline,
    # and then try to improve it.
    
    # Actually, let's try a direct approach:
    # The tensor T_{i,j,k} = 1 if the product A[i,j]B[j,k] contributes to C[i,k].
    # We want to decompose T = sum_r u_r \otimes v_r \otimes w_r.
    
    # A known lower bound is max(n*m, m*p, p*n) = max(8, 32, 16) = 32.
    # The schoolbook rank is 64.
    # Can we achieve rank 48 or 56?
    
    # Let's try to construct a rank-56 decomposition using Strassen on 2x2 blocks.
    # A is 2x4, B is 4x8.
    # Split A into two 2x2 blocks: A_L (cols 0,1), A_R (cols 2,3).
    # Split B into four 2x2 blocks: B_0 (rows 0,1; cols 0,1), B_1 (rows 0,1; cols 2,3),
    #                               B_2 (rows 2,3; cols 0,1), B_3 (rows 2,3; cols 2,3).
    # C is 2x8. Split into four 2x2 blocks: C_0 (cols 0,1), C_1 (cols 2,3),
    #                                         C_2 (cols 4,5), C_3 (cols 6,7).
    # C_0 = A_L @ B_0 + A_R @ B_2
    # C_1 = A_L @ B_1 + A_R @ B_3
    # C_2 = A_L @ B_2 + A_R @ B_0  -- Wait, this is not correct.
    # Let's re-derive:
    # C[i,k] = sum_{j=0}^3 A[i,j] B[j,k]
    # For k in {0,1} (C_0):
    #   C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0] + A[i,2]B[2,0] + A[i,3]B[3,0]
    #   C[i,1] = A[i,0]B[0,1] + A[i,1]B[1,1] + A[i,2]B[2,1] + A[i,3]B[3,1]
    # So C_0 (2x2) = A_L @ B_top_left + A_R @ B_bottom_left
    # where B_top_left is B[0:2, 0:2] and B_bottom_left is B[2:4, 0:2].
    # Similarly:
    # C_1 = A_L @ B[0:2, 2:4] + A_R @ B[2:4, 2:4]
    # C_2 = A_L @ B[0:2, 4:6] + A_R @ B[2:4, 4:6]
    # C_3 = A_L @ B[0:2, 6:8] + A_R @ B[2:4, 6:8]
    
    # So we have 8 2x2 matrix multiplications:
    # A_L @ B[0:2, 0:2], A_R @ B[2:4, 0:2],
    # A_L @ B[0:2, 2:4], A_R @ B[2:4, 2:4],
    # A_L @ B[0:2, 4:6], A_R @ B[2:4, 4:6],
    # A_L @ B[0:2, 6:8], A_R @ B[2:4, 6:8]
    
    # Each 2x2 multiplication has rank 7 (Strassen).
    # Total rank = 8 * 7 = 56.
    
    # Let's implement this Strassen-based decomposition.
    
    # Strassen formulas for 2x2 matrices X, Y:
    # M1 = (X00 - X11)(Y11 - Y00)
    # M2 = (X00 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 + X11)(Y10 - Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # u, v, w vectors of length 4 for 2x2 matrices.
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = [
        ([1, 0, 0, -1], [-1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, -1]),    # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),    # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),    # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),    # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),    # M6
        ([0, 1, 0, 1], [0, 0, 1, -1], [1, 0, 0, -1]),   # M7
    ]
    
    # Verify Strassen locally
    def verify_strassen():
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C_ref = A @ B
        C_calc = np.zeros((2, 2))
        for u, v, w in strassen_terms:
            u_a = np.array(u)
            v_b = np.array(v)
            w_c = np.array(w)
            A_flat = A.flatten()
            B_flat = B.flatten()
            scalar_u = np.dot(u_a, A_flat)
            scalar_v = np.dot(v_b, B_flat)
            C_calc_flat = C_calc.flatten()
            C_calc_flat += scalar_u * scalar_v * w_c
            C_calc = C_calc_flat.reshape(2, 2)
        return np.allclose(C_calc, C_ref)
    
    if not verify_strassen():
        # Fallback to schoolbook if Strassen implementation is wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    # Build the full decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    current_col = 0
    
    # 8 2x2 multiplications:
    # For each of the 4 column blocks of B (2x2 each), we have:
    #   A_L @ B_block (for C_L part)
    #   A_R @ B_block (for C_R part)
    # But actually, C is 2x8, so each column block of C (2x2) is:
    #   C_block = A_L @ B_top_block + A_R @ B_bottom_block
    # where B_top_block is B[0:2, col_range] and B_bottom_block is B[2:4, col_range].
    
    for i in range(4):
        # Column block i: cols 2i, 2i+1 of C
        # B_top_block: rows 0,1; cols 2i, 2i+1
        # B_bottom_block: rows 2,3; cols 2i, 2i+1
        
        # A_L @ B_top_block
        for u, v, w in strassen_terms:
            for li in range(2):
                for lj in range(2):
                    val = u[li * 2 + lj]
                    if val != 0:
                        U[li * m + lj][current_col] += val
            
            for lj in range(2):
                for lk in range(2):
                    val = v[lj * 2 + lk]
                    if val != 0:
                        V[lj * p + (2 * i + lk)][current_col] += val
            
            for lk in range(2):
                for li in range(2):
                    val = w[lk * 2 + li]
                    if val != 0:
                        W[(2 * i + lk) * n + li][current_col] += val
            
            current_col += 1
            
        # A_R @ B_bottom_block
        for u, v, w in strassen_terms:
            for li in range(2):
                for lj in range(2):
                    val = u[li * 2 + lj]
                    if val != 0:
                        U[li * m + (2 + lj)][current_col] += val
            
            for lj in range(2):
                for lk in range(2):
                    val = v[lj * 2 + lk]
                    if val != 0:
                        V[(2 + lj) * p + (2 * i + lk)][current_col] += val
            
            for lk in range(2):
                for li in range(2):
                    val = w[lk * 2 + li]
                    if val != 0:
                        W[(2 * i + lk) * n + li][current_col] += val
            
            current_col += 1

    # Final Verification
    def verify_full():
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C_ref = A @ B
        
        U_mat = np.array(U).T # rank x n*m
        V_mat = np.array(V).T # rank x m*p
        W_mat = np.array(W).T # rank x p*n
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0
                for r in range(rank):
                    w_val = W_mat[r, k * n + i]
                    if w_val == 0: continue
                    u_sum = 0
                    for j in range(m):
                        u_sum += A[i, j] * U_mat[r, i * m + j]
                    v_sum = 0
                    for j in range(m):
                        v_sum += B[j, k] * V_mat[r, j * p + k]
                    val += w_val * u_sum * v_sum
                C_calc[i, k] = val
        
        return np.allclose(C_calc, C_ref)

    if not verify_full():
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
