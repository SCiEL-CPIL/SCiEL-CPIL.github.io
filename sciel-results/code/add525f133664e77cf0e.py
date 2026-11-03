import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback (Rank 64)
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

    # The tensor T is defined by T[i*m+j, j*p+k, k*n+i] = 1
    # We seek rank r such that T = sum_{c=0}^{r-1} u_c v_c^T w_c^T
    
    # New strategy: Embed 2x4 * 4x8 into 4x4 * 4x4 multiplication
    # A (2x4) -> embed into 4x4: [[A, 0], [0, 0]] where A is 2x4
    # B (4x8) -> embed into 4x4: [[B_top, 0], [0, 0]] where B_top is 4x4 (first 4 cols of B)
    # Wait, this doesn't work directly because B is 4x8, not 4x4.
    
    # Let's rethink: The task is 2x4 * 4x8 -> 2x8
    # We can view this as two 2x4 * 4x4 multiplications: A * B1 + A * B2 where B = [B1 | B2]
    # Each 2x4 * 4x4 can be embedded into 4x4 * 4x4.
    # But that would be 2 * rank(4x4) = 2 * 32 = 64, which is not better than schoolbook.
    
    # Alternative: Use the fact that we have a 2x4 * 4x8 matrix.
    # We can use a block Strassen approach on the 2x4 and 4x8 directly.
    # Split A (2x4) into two 2x2 blocks: A = [A1 | A2]
    # Split B (4x8) into four 2x2 blocks: B = [[B11, B12], [B21, B22]]
    # Then A*B = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # This is 4 multiplications of 2x2 matrices, but they are not independent.
    # Actually, we can use Strassen on the 2x4 * 4x8 by padding to 4x8 * 4x8? No.
    
    # Let's try a different embedding:
    # Embed A (2x4) into a 4x4 matrix: A_emb = [[A, 0_2x2], [0_2x2, 0_2x2]]
    # Embed B (4x8) into a 4x8 matrix? No, we need square matrices for Strassen.
    
    # Actually, the known result for 2x4 * 4x8 is that it can be done with rank 28 or so?
    # Let's check the literature. The rank of 2x4 * 4x8 is at most 2 * rank(2x2*2x4) + ...
    # A 2x2 * 2x4 multiplication has rank 8 (schoolbook) or 7 (Strassen-like).
    # If we use rank 7 for 2x2*2x4, then for 2x4*4x8 we can split into two 2x2*2x4:
    # A = [A1 | A2], B = [[B11, B12], [B21, B22]]
    # A*B = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # This requires computing A1*B11, A2*B21, A1*B12, A2*B22.
    # Each is 2x2 * 2x4. If each has rank 7, total rank is 4*7 = 28? No, they share inputs.
    # Actually, A1 and A2 are shared, B11, B12, B21, B22 are shared.
    # We can use a tensor product structure.
    
    # Let's use the block decomposition from the current program but try to optimize it.
    # The current program uses rank 56. Can we do better?
    
    # Actually, let's try a direct Strassen-like approach for 2x4 * 4x8.
    # Split A (2x4) into [A1 | A2] where A1, A2 are 2x2.
    # Split B (4x8) into [B1 | B2] where B1, B2 are 4x4.
    # Then A*B = A1*B1 + A2*B2.
    # Each term is 2x2 * 4x4.
    # A 2x2 * 4x4 multiplication can be done with rank 12? (2*2*4=16 schoolbook, Strassen might reduce it)
    # Actually, 2x2 * 4x4: pad 2x2 to 4x4, then use 4x4*4x4 Strassen (rank 32), but that's wasteful.
    # Alternatively, use the fact that 2x2 * 4x4 = (2x2 * 2x2) + (2x2 * 2x2) with shared 2x2 block.
    # Rank of 2x2*4x4 is at most 2*7 + 7 = 21? No, that's not right.
    
    # Let's stick with the block decomposition but try to find a better one.
    # The current block decomposition uses 8 independent 2x2*2x2 Strassen blocks, giving rank 56.
    # Can we share some of the Strassen terms?
    
    # For 2x4 * 4x8:
    # A = [A1 | A2] (2x2 each)
    # B = [[B11, B12], [B21, B22]] (2x2 each)
    # A*B = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # We need to compute:
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    # Each Ci is 2x2.
    # We can compute A1*B11, A2*B21, A1*B12, A2*B22 using Strassen.
    # But A1 and A2 are shared, B11, B12, B21, B22 are shared.
    # Total Strassen multiplications: 4 * 7 = 28? But with sharing, maybe less.
    
    # Let's try to construct a rank 28 decomposition.
    # This is complex. Given the time limit, let's return the current best (rank 56) if we can't do better quickly.
    
    # Actually, the known optimal rank for 2x4 * 4x8 is 28? Let's check.
    # R(2,4,8) <= R(2,2,2) * R(1,2,4) = 7 * 8 = 56. Not helpful.
    # R(2,4,8) <= 2 * R(2,2,4) = 2 * 12 = 24? If R(2,2,4)=12.
    # Actually, R(2,2,4) is at most 8 (schoolbook) or 7 (Strassen-like for 2x2*2x4).
    # If R(2,2,4)=7, then R(2,4,8) <= 2*7 + 2*7 = 28? No, that's not how tensor ranks compose.
    
    # Given the complexity, let's return the rank 56 solution from the current program.
    
    def get_block_decomposition():
        U_22 = np.zeros((4, 7))
        V_22 = np.zeros((4, 7))
        W_22 = np.zeros((4, 7))
        
        U_22[0, 0] = 1; U_22[3, 0] = 1
        V_22[0, 0] = 1; V_22[3, 0] = 1
        W_22[0, 0] = 1; W_22[3, 0] = 1
        U_22[2, 1] = 1; U_22[3, 1] = 1
        V_22[0, 1] = 1
        W_22[2, 1] = 1; W_22[3, 1] = -1
        U_22[0, 2] = 1
        V_22[1, 2] = 1; V_22[3, 2] = -1
        W_22[1, 2] = 1; W_22[3, 2] = 1
        U_22[3, 3] = 1
        V_22[2, 3] = 1; V_22[0, 3] = -1
        W_22[0, 3] = 1; W_22[2, 3] = 1
        U_22[0, 4] = 1; U_22[1, 4] = 1
        V_22[3, 4] = 1
        W_22[0, 4] = -1; W_22[1, 4] = 1
        U_22[2, 5] = 1; U_22[0, 5] = -1
        V_22[0, 5] = 1; V_22[1, 5] = 1
        W_22[3, 5] = 1
        U_22[1, 6] = 1; U_22[3, 6] = -1
        V_22[2, 6] = 1; V_22[3, 6] = 1
        W_22[0, 6] = 1; W_22[1, 6] = -1

        U_global = np.zeros((8, 56))
        V_global = np.zeros((32, 56))
        W_global = np.zeros((16, 56))
        
        def add_strassen_block(U_g, V_g, W_g, u_map, v_map, w_map, start_col):
            for c in range(7):
                for l in range(4):
                    if U_22[l, c] != 0:
                        U_g[u_map[l], start_col + c] += U_22[l, c]
                    if V_22[l, c] != 0:
                        V_g[v_map[l], start_col + c] += V_22[l, c]
                    if W_22[l, c] != 0:
                        W_g[w_map[l], start_col + c] += W_22[l, c]

        add_strassen_block(U_global, V_global, W_global, [0, 1, 4, 5], [0, 1, 8, 9], [0, 1, 2, 3], 0)
        add_strassen_block(U_global, V_global, W_global, [0, 1, 4, 5], [2, 3, 10, 11], [4, 5, 6, 7], 7)
        add_strassen_block(U_global, V_global, W_global, [0, 1, 4, 5], [4, 5, 12, 13], [8, 9, 10, 11], 14)
        add_strassen_block(U_global, V_global, W_global, [0, 1, 4, 5], [6, 7, 14, 15], [12, 13, 14, 15], 21)
        add_strassen_block(U_global, V_global, W_global, [2, 3, 6, 7], [16, 17, 24, 25], [0, 1, 2, 3], 28)
        add_strassen_block(U_global, V_global, W_global, [2, 3, 6, 7], [18, 19, 26, 27], [4, 5, 6, 7], 35)
        add_strassen_block(U_global, V_global, W_global, [2, 3, 6, 7], [20, 21, 28, 29], [8, 9, 10, 11], 42)
        add_strassen_block(U_global, V_global, W_global, [2, 3, 6, 7], [22, 23, 30, 31], [12, 13, 14, 15], 49)
        
        return U_global, V_global, W_global

    U_block, V_block, W_block = get_block_decomposition()
    
    def verify(U, V, W):
        r = U.shape[1]
        T_recon = np.zeros((8, 32, 16))
        for c in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        T_orig = np.zeros((8, 32, 16))
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    T_orig[i*4+j, j*8+k, k*2+i] = 1
        
        return np.allclose(T_recon, T_orig, atol=1e-9)

    if verify(U_block, V_block, W_block):
        return {"U": U_block.tolist(), "V": V_block.tolist(), "W": W_block.tolist()}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
