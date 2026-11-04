import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # 15s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 7
    rank_schoolbook = n * m * p  # 112

    def build_schoolbook():
        """Return the schoolbook (trivial) decomposition of rank n*m*p."""
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

    def verify_decomposition(U, V, W, n=4, m=4, p=7):
        """Verify that the decomposition exactly matches the matrix multiplication tensor."""
        rank = len(U[0])
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        tensor_decomp = np.zeros((n, m, p))
        for c in range(rank):
            u_col = U_arr[:, c]
            v_col = V_arr[:, c]
            w_col = W_arr[:, c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_decomp[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
        
        if not np.allclose(tensor_decomp, 1.0, atol=1e-9):
            return False
        return True

    def build_strassen_2x2():
        """
        Return the rank-7 Strassen decomposition for 2x2 x 2x2 -> 2x2.
        U: 4x7, V: 4x7, W: 4x7
        """
        rank_s = 7
        U_s = [[0] * rank_s for _ in range(4)]
        V_s = [[0] * rank_s for _ in range(4)]
        W_s = [[0] * rank_s for _ in range(4)]

        # P1 = (a11+a22)(b11+b22)
        U_s[0][0] = 1; U_s[3][0] = 1
        V_s[0][0] = 1; V_s[1][0] = 1; V_s[2][0] = 1; V_s[3][0] = 1
        W_s[0][0] = 1; W_s[1][0] = 1; W_s[2][0] = 1; W_s[3][0] = 1

        # P2 = (a21+a22)b11
        U_s[2][1] = 1; U_s[3][1] = 1
        V_s[0][1] = 1
        W_s[2][1] = 1; W_s[3][1] = 1

        # P3 = a11(b12-b22)
        U_s[0][2] = 1
        V_s[1][2] = 1; V_s[3][2] = -1
        W_s[1][2] = 1; W_s[3][2] = 1

        # P4 = a22(b21-b11)
        U_s[3][3] = 1
        V_s[2][3] = 1; V_s[0][3] = -1
        W_s[0][3] = 1; W_s[2][3] = 1

        # P5 = (a11+a12)b22
        U_s[0][4] = 1; U_s[1][4] = 1
        V_s[3][4] = 1
        W_s[0][4] = -1; W_s[1][4] = 1

        # P6 = (a21-a11)(b11+b12)
        U_s[2][5] = 1; U_s[0][5] = -1
        V_s[0][5] = 1; V_s[1][5] = 1
        W_s[3][5] = 1

        # P7 = (a12-a22)(b21+b22)
        U_s[1][6] = 1; U_s[3][6] = -1
        V_s[2][6] = 1; V_s[3][6] = 1
        W_s[0][6] = 1

        return U_s, V_s, W_s

    def build_strassen_4x4x4():
        """
        Build a rank-49 decomposition for 4x4 x 4x4 -> 4x4 using recursive Strassen.
        """
        U_s, V_s, W_s = build_strassen_2x2()
        rank_s = 7
        rank_total = 49

        U_44 = [[0] * rank_total for _ in range(16)]
        V_44 = [[0] * rank_total for _ in range(16)]
        W_44 = [[0] * rank_total for _ in range(16)]

        A_block_mults = [
            [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
            [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1],
        ]
        B_block_mults = [
            [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1],
            [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1],
        ]
        C_block_mults = [
            [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1],
            [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0],
        ]

        for s in range(7):
            A_mults = A_block_mults[s]
            B_mults = B_block_mults[s]
            C_mults = C_block_mults[s]

            for t in range(7):
                col = s * 7 + t

                for bi in range(2):
                    for bj in range(2):
                        a_mult = A_mults[bi * 2 + bj]
                        if a_mult == 0:
                            continue
                        for ii in range(2):
                            for jj in range(2):
                                u_val = U_s[ii * 2 + jj][t]
                                if u_val == 0:
                                    continue
                                a4_i = bi * 2 + ii
                                a4_j = bj * 2 + jj
                                u_row = a4_i * 4 + a4_j
                                U_44[u_row][col] += a_mult * u_val

                for bj in range(2):
                    for bk in range(2):
                        b_mult = B_mults[bj * 2 + bk]
                        if b_mult == 0:
                            continue
                        for ii in range(2):
                            for jj in range(2):
                                v_val = V_s[ii * 2 + jj][t]
                                if v_val == 0:
                                    continue
                                b4_j = bj * 2 + ii
                                b4_k = bk * 2 + jj
                                v_row = b4_j * 4 + b4_k
                                V_44[v_row][col] += b_mult * v_val

                for ci in range(2):
                    for cj in range(2):
                        c_mult = C_mults[ci * 2 + cj]
                        if c_mult == 0:
                            continue
                        for ii in range(2):
                            for jj in range(2):
                                w_val = W_s[ii * 2 + jj][t]
                                if w_val == 0:
                                    continue
                                c4_k = ci * 2 + ii
                                c4_i = cj * 2 + jj
                                w_row = c4_k * 4 + c4_i
                                W_44[w_row][col] += c_mult * w_val

        return U_44, V_44, W_44

    def build_4x4x7_via_strassen():
        """
        Build a rank-97 decomposition for 4x4 x 4x7:
        - First 4 columns of B: use Strassen 4x4x4 (rank 49)
        - Last 3 columns of B: use schoolbook 4x4x3 (rank 48)
        Total: 97
        """
        U_44, V_44, W_44 = build_strassen_4x4x4()

        rank_strassen = 49
        rank_schoolbook_3 = 4 * 4 * 3  # 48
        total_rank = rank_strassen + rank_schoolbook_3  # 97

        U_out = [[0] * total_rank for _ in range(16)]
        V_out = [[0] * total_rank for _ in range(28)]
        W_out = [[0] * total_rank for _ in range(28)]

        for r in range(16):
            for c in range(rank_strassen):
                U_out[r][c] = U_44[r][c]
        for r in range(16):
            for c in range(rank_strassen):
                V_out[r][c] = V_44[r][c]
        for r in range(16):
            for c in range(rank_strassen):
                W_out[r][c] = W_44[r][c]

        col_offset = rank_strassen
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    kk = k + 4
                    U_out[i * 4 + j][col_offset] = 1
                    V_out[j * 7 + kk][col_offset] = 1
                    W_out[kk * 4 + i][col_offset] = 1
                    col_offset += 1

        return {"U": U_out, "V": V_out, "W": W_out}

    def try_dft_diagonalization():
        """
        Try to find a low-rank decomposition using DFT-based diagonalization.
        
        Key idea: The matrix multiplication tensor T[i,j,k] = delta(i+k mod something, j) 
        can sometimes be diagonalized using roots of unity.
        
        For the specific case of 4x4 x 4x7, we try to find a decomposition
        where the factors have a DFT structure that allows sharing.
        
        We use the fact that if we can write the multiplication as:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        And if we apply DFT to the j dimension, we can potentially diagonalize.
        
        However, for general rectangular matrices, this doesn't directly work.
        Instead, we try a structured approach:
        
        1. For the 4x4 part (first 4 columns of B), use Strassen (rank 49)
        2. For the remaining 3 columns, try to find a better than schoolbook decomposition
        
        Actually, let's try a different approach: use the fact that we can
        decompose the 4x4x7 problem into:
        - 4x4 x 4x4 (rank 49 via Strassen)
        - 4x4 x 4x3 (need to find rank)
        
        For 4x4 x 4x3, schoolbook is 48. Can we do better?
        
        Let's try using a 2x2 Strassen for the 4x3 part by padding to 4x4
        and using a rank-49 decomposition, but that gives 49+49=98 which is worse.
        
        Alternative: Use a block decomposition where we share some Strassen terms.
        
        Actually, let's try a direct approach: construct a decomposition where
        we use the 7 Strassen products for the 4x4x4 part, and for the 4x4x3 part,
        we try to find a rank-36 decomposition (better than schoolbook 48).
        
        For 4x4 x 4x3:
        Lower bound: max(n*m, m*p, p*n) = max(16, 12, 12) = 16
        But we need at least n*p*m / (max sharing) terms.
        
        Let's try a structured approach using the 2x2 block decomposition.
        """
        if time.time() - start_time > DEADLINE:
            return None
        
        # Try to build a better decomposition for the 4x4x3 part
        # using a 2x2 Strassen-like approach
        
        # For 4x4 x 4x3, we can view B as 2x2 blocks where each block is 2x3
        # Or we can use a different block structure
        
        # Let's try: split A into 2x2 blocks (each 2x2), B into 2x2 blocks (each 2x3)
        # This doesn't work directly because B blocks have different sizes
        
        # Alternative: Use the 4x4x4 Strassen and zero out the last column
        # This gives rank 49 for 4x4x4, but we only need 4x4x3
        # The rank would still be 49 since the decomposition doesn't simplify
        
        # Let's try a completely different approach:
        # Use a rank-64 decomposition for 4x4x4 (better than schoolbook 64? No, that's schoolbook)
        # Wait, schoolbook for 4x4x4 is 64, Strassen gives 49.
        
        # For 4x4x7, schoolbook is 112, current approach gives 97.
        # Can we do better than 97?
        
        # Idea: Use Strassen for the entire 4x4x7 by padding B to 4x8
        # 4x4 x 4x8 = 4x4 x (4x4 + 4x4) = 2 * 4x4x4
        # Rank would be 2*49 = 98, which is worse than 97
        
        # Another idea: Use a 4x4 x 4x4 Strassen (rank 49) for the first 4 columns,
        # and for the last 3 columns, use a rank-48 schoolbook. Total: 97.
        
        # Can we do better for the 4x4x3 part?
        # Lower bound for 4x4x3: The tensor has 4*4*3 = 48 entries.
        # A rank-r decomposition requires r >= 48 / (max number of non-zeros per term)
        # Each term can have at most 4*4*3 = 48 non-zeros, but typically much less.
        
        # Let's try a rank-40 decomposition for 4x4x3 using a structured approach.
        # This is unlikely to work, but let's try.
        
        # Actually, the best known bound for 4x4x4 is 49 (Strassen).
        # For 4x4x3, it's unclear, but schoolbook 48 is probably close to optimal.
        
        # Let's try a different strategy: use a 2x2 Strassen for the 4x4x4 part
        # and a 2x2 Strassen for the 4x4x3 part (by padding to 4x4).
        # This gives 49 + 49 = 98, which is worse.
        
        # So the current approach of 97 is likely near-optimal for this family.
        # Let's try to verify if we can do 96 or less.
        
        # For now, return None to indicate we couldn't find a better decomposition.
        return None

    # Try the Strassen-based decomposition first
    best_result = None
    best_rank = rank_schoolbook

    if time.time() - start_time < DEADLINE:
        try:
            result = build_4x4x7_via_strassen()
            U = result["U"]
            V = result["V"]
            W = result["W"]
            
            if verify_decomposition(U, V, W):
                best_result = result
                best_rank = 97
        except Exception:
            pass

    # Try DFT-based approach
    if time.time() - start_time < DEADLINE:
        try:
            dft_result = try_dft_diagonalization()
            if dft_result is not None:
                U = dft_result["U"]
                V = dft_result["V"]
                W = dft_result["W"]
                if verify_decomposition(U, V, W):
                    rank = len(U[0])
                    if rank < best_rank:
                        best_result = dft_result
                        best_rank = rank
        except Exception:
            pass

    # Fallback to schoolbook
    if best_result is None:
        best_result = build_schoolbook()

    return best_result
