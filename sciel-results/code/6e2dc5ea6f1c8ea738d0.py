"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_2x2x1():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        rank = 4
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        return U, V, W

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        # Term 0: M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # Term 1: M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        # Term 2: M3
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        # Term 3: M4
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        # Term 4: M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        # Term 5: M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # Term 6: M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        return U, V, W

    def build_2x2x3():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3 (4 schoolbook + 7 Strassen)."""
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]

        U1, V1, W1 = build_2x2x1()
        for c in range(4):
            for r in range(4):
                if U1[r][c] != 0:
                    U[r][c] += U1[r][c]
            for r in range(2):
                if V1[r][c] != 0:
                    V[r*3][c] += V1[r][c]
            for r in range(2):
                if W1[r][c] != 0:
                    W[r][c] += W1[r][c]

        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        v2_map = {0: 1, 1: 2, 2: 4, 3: 5}
        w2_map = {0: 2, 1: 4, 2: 3, 3: 5}
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0:
                    U[r][base + c] += U2[r][c]
            for r in range(4):
                if V2[r][c] != 0:
                    V[v2_map[r]][base + c] += V2[r][c]
            for r in range(4):
                if W2[r][c] != 0:
                    W[w2_map[r]][base + c] += W2[r][c]

        return U, V, W

    def build_4x4x6():
        """Rank-77 Strassen decomposition for 4x4 * 4x6 -> 4x6."""
        n, m, p = 4, 4, 6
        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = len(U_sub[0])  # 11
        total_rank = 7 * rank_sub  # 77

        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]

        terms = [
            {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
            {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)]},
            {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
            {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
            {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
            {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
            {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]},
        ]

        u_offset = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
        v_offset = {0: (0, 0), 1: (0, 3), 2: (2, 0), 3: (2, 3)}
        w_offset = {0: (0, 0), 1: (3, 0), 2: (0, 2), 3: (3, 2)}

        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0:
                        continue
                    i_local = u_local // 2
                    j_local = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offset[block]
                        g_u = (i_local + ro) * m + (j_local + co)
                        U[g_u][base_col + c] += u_val * coeff

                for v_local in range(6):
                    v_val = V_sub[v_local][c]
                    if v_val == 0:
                        continue
                    j_local = v_local // 3
                    k_local = v_local % 3
                    for block, coeff in term['B']:
                        ro, co = v_offset[block]
                        g_v = (j_local + ro) * p + (k_local + co)
                        V[g_v][base_col + c] += v_val * coeff

                for w_local in range(6):
                    w_val = W_sub[w_local][c]
                    if w_val == 0:
                        continue
                    k_local = w_local // 2
                    i_local = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offset[block]
                        g_w = (k_local + ro) * n + (i_local + co)
                        W[g_w][base_col + c] += w_val * coeff

        return U, V, W

    def build_4x4x6_bcl():
        """
        Attempt to build a lower-rank decomposition using BCL for 4x4x4 core
        and a shared strip for the remaining columns.
        
        BCL 4x4x4 has rank 13.
        For 4x4x6, we can view it as 4x4x4 + 4x4x2.
        If we can find a rank-13 decomposition for 4x4x4 and a rank-13 for 4x4x2
        that share some structure, or simply use BCL for the first 4 columns and
        schoolbook for the last 2, the rank would be 13 + 8 = 21?
        No, the decomposition must be for the whole tensor.
        
        A known rank for 4x4x6 is 49 (Strassen) or lower?
        Actually, the Strassen rank for 4x4x4 is 49.
        The Strassen rank for 4x4x2 is 4*4*2 = 32 schoolbook, or less?
        Using Strassen on 4x4x2:
        Split into 2x2 blocks?
        4x4x2 can be seen as two 4x4x1 multiplications.
        Rank of 4x4x1 is 16 (schoolbook).
        So Strassen 4x4x2 is 32.
        Total 49 + 32 = 81. This is worse than 77.
        
        Why is 77 better?
        Because the recursive Strassen for 4x4x6 uses the 2x2x3 block.
        Rank 2x2x3 is 11.
        7 * 11 = 77.
        
        Can we do better than 11 for 2x2x3?
        2x2x3 tensor.
        Schoolbook is 12.
        Strassen-like:
        Split 2x2x3 into (2x2x2) + (2x2x1).
        Rank 2x2x2 = 7.
        Rank 2x2x1 = 4.
        Total 11.
        Is 11 minimal?
        Lower bound for 2x2x3:
        It contains 2x2x2 (rank 7) and 2x2x1 (rank 4).
        They share no terms in the simple concatenation.
        So 11 is likely optimal for this split.
        
        What about a different split?
        2x2x3.
        Maybe a rank-10 decomposition exists?
        If rank 2x2x3 is 10, then 4x4x6 is 70.
        
        Let's try to find a rank-10 decomposition for 2x2x3.
        Or use a different base block.
        
        Alternative: Use BCL 4x4x4 (rank 13) and add 4x4x2 terms.
        If we can find a rank-13 decomposition for 4x4x4, and then add 4x4x2 terms
        that are "orthogonal" or independent, the rank is 13 + rank(4x4x2).
        Rank 4x4x2 is at least 8 (since 4x4x1 is 4? No, 4x4x1 is 16).
        Wait, 4x4x1 matrix multiplication: A(4x4) * B(4x1) = C(4x1).
        Rank is 16 (schoolbook).
        So 4x4x2 is 32.
        Total 13 + 32 = 45? No, the tensor is 4x4x6.
        The decomposition is not additive in that way unless the supports are disjoint.
        The BCL decomposition for 4x4x4 uses all 16 entries of A and B and 16 of C (for the 4x4 part).
        The 4x4x2 part uses the same A, different B columns, and different C columns.
        So we can combine them.
        Total rank = 13 (for 4x4x4 part) + 32 (for 4x4x2 part) = 45?
        This assumes the BCL terms don't interfere with the 4x4x2 terms.
        Since the C indices are disjoint (0-3 vs 4-5), and B indices are disjoint (0-3 vs 4-5),
        and A indices are shared, we can just concatenate the terms.
        The A factors for the 4x4x2 part would be the same A, so we need to match the A factors.
        This is complex.
        
        Let's stick to the current 77 and try to optimize the 2x2x3 block.
        If we can get 2x2x3 down to 10, we get 70.
        
        Let's try to construct a rank-10 2x2x3.
        U (4x10), V (6x10), W (6x10).
        This is a search problem.
        Given time constraints, I will stick to the known 77.
        """
        return None

    # Try the BCL-based approach if feasible, otherwise fall back to Strassen
    U, V, W = None
    if time.time() < deadline - 5:
        # We could try to search for a better rank, but 77 is solid.
        # For now, return the 77 rank solution.
        U, V, W = build_4x4x6()

    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    if abs(sum(prod) - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
