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

    # BCL rank-13 for 4x4x4, extended to 4x4x6
    # The BCL algorithm uses 13 multiplications for 4x4x4.
    # For 4x4x6, we need 2 more columns of output.
    # We can handle the extra 2 columns with 2 additional rank-1 terms
    # (each column of C requires at most m=4 rank-1 terms in schoolbook,
    #  but we can share structure).
    #
    # Actually, let me think more carefully.
    # C = A * B where A is 4x4, B is 4x6, C is 4x6.
    # We can split B into B1 (4x4) and B2 (4x2).
    # C1 = A * B1 (4x4) -- use BCL rank 13
    # C2 = A * B2 (4x2) -- use rank 4 schoolbook
    # Total rank = 17
    #
    # But we can do better by not fully separating.
    # Let me try a direct construction.
    #
    # Alternative: Use the fact that 4x4x6 can be viewed as
    # computing 6 columns of a 4x4 matrix product.
    # Each column k of C is A * b_k where b_k is the k-th column of B.
    #
    # Let me try rank 15: 13 for the first 4 columns + 2 for the last 2.
    # But the 2 extra columns need at least 2 rank-1 terms each in general,
    # so that's 13 + 4 = 17.
    #
    # Let me try a different approach: direct rank-15 construction.
    #
    # Actually, the minimum rank for 4x4x6 is at least 16 (by various bounds).
    # Let me try to construct rank 16.
    #
    # For now, let me build a valid rank-17 solution (13+4) and verify it.
    # Then try to optimize down.

    rank = 17
    U = [[0]*rank for _ in range(n*m)]  # 16 x 17
    V = [[0]*rank for _ in range(m*p)]  # 24 x 17
    W = [[0]*rank for _ in range(p*n)]  # 24 x 17

    # BCL rank-13 for 4x4x4 (first 4 columns of B, first 4 columns of C)
    # BCL uses the following 13 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B01
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B10
    # M6 = (A10+A11)B11
    # M7 = (A01-A10)(B00+B10)
    # M8 = (A00+A10)(B01+B11)
    # M9 = (A01+A11)(B00-B10)
    # M10 = (A00+A01)(B11-B01)
    # M11 = (A10-A00)(B00+B01)
    # M12 = (A11-A01)(B01+B11)
    # M13 = (A00+A11)(B00+B11) -- wait, this is same as M1
    #
    # Let me use the standard BCL formulation properly.
    # BCL 1979: rank 13 for 4x4x4
    #
    # Actually, let me just use a known rank-13 BCL construction.
    # The BCL algorithm:
    #
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)(B12)
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)(B21)
    # P6 = (A21+A22)(B22)
    # P7 = (A12-A21)(B11+B21)
    # P8 = (A11+A21)(B12+B22)
    # P9 = (A12+A22)(B11-B21)
    # P10 = (A11+A12)(B22-B12)
    # P11 = (A21-A11)(B11+B12)
    # P12 = (A22-A12)(B12+B22)
    # P13 = (A11+A22)(B11+B22) -- duplicate of P1?
    #
    # I'm getting confused with the exact BCL coefficients. Let me use a
    # verified rank-13 construction.
    #
    # Let me use the approach from Landsberg's notes or a known implementation.
    #
    # Actually, for reliability, let me just use rank 16 schoolbook-like
    # but with Strassen-style optimizations on 2x2 blocks.
    #
    # 4x4x6: split into 2x2 blocks of A (4 blocks), B has 2x3 blocks (4 blocks)
    # C has 2x3 blocks (4 blocks)
    #
    # For each 2x2 block of C: C_ij = sum_k A_ik * B_kj
    # This is a 2x2 * 2x3 -> 2x3 problem for each of 4 output blocks.
    # But the A and B blocks are shared.
    #
    # Total: 4 output blocks, each needs a 2x2x3 decomposition.
    # If 2x2x3 has rank r, total rank = 4*r.
    # With r=11 (4+7), total = 44.
    # With r=10, total = 40.
    #
    # The current program gets 77 = 7*11, which uses a different blocking.
    #
    # Let me try the direct 4-block approach with rank 11 for 2x2x3:
    # Total rank = 4 * 11 = 44.
    #
    # But we can do better by sharing across blocks.
    #
    # Let me try a completely different approach: direct rank-16 construction.
    #
    # For 4x4x6, the tensor has 4*4*6 = 96 entries to match.
    # Rank r means we need r rank-1 terms.
    #
    # Let me try to find a rank-16 solution using the structure.
    #
    # Approach: Use Strassen on the 4x4x4 part (rank 13 via BCL or 16 via Strassen^2)
    # and add 2 more terms for the extra columns.
    #
    # Strassen for 4x4x4: 2x2 blocks, each 2x2x2 uses rank 7.
    # 4 output blocks * 7 = 28. That's worse than BCL 13.
    #
    # Let me just build a valid solution and verify. I'll use the
    # 2x2x3 rank-11 approach with 4 blocks, giving rank 44.
    # Then try to reduce.

    # Build 2x2x3 rank-11 decomposition
    def build_2x2x3():
        rank_sub = 11
        U = [[0]*rank_sub for _ in range(4)]
        V = [[0]*rank_sub for _ in range(6)]
        W = [[0]*rank_sub for _ in range(6)]
        
        # First 4: schoolbook for 2x2x1 (columns 0,1 of B -> column 0 of C)
        # Actually, let me build 2x2x3 properly.
        # C (2x3) = A (2x2) * B (2x3)
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        #
        # Schoolbook: 12 terms. Can we do better?
        # Use Strassen for the first 2 columns (2x2x2, rank 7) + 4 schoolbook for col 2.
        # Total: 7 + 4 = 11.
        
        # Strassen for 2x2x2 (columns 0,1):
        # S1 = (A00+A11)(B00+B11)
        # S2 = (A10+A11)B01
        # S3 = A00(B10-B11)
        # S4 = A11(B01-B00)
        # S5 = (A00+A01)B10
        # S6 = (A10+A11)B11
        # S7 = (A01-A10)(B00+B10)
        #
        # C00 = S1 - S4 + S6
        # C01 = S2 + S4
        # C10 = S3 + S5
        # C11 = S1 - S3 + S7
        
        # Schoolbook for column 2:
        # T1 = A00*B02
        # T2 = A01*B12
        # T3 = A10*B02
        # T4 = A11*B12
        # C02 = T1 + T2
        # C12 = T3 + T4
        
        # U indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        # V indices: B[0,0]=0, B[0,1]=1, B[0,2]=2, B[1,0]=3, B[1,1]=4, B[1,2]=5
        # W indices: C[0,0]=0, C[0,1]=1, C[0,2]=2, C[1,0]=3, C[1,1]=4, C[1,2]=5
        
        # S1: (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[4][0] = 1; W[3][0] = -1; W[1][0] = 0
        # C00 += S1, C11 += S1
        # C00 = S1 - S4 + S6, so W[0][0] += 1
        # C11 = S1 - S3 + S7, so W[4][0] += 1
        
        # S2: (A10+A11)B01
        U[2][1] = 1; U[3][1] = 1
        V[1][1] = 1
        W[1][1] = 1
        # C01 += S2
        
        # S3: A00(B10-B11)
        U[0][2] = 1
        V[3][2] = 1; V[4][2] = -1
        W[3][2] = 1
        # C10 += S3
        
        # S4: A11(B01-B00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = -1; W[1][3] = 1
        # C00 -= S4, C01 += S4
        
        # S5: (A00+A01)B10
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[3][4] = 1
        # C10 += S5
        
        # S6: (A10+A11)B11
        U[2][5] = 1; U[3][5] = 1
        V[4][5] = 1
        W[0][5] = 1
        # C00 += S6
        
        # S7: (A01-A10)(B00+B10)
        U[1][6] = 1; U[2][6] = -1
        V[0][6] = 1; V[3][6] = 1
        W[4][6] = 1
        # C11 += S7
        
        # T1: A00*B02
        U[0][7] = 1
        V[2][7] = 1
        W[2][7] = 1
        # C02 += T1
        
        # T2: A01*B12
        U[1][8] = 1
        V[5][8] = 1
        W[2][8] = 1
        # C02 += T2
        
        # T3: A10*B02
        U[2][9] = 1
        V[2][9] = 1
        W[5][9] = 1
        # C12 += T3
        
        # T4: A11*B12
        U[3][10] = 1
        V[5][10] = 1
        W[5][10] = 1
        # C12 += T4
        
        return U, V, W

    # Build 4x4x6 from 2x2x3 sub-decomposition
    # 4x4 A = 4 blocks of 2x2: A00, A01, A10, A11
    # 4x6 B = 4 blocks of 2x3: B00, B01, B10, B11
    # 4x6 C = 4 blocks of 2x3: C00, C01, C10, C11
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Each Ci_j is a 2x3 matrix, computed from two 2x2x3 products.
    # So for each Ci_j, we need 2 * 11 = 22 rank terms.
    # Total: 4 * 22 = 88. That's worse than 77.
    #
    # The current program's approach is better. Let me just use the
    # existing structure but verify it works.
    #
    # Actually, let me try a different blocking.
    # 4x4 A, 4x6 B, 4x6 C.
    # Split A into 2 blocks of 2x4 (rows 0-1, rows 2-3).
    # Split B into 3 blocks of 4x2 (cols 0-1, cols 2-3, cols 4-5).
    # Split C into 6 blocks of 2x2 (row blocks 0-1, 2-3; col blocks 0-1, 2-3, 4-5).
    #
    # C[i,j] (2x2) = A[i,:] (2x4) * B[:,j] (4x2)
    # = A[i,0:2] * B[0:2,j] + A[i,2:4] * B[2:4,j]
    #
    # Each 2x4 * 4x2 -> 2x2 is a 2x2x2 problem (rank 7 Strassen).
    # 6 output blocks * 7 = 42.
    #
    # Wait, 2x4 * 4x2: A is 2x4, B is 4x2, C is 2x2.
    # This is equivalent to (2x2 block) * (2x2 block) where:
    # A = [A_left | A_right] (2x4 = two 2x2 blocks)
    # B = [B_top; B_bottom] (4x2 = two 2x2 blocks)
    # C = A_left * B_top + A_right * B_bottom
    #
    # This is a sum of two 2x2x2 products. Each can use Strassen rank 7.
    # Total: 6 * 14 = 84. Worse.
    #
    # Let me just use the current program's approach which gives 77.
    # The current program already works. Let me just return a valid
    # rank-77 decomposition.

    # Use the build_4x4x6 from the current program with rank-11 2x2x3
    U_sub, V_sub, W_sub = build_2x2x3()
    rank_sub = 11
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

    # Verify
    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j, :] * V_arr[j*p+k, :], W_arr[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False, (i, j, k, val)
        return True, None

    valid, info = verify(U, V, W, n, m, p)
    if not valid:
        # Fall back to schoolbook rank 96
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
