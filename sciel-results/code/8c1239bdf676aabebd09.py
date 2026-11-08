import time

_DEADLINE = 115.0
_START = None


def construct():
    global _START
    if _START is None:
        _START = time.time()

    n, m, p = 2, 3, 9
    # Target rank 49: save 1 rank by using 1 rank-6 block for 2x2x3 and 1 rank-7 block for 2x2x3

    U = [[0] * 49 for _ in range(n * m)]
    V = [[0] * 49 for _ in range(m * p)]
    W = [[0] * 49 for _ in range(p * n)]

    col = 0

    # Part 1: A_L (2x2) * B_T (2x9)
    # Columns 0,1 -> 2x2x2 Strassen (rank 7)
    # Columns 2,3 -> 2x2x2 Strassen (rank 7)
    # Columns 4,5 -> 2x2x2 Strassen (rank 7)
    # Columns 6,7,8 -> 2x2x3 rank 6 block
    # Total for A_L*B_T: 7+7+7+6 = 27

    # Strassen block for columns (k0, k1)
    def strassen_block(k0, k1, col_start):
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1

        u_a = 0
        u_b = 1
        u_c = 3
        u_d = 4

        # M1 = (a+d)(e+h)
        U[u_a][col_start] += 1
        U[u_d][col_start] += 1
        V[v_e][col_start] += 1
        V[v_h][col_start] += 1

        # M2 = (c+d)e
        U[u_c][col_start+1] += 1
        U[u_d][col_start+1] += 1
        V[v_e][col_start+1] += 1

        # M3 = a(f-h)
        U[u_a][col_start+2] += 1
        V[v_f][col_start+2] += 1
        V[v_h][col_start+2] -= 1

        # M4 = d(g-e)
        U[u_d][col_start+3] += 1
        V[v_g][col_start+3] += 1
        V[v_e][col_start+3] -= 1

        # M5 = (a+b)h
        U[u_a][col_start+4] += 1
        U[u_b][col_start+4] += 1
        V[v_h][col_start+4] += 1

        # M6 = (c-a)(e+f)
        U[u_c][col_start+5] += 1
        U[u_a][col_start+5] -= 1
        V[v_e][col_start+5] += 1
        V[v_f][col_start+5] += 1

        # M7 = (b-d)(g+h)
        U[u_b][col_start+6] += 1
        U[u_d][col_start+6] -= 1
        V[v_g][col_start+6] += 1
        V[v_h][col_start+6] += 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # C00 = M1 + M4 - M5 + M7
        W[w_00][col_start] += 1
        W[w_00][col_start+3] += 1
        W[w_00][col_start+4] -= 1
        W[w_00][col_start+6] += 1

        # C01 = M3 + M5
        W[w_01][col_start+2] += 1
        W[w_01][col_start+4] += 1

        # C10 = M2 + M4
        W[w_10][col_start+1] += 1
        W[w_10][col_start+3] += 1

        # C11 = M1 - M2 + M3 + M6
        W[w_11][col_start] += 1
        W[w_11][col_start+1] -= 1
        W[w_11][col_start+2] += 1
        W[w_11][col_start+5] += 1

    # Three 2x2x2 Strassen blocks for columns (0,1), (2,3), (4,5)
    strassen_block(0, 1, 0)
    strassen_block(2, 3, 7)
    strassen_block(4, 5, 14)

    # Rank-6 block for 2x2x3 (columns 6,7,8)
    # We need: C_{0,6}, C_{0,7}, C_{0,8}, C_{1,6}, C_{1,7}, C_{1,8}
    # C_{i,k} = A_{i0}*B_{0k} + A_{i1}*B_{1k}
    # This is a 2x2x3 tensor. Rank lower bound: max(4, 6, 4) = 6? Actually for 2x2xn, rank is n+1 for n>=1? No.
    # For 2x2x2, rank 7. For 2x2x1, rank 4. For 2x2x3, let's try rank 6.
    # Actually, the rank of the multiplication tensor M_{2,2,n} is known: for n=1, rank 4; n=2, rank 7; n=3, rank 10? 
    # Wait, let me think. The tensor is sum_{i,j,k} A_{ij} B_{jk} C_{ik}. For fixed i,k, it's sum_j A_{ij} B_{jk}.
    # This is a linear map from A ⊗ B to C. The rank is the tensor rank.
    # For 2x2x3, I'm not sure of the exact minimal rank. Let me just use rank 7 to be safe, but that gives 7+7+7+7=28+18=46? No, 28+18=46 is less than 50.
    # Wait: 4 Strassen blocks of 7 = 28. Plus 18 for the 2x1x9 part. Total 46. That's already better than 50!
    # But wait, the original had 4 blocks of 2x2x2 (cols 0-1, 2-3, 4-5, 6-7) = 28, plus 2x2x1 (col 8) = 4, total 32. Plus 18 = 50.
    # If I use 3 blocks of 2x2x2 (cols 0-1, 2-3, 4-5) = 21, plus 1 block of 2x2x3 (cols 6-8), I need rank for 2x2x3.
    # The direct rank for 2x2x3 is 2*2*3=12. Can I do better?
    # Actually, the tensor M_{2,2,3} has entries T[i,j,k] = 1 if i,j,k are all in range and we're looking at the multiplication tensor.
    # Wait no. The multiplication tensor for (2x2)*(2x3) -> (2x3) has indices (i,j) for A, (j,k) for B, (i,k) for C.
    # So T[(i,j),(j,k),(i,k)] = 1.
    # This is a 3-tensor of size 4x6x6.
    # The rank of this tensor... I think it's 10 for 2x2x3. Let me just use 7 for safety (same as 2x2x2) but that won't save rank.
    # Actually, let me just use the straightforward approach: 4 blocks of 2x2x2 = 28, and 1 block of 2x2x1 = 4. Total 32.
    # That's the original. To save rank, I need a better decomposition of the last part.
    # 
    # Alternative: use a single 2x2x9 block. What's the rank of M_{2,2,9}?
    # The direct rank is 2*2*9 = 36. Strassen-based: split into 4x2x2 + 1x2x2 = 4*7+4 = 32.
    # Can we do better? The rank of M_{2,2,n} is known to be 2n+2 for n even? Or something like that.
    # Actually, I recall that rank(M_{2,2,n}) = 2n+1 for n odd, 2n for n even? Not sure.
    # Let me just stick with the known decomposition and try to save 1 rank elsewhere.
    #
    # Actually, let me reconsider. The 2x1x9 part has rank 18 (direct). Can we save there?
    # M_{2,1,9}: T[(i,0),(0,k),(i,k)] = 1. This is just A_i * B_k * C_{i,k}.
    # This decomposes as sum_{i,k} A_i B_k C_{i,k}. Rank is 2*9 = 18. Minimal.
    #
    # So the only place to save is in the 2x2x9 part. Current: 32.
    # If rank(M_{2,2,9}) = 31, we save 1.
    # I'll try a construction with rank 31 for the 2x2x9 part.
    #
    # Let me use: 3 Strassen blocks (2x2x2) for cols 0-5 = 21, and 1 block of 2x2x3 for cols 6-8.
    # If I can do 2x2x3 in rank 10, total is 31.
    # Rank 10 for 2x2x3: Let me try to construct it.
    # C_{i,k} = A_{i0}B_{0k} + A_{i1}B_{1k} for i=0,1; k=0,1,2.
    # 
    # Let me use a different approach. I'll just use rank 7 for the last block too (treating col 8 as 2x2x2 with col 8 duplicated or something). No, that's wrong.
    #
    # Actually, the simplest safe approach: keep rank 50 but make the code cleaner. The previous rejection was for "empty" reason, which might mean the code had a bug.
    # Let me carefully re-verify the rank 50 solution.

    # Let me redo this more carefully. I'll use the known rank-50 construction.

    # Actually, let me try rank 49 with a specific construction.
    # I'll use:
    # - 4 Strassen blocks of 2x2x2 for columns (0,1),(2,3),(4,5),(6,7): rank 28
    # - 1 block of 2x2x1 for column 8: rank 4
    # - 1 block of 2x1x9: rank 18
    # Total: 50.
    #
    # To get 49, I need to save 1 rank. Let me try merging the last 2x2x1 block with the 2x1x9 block.
    # The 2x2x1 block handles A_{00},A_{01},A_{10},A_{11} times B_{08},B_{18} to give C_{08},C_{18}.
    # The 2x1x9 block handles A_{02},A_{12} times B_{2k} for all k.
    # These are independent, so I can't easily merge them.
    #
    # I'll just output the rank 50 solution cleanly.
    
    # Reset U, V, W
    U = [[0] * 50 for _ in range(n * m)]
    V = [[0] * 50 for _ in range(m * p)]
    W = [[0] * 50 for _ in range(p * n)]

    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        strassen_block(k0, k1, col)
        col += 7

    # 2x2x1 block for column 8
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1

    col += 4

    # 2x1x9 block
    for k in range(9):
        v_b = 2 * p + k
        U[2][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1

        U[5][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1

        col += 2

    return {"U": U, "V": V, "W": W}
