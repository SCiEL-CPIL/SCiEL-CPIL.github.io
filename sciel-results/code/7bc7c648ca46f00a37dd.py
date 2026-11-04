"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 2, 4, 6

    # Try to find a low-rank decomposition using numerical optimization
    # First, try rank 24 (half of schoolbook 48)
    
    def verify_decomp(U_list, V_list, W_list, rnk):
        U_np = np.array(U_list, dtype=float)
        V_np = np.array(V_list, dtype=float)
        W_np = np.array(W_list, dtype=float)
        U_reshaped = U_np.reshape(n, m, rnk)
        V_reshaped = V_np.reshape(m, p, rnk)
        W_reshaped = W_np.reshape(p, n, rnk)
        result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        if not np.allclose(result, np.ones((n, m, p)), atol=1e-7):
            return False
        return True

    # Build rank-42 Strassen decomposition as fallback
    def build_strassen_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_strassen_block(a_idx, b_idx, w_idx, start_col):
            c = start_col
            U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
            V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
            W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
            U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
            V[b_idx[0]][c+1] += 1
            W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
            U[a_idx[0]][c+2] += 1
            V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
            W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
            U[a_idx[3]][c+3] += 1
            V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
            W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
            U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
            V[b_idx[3]][c+4] += 1
            W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
            U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
            V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
            W[w_idx[3]][c+5] += 1
            U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
            V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
            W[w_idx[0]][c+6] += 1

        a_L = [0, 1, 4, 5]
        a_R = [2, 3, 6, 7]

        b_1_T = [0, 1, 6, 7]
        b_1_B = [12, 13, 18, 19]
        w_1 = [0, 2, 1, 3]

        add_strassen_block(a_L, b_1_T, w_1, 0)
        add_strassen_block(a_R, b_1_B, w_1, 7)

        b_2_T = [2, 3, 8, 9]
        b_2_B = [14, 15, 20, 21]
        w_2 = [4, 6, 5, 7]

        add_strassen_block(a_L, b_2_T, w_2, 14)
        add_strassen_block(a_R, b_2_B, w_2, 21)

        b_3_T = [4, 5, 10, 11]
        b_3_B = [16, 17, 22, 23]
        w_3 = [8, 10, 9, 11]

        add_strassen_block(a_L, b_3_T, w_3, 28)
        add_strassen_block(a_R, b_3_B, w_3, 35)

        return U, V, W, rank

    # Try Kronecker product approach: rank 24
    # M_{2,4,6} can be decomposed using block structure
    # A (2x4) as 2x2 blocks of 1x2, B (4x6) as 2x2 blocks of 2x3
    # C (2x6) as 2x2 blocks of 1x3
    
    # The 2x2 block mult uses Strassen (rank 7)
    # Each entry mult is 1x2 * 2x3 = rank 6
    # But we can share: total rank = 7 * 6 = 42 (same as before)
    
    # Try direct rank-24 construction via structured ansatz
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # We want: C[i,k] = sum_r U[i,j]*V[j,k]*W[k,i] for some j
    
    # Let's try a different approach: use the fact that n=2 is small
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    # Try rank 24 with specific structure
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Structure: for each k (0..5) and i (0..1), we have 2 terms
    # Total: 6 * 2 * 2 = 24 terms
    # Term (i, k, s) where s in {0,1}
    # U[i, j] contributes to term (i,k,s) if j is in the right range
    # V[j, k] contributes to term (i,k,s)
    # W[k, i] = 1 for all
    
    # Actually, let's use a cleaner approach:
    # Group by k: for each k, we need C[0,k] and C[1,k]
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a 2x4 * 4x1 multiplication for each k
    # Rank of M_{2,4,1} is 8 (schoolbook) or less?
    # Actually M_{2,4,1} has rank 8 (it's just dot products)
    # So 6 k's * 8 = 48... that's schoolbook
    
    # Let me try a different grouping. For each pair of k values...
    # Or use the Kronecker structure more carefully.
    
    # Let's just try to solve for rank 24 numerically
    # and see if we can find a valid decomposition
    
    # Actually, let's try a known result: the border rank of M_{2,4,6}
    # is at most 36 (from Landsberg's bounds), and the actual rank might be 36 or less.
    
    # For now, let's stick with the proven rank-42 Strassen decomposition
    # and verify it works correctly.
    
    U, V, W, rank = build_strassen_42()
    
    if not verify_decomp(U, V, W, rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
