"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses block decomposition with Strassen's algorithm for 2x2 blocks.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses block decomposition with Strassen for 2x2x2 subproblems.
    """
    start_time = time.time()
    deadline = 140.0

    n, m, p = DIMENSIONS  # 3, 4, 7

    def strassen_222():
        """
        Strassen decomposition for 2x2 * 2x2.
        Returns (U, V, W) each of shape (4, 7).
        Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # M1 = (A11+A22)*(B11+B22)
        U[0][0] = 1; U[3][0] = 1  # A11=row0, A22=row3
        V[0][0] = 1; V[3][0] = 1  # B11=row0, B22=row3
        # C11 += M1, C22 += M1
        W[0][0] += 1  # C[0,0]
        W[3][0] += 1  # C[1,1]

        # M2 = (A21+A22)*B11
        U[2][1] = 1; U[3][1] = 1  # A21=row2, A22=row3
        V[0][1] = 1  # B11=row0
        # C21 += M2, C22 -= M2
        W[2][1] += 1  # C[1,0]
        W[3][1] += -1 # C[1,1]

        # M3 = A11*(B12-B22)
        U[0][2] = 1  # A11=row0
        V[1][2] = 1; V[3][2] = -1  # B12=row1, B22=row3
        # C12 += M3, C22 += M3
        W[1][2] += 1  # C[0,1]
        W[3][2] += 1  # C[1,1]

        # M4 = A22*(B21-B11)
        U[3][3] = 1  # A22=row3
        V[2][3] = 1; V[0][3] = -1  # B21=row2, B11=row0
        # C11 += M4, C21 += M4
        W[0][3] += 1  # C[0,0]
        W[2][3] += 1  # C[1,0]

        # M5 = (A11+A12)*B22
        U[0][4] = 1; U[1][4] = 1  # A11=row0, A12=row1
        V[3][4] = 1  # B22=row3
        # C11 -= M5, C12 += M5
        W[0][4] += -1 # C[0,0]
        W[1][4] += 1  # C[0,1]

        # M6 = (A21-A11)*(B11+B12)
        U[2][5] = 1; U[0][5] = -1  # A21=row2, A11=row0
        V[0][5] = 1; V[1][5] = 1  # B11=row0, B12=row1
        # C22 += M6
        W[3][5] += 1  # C[1,1]

        # M7 = (A12-A22)*(B21+B22)
        U[1][6] = 1; U[3][6] = -1  # A12=row1, A22=row3
        V[2][6] = 1; V[3][6] = 1  # B21=row2, B22=row3
        # C11 += M7
        W[0][6] += 1  # C[0,0]

        return U, V, W

    def schoolbook(n_sub, m_sub, p_sub):
        """
        Schoolbook decomposition for n_sub x m_sub * m_sub x p_sub.
        Rank = n_sub * m_sub * p_sub
        """
        rank = n_sub * m_sub * p_sub
        U = [[0]*rank for _ in range(n_sub * m_sub)]
        V = [[0]*rank for _ in range(m_sub * p_sub)]
        W = [[0]*rank for _ in range(p_sub * n_sub)]
        
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[i * m_sub + j][col] = 1
                    V[j * p_sub + k][col] = 1
                    W[k * n_sub + i][col] = 1
                    col += 1
        
        return U, V, W

    def combine_decomps(decomps, n_global, m_global, p_global, 
                        n_offsets, m_offsets, p_offsets):
        """
        Combine multiple sub-decompositions into a global decomposition.
        """
        total_rank = sum(len(U[0]) if U else 0 for U, V, W in decomps)
        
        U_global = [[0]*total_rank for _ in range(n_global * m_global)]
        V_global = [[0]*total_rank for _ in range(m_global * p_global)]
        W_global = [[0]*total_rank for _ in range(p_global * n_global)]
        
        rank_offset = 0
        for idx, (U_sub, V_sub, W_sub) in enumerate(decomps):
            n_start, n_len = n_offsets[idx]
            m_start, m_len = m_offsets[idx]
            p_start, p_len = p_offsets[idx]
            
            sub_rank = len(U_sub[0]) if U_sub else 0
            
            for i_sub in range(n_len):
                for j_sub in range(m_len):
                    global_row = (n_start + i_sub) * m_global + (m_start + j_sub)
                    sub_row = i_sub * m_len + j_sub
                    for c in range(sub_rank):
                        U_global[global_row][rank_offset + c] = U_sub[sub_row][c]
            
            for j_sub in range(m_len):
                for k_sub in range(p_len):
                    global_row = (m_start + j_sub) * p_global + (p_start + k_sub)
                    sub_row = j_sub * p_len + k_sub
                    for c in range(sub_rank):
                        V_global[global_row][rank_offset + c] = V_sub[sub_row][c]
            
            for k_sub in range(p_len):
                for i_sub in range(n_len):
                    global_row = (p_start + k_sub) * n_global + (n_start + i_sub)
                    sub_row = k_sub * n_len + i_sub
                    for c in range(sub_rank):
                        W_global[global_row][rank_offset + c] = W_sub[sub_row][c]
            
            rank_offset += sub_rank
        
        return U_global, V_global, W_global

    # Build base decompositions
    S_222 = strassen_222()  # rank 7
    S_221 = schoolbook(2, 2, 1)  # rank 4
    S_122 = schoolbook(1, 2, 2)  # rank 4
    S_121 = schoolbook(1, 2, 1)  # rank 2

    # Build M_{3,2,4}: split n=3 into 2+1, p=4 into 2+2
    # Blocks: (2,2,2)Strassen, (2,2,2)Strassen, (1,2,2)school, (1,2,2)school
    # Rank: 7+7+4+4 = 22
    decomps_324 = [S_222, S_222, S_122, S_122]
    n_off_324 = [(0,2), (0,2), (2,1), (2,1)]
    m_off_324 = [(0,2), (0,2), (0,2), (0,2)]
    p_off_324 = [(0,2), (2,2), (0,2), (2,2)]
    U_324, V_324, W_324 = combine_decomps(decomps_324, 3, 2, 4, n_off_324, m_off_324, p_off_324)

    # Build M_{3,2,3}: split n=3 into 2+1, p=3 into 2+1
    # Blocks: (2,2,2)Strassen, (2,2,1)school, (1,2,2)school, (1,2,1)school
    # Rank: 7+4+4+2 = 17
    decomps_323 = [S_222, S_221, S_122, S_121]
    n_off_323 = [(0,2), (0,2), (2,1), (2,1)]
    m_off_323 = [(0,2), (0,2), (0,2), (0,2)]
    p_off_323 = [(0,2), (2,1), (0,2), (2,1)]
    U_323, V_323, W_323 = combine_decomps(decomps_323, 3, 2, 3, n_off_323, m_off_323, p_off_323)

    # Build M_{3,2,7}: combine M_{3,2,4} and M_{3,2,3}
    # Rank: 22 + 17 = 39
    decomps_327 = [(U_324, V_324, W_324), (U_323, V_323, W_323)]
    n_off_327 = [(0,3), (0,3)]
    m_off_327 = [(0,2), (0,2)]
    p_off_327 = [(0,4), (4,3)]
    U_327, V_327, W_327 = combine_decomps(decomps_327, 3, 2, 7, n_off_327, m_off_327, p_off_327)

    # Build M_{3,4,7}: combine two M_{3,2,7}
    # Rank: 39 + 39 = 78
    decomps_347 = [(U_327, V_327, W_327), (U_327, V_327, W_327)]
    n_off_347 = [(0,3), (0,3)]
    m_off_347 = [(0,2), (2,2)]
    p_off_347 = [(0,7), (0,7)]
    U_final, V_final, W_final = combine_decomps(decomps_347, 3, 4, 7, n_off_347, m_off_347, p_off_347)

    rank = len(U_final[0]) if U_final else 0

    # Verify correctness
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 7)
    C_expected = A @ B

    U_arr = np.array(U_final, dtype=float)
    V_arr = np.array(V_final, dtype=float)
    W_arr = np.array(W_final, dtype=float)

    C_computed = np.zeros((3, 7))
    for i in range(3):
        for k in range(7):
            val = 0.0
            for j in range(4):
                for r in range(rank):
                    val += A[i, j] * B[j, k] * U_arr[i*4+j, r] * V_arr[j*7+k, r] * W_arr[k*3+i, r]
            C_computed[i, k] = val

    error = np.max(np.abs(C_computed - C_expected))
    if error > 1e-9:
        # Fallback to schoolbook
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
        
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
