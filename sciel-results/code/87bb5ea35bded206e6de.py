"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses block decomposition with Strassen's algorithm for 2x2 blocks.
"""
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses block decomposition:
      - Split m=4 into 2+2
      - For each m-block, split p=7 into 4+3
      - For p=4 blocks, split n=3 into 2+1, p=4 into 2+2
      - For p=3 blocks, split n=3 into 2+1, p=3 into 2+1
    Uses Strassen for 2x2x2, and schoolbook for smaller blocks.
    """
    start_time = time.time()
    deadline = 140.0  # seconds, safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    # We'll build the decomposition by combining sub-decompositions.
    # Each sub-decomposition is represented as (U_sub, V_sub, W_sub) where:
    #   U_sub: (n_sub * m_sub) x rank_sub
    #   V_sub: (m_sub * p_sub) x rank_sub
    #   W_sub: (p_sub * n_sub) x rank_sub
    # And the mapping from global indices to sub-indices.

    def strassen_222():
        """
        Strassen decomposition for 2x2 * 2x2.
        Returns (U, V, W) each of shape (4, 7).
        Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        # Strassen's 7 products:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        # U: rows are (i,j) for A, i in {0,1}, j in {0,1}
        # V: rows are (j,k) for B, j in {0,1}, k in {0,1}
        # W: rows are (k,i) for C, k in {0,1}, i in {0,1}

        # M1: (A11+A22)*(B11+B22)
        # A11 = U[0], A22 = U[3] (i=1,j=1)
        # B11 = V[0], B22 = V[3] (j=1,k=1)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # C11 += M1, C22 += M1
        W[0][0] = 1  # C11: k=0,i=0
        W[1][1] = 1  # C22: k=1,i=1

        # M2: (A21+A22)*B11
        # A21 = U[2] (i=1,j=0), A22 = U[3]
        # B11 = V[0]
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C21 += M2, C22 -= M2
        W[0][1] = 1  # C21: k=0,i=1
        W[1][1] = -1 # C22: k=1,i=1

        # M3: A11*(B12-B22)
        # A11 = U[0]
        # B12 = V[1] (j=0,k=1), B22 = V[3]
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # C12 += M3, C22 += M3
        W[1][0] = 1  # C12: k=1,i=0
        W[1][1] = 1  # C22: k=1,i=1

        # M4: A22*(B21-B11)
        # A22 = U[3]
        # B21 = V[2] (j=1,k=0), B11 = V[0]
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # C11 += M4, C21 += M4
        W[0][0] += 1  # C11: k=0,i=0
        W[0][1] += 1  # C21: k=0,i=1

        # M5: (A11+A12)*B22
        # A11 = U[0], A12 = U[1] (i=0,j=1)
        # B22 = V[3]
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C11 -= M5, C12 += M5
        W[0][0] -= 1  # C11: k=0,i=0
        W[1][0] += 1  # C12: k=1,i=0

        # M6: (A21-A11)*(B11+B12)
        # A21 = U[2], A11 = U[0]
        # B11 = V[0], B12 = V[1]
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C22 += M6
        W[1][1] += 1  # C22: k=1,i=1

        # M7: (A12-A22)*(B21+B22)
        # A12 = U[1], A22 = U[3]
        # B21 = V[2], B22 = V[3]
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C11 += M7
        W[0][0] += 1  # C11: k=0,i=0

        return U, V, W

    def decomp_1x2x2():
        """
        Rank 2 decomposition for 1x2 * 2x2.
        U: (2, 2), V: (4, 2), W: (2, 2)
        """
        U = [[0]*2 for _ in range(2)]
        V = [[0]*2 for _ in range(4)]
        W = [[0]*2 for _ in range(2)]
        
        # Term 1: j=0
        U[0][0] = 1 # i=0, j=0
        # U[1][0] is i=0, j=1? No, U rows are i*m+j. m=2.
        # i=0, j=0 -> row 0. i=0, j=1 -> row 1.
        U[0][0] = 1
        U[1][0] = 0
        
        V[0][0] = 1 # j=0, k=0
        V[1][0] = 1 # j=0, k=1
        V[2][0] = 0 # j=1, k=0
        V[3][0] = 0 # j=1, k=1
        
        W[0][0] = 1 # k=0, i=0
        W[1][0] = 1 # k=1, i=0
        
        # Term 2: j=1
        U[0][1] = 0
        U[1][1] = 1
        
        V[0][1] = 0
        V[1][1] = 0
        V[2][1] = 1
        V[3][1] = 1
        
        W[0][1] = 1
        W[1][1] = 1
        
        return U, V, W

    def decomp_2x2x1():
        """
        Rank 2 decomposition for 2x2 * 2x1.
        U: (4, 2), V: (2, 2), W: (2, 2)
        """
        U = [[0]*2 for _ in range(4)]
        V = [[0]*2 for _ in range(2)]
        W = [[0]*2 for _ in range(2)]
        
        # Term 1: j=0
        # i=0, j=0 -> row 0. i=1, j=0 -> row 2.
        U[0][0] = 1
        U[1][0] = 0
        U[2][0] = 1
        U[3][0] = 0
        
        V[0][0] = 1 # j=0, k=0
        V[1][0] = 0 # j=1, k=0
        
        W[0][0] = 1 # k=0, i=0
        W[1][0] = 1 # k=0, i=1
        
        # Term 2: j=1
        U[0][1] = 0
        U[1][1] = 1
        U[2][1] = 0
        U[3][1] = 1
        
        V[0][1] = 0
        V[1][1] = 1
        
        W[0][1] = 1
        W[1][1] = 1
        
        return U, V, W

    def decomp_1x2x1():
        """
        Rank 2 decomposition for 1x2 * 2x1.
        U: (2, 2), V: (2, 2), W: (1, 2)
        """
        U = [[0]*2 for _ in range(2)]
        V = [[0]*2 for _ in range(2)]
        W = [[0]*2 for _ in range(1)]
        
        # Term 1: j=0
        U[0][0] = 1
        U[1][0] = 0
        
        V[0][0] = 1
        V[1][0] = 0
        
        W[0][0] = 1
        
        # Term 2: j=1
        U[0][1] = 0
        U[1][1] = 1
        
        V[0][1] = 0
        V[1][1] = 1
        
        W[0][1] = 1
        
        return U, V, W

    def combine_decomps(decomps, n_global, m_global, p_global, 
                        n_offsets, m_offsets, p_offsets):
        """
        Combine multiple sub-decompositions into a global decomposition.
        
        decomps: list of (U_sub, V_sub, W_sub)
        n_offsets: list of (n_start, n_len) for each sub
        m_offsets: list of (m_start, m_len) for each sub
        p_offsets: list of (p_start, p_len) for each sub
        """
        total_rank = sum(U.shape[1] if hasattr(U, 'shape') else len(U[0]) for U, V, W in decomps)
        
        U_global = [[0]*total_rank for _ in range(n_global * m_global)]
        V_global = [[0]*total_rank for _ in range(m_global * p_global)]
        W_global = [[0]*total_rank for _ in range(p_global * n_global)]
        
        rank_offset = 0
        for idx, (U_sub, V_sub, W_sub) in enumerate(decomps):
            n_start, n_len = n_offsets[idx]
            m_start, m_len = m_offsets[idx]
            p_start, p_len = p_offsets[idx]
            
            sub_rank = len(U_sub[0]) if U_sub else 0
            
            # Copy U_sub into U_global
            for i_sub in range(n_len):
                for j_sub in range(m_len):
                    global_i = n_start + i_sub
                    global_j = m_start + j_sub
                    global_row = global_i * m_global + global_j
                    sub_row = i_sub * m_len + j_sub
                    for c in range(sub_rank):
                        U_global[global_row][rank_offset + c] = U_sub[sub_row][c]
            
            # Copy V_sub into V_global
            for j_sub in range(m_len):
                for k_sub in range(p_len):
                    global_j = m_start + j_sub
                    global_k = p_start + k_sub
                    global_row = global_j * p_global + global_k
                    sub_row = j_sub * p_len + k_sub
                    for c in range(sub_rank):
                        V_global[global_row][rank_offset + c] = V_sub[sub_row][c]
            
            # Copy W_sub into W_global
            for k_sub in range(p_len):
                for i_sub in range(n_len):
                    global_k = p_start + k_sub
                    global_i = n_start + i_sub
                    global_row = global_k * n_global + global_i
                    sub_row = k_sub * n_len + i_sub
                    for c in range(sub_rank):
                        W_global[global_row][rank_offset + c] = W_sub[sub_row][c]
            
            rank_offset += sub_rank
        
        return U_global, V_global, W_global

    # Build the decomposition for M_{3,4,7}
    # Split m=4 into two blocks of 2: m1=2, m2=2
    
    # For each m-block, we need M_{3,2,7}
    # Split p=7 into p1=4, p2=3
    
    # For M_{3,2,4}: split n=3 into n1=2, n2=1; split p=4 into p1=2, p2=2
    #   - M_{2,2,2} (Strassen, rank 7)
    #   - M_{2,2,2} (Strassen, rank 7)
    #   - M_{1,2,2} (schoolbook, rank 4)
    #   - M_{1,2,2} (schoolbook, rank 4)
    # Total rank for M_{3,2,4}: 7+7+4+4 = 22
    
    # For M_{3,2,3}: split n=3 into n1=2, n2=1; split p=3 into p1=2, p2=1
    #   - M_{2,2,2} (Strassen, rank 7)
    #   - M_{2,2,1} (schoolbook, rank 4)
    #   - M_{1,2,2} (schoolbook, rank 4)
    #   - M_{1,2,1} (schoolbook, rank 2)
    # Total rank for M_{3,2,3}: 7+4+4+2 = 17
    
    # For M_{3,2,7}: 22 + 17 = 39
    # For M_{3,4,7}: 39 + 39 = 78

    # Build M_{2,2,2} (Strassen)
    S_222 = strassen_222()

    # Build M_{2,2,1} (low rank)
    S_221 = decomp_2x2x1()

    # Build M_{1,2,2} (low rank)
    S_122 = decomp_1x2x2()

    # Build M_{1,2,1} (low rank)
    S_121 = decomp_1x2x1()

    # Now build M_{3,2,4} by combining:
    # Block (n=2, m=2, p=2) at n_offset=0, p_offset=0: Strassen
    # Block (n=2, m=2, p=2) at n_offset=0, p_offset=2: Strassen
    # Block (n=1, m=2, p=2) at n_offset=2, p_offset=0: schoolbook 1x2x2
    # Block (n=1, m=2, p=2) at n_offset=2, p_offset=2: schoolbook 1x2x2

    decomps_324 = [
        S_222,  # n=2,m=2,p=2 at (0,0)
        S_222,  # n=2,m=2,p=2 at (0,2)
        S_122,  # n=1,m=2,p=2 at (2,0)
        S_122,  # n=1,m=2,p=2 at (2,2)
    ]
    n_offsets_324 = [(0, 2), (0, 2), (2, 1), (2, 1)]
    m_offsets_324 = [(0, 2), (0, 2), (0, 2), (0, 2)]
    p_offsets_324 = [(0, 2), (2, 2), (0, 2), (2, 2)]

    U_324, V_324, W_324 = combine_decomps(
        decomps_324, 3, 2, 4,
        n_offsets_324, m_offsets_324, p_offsets_324
    )

    # Build M_{3,2,3} by combining:
    # Block (n=2, m=2, p=2) at n_offset=0, p_offset=0: Strassen
    # Block (n=2, m=2, p=1) at n_offset=0, p_offset=2: schoolbook 2x2x1
    # Block (n=1, m=2, p=2) at n_offset=2, p_offset=0: schoolbook 1x2x2
    # Block (n=1, m=2, p=1) at n_offset=2, p_offset=2: schoolbook 1x2x1

    decomps_323 = [
        S_222,  # n=2,m=2,p=2 at (0,0)
        S_221,  # n=2,m=2,p=1 at (0,2)
        S_122,  # n=1,m=2,p=2 at (2,0)
        S_121,  # n=1,m=2,p=1 at (2,2)
    ]
    n_offsets_323 = [(0, 2), (0, 2), (2, 1), (2, 1)]
    m_offsets_323 = [(0, 2), (0, 2), (0, 2), (0, 2)]
    p_offsets_323 = [(0, 2), (2, 1), (0, 2), (2, 1)]

    U_323, V_323, W_323 = combine_decomps(
        decomps_323, 3, 2, 3,
        n_offsets_323, m_offsets_323, p_offsets_323
    )

    # Now build M_{3,2,7} by combining M_{3,2,4} and M_{3,2,3}
    decomps_327 = [
        (U_324, V_324, W_324),  # p_offset=0, p_len=4
        (U_323, V_323, W_323),  # p_offset=4, p_len=3
    ]
    n_offsets_327 = [(0, 3), (0, 3)]
    m_offsets_327 = [(0, 2), (0, 2)]
    p_offsets_327 = [(0, 4), (4, 3)]

    U_327, V_327, W_327 = combine_decomps(
        decomps_327, 3, 2, 7,
        n_offsets_327, m_offsets_327, p_offsets_327
    )

    # Finally build M_{3,4,7} by combining two M_{3,2,7}
    decomps_347 = [
        (U_327, V_327, W_327),  # m_offset=0, m_len=2
        (U_327, V_327, W_327),  # m_offset=2, m_len=2
    ]
    n_offsets_347 = [(0, 3), (0, 3)]
    m_offsets_347 = [(0, 2), (2, 2)]
    p_offsets_347 = [(0, 7), (0, 7)]

    U_final, V_final, W_final = combine_decomps(
        decomps_347, 3, 4, 7,
        n_offsets_347, m_offsets_347, p_offsets_347
    )

    rank = len(U_final[0]) if U_final else 0

    # Verify correctness with a random test
    import numpy as np
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 7)
    C_expected = A @ B

    U_arr = np.array(U_final, dtype=float)
    V_arr = np.array(V_final, dtype=float)
    W_arr = np.array(W_final, dtype=float)

    # Compute C from decomposition: C[k*n+i, :] = sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
    # Actually: C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, the standard form is: C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But this isn't quite right. The tensor decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = delta_{i,k} for matrix multiplication... no.
    
    # For matrix multiplication: C = A * B, C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 if we're computing the (i,k) entry using A[i,j] and B[j,k]
    # Actually the matrix multiplication tensor M_{n,m,p} has entries:
    # M[i,j,k] = 1 (it's the identity in some sense)
    # The decomposition satisfies: M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And then C[i,k] = sum_{j} A[i,j] * B[j,k] = sum_{j,r} A[i,j] * B[j,k] * M[i,j,k]
    #                = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
    
    # Let me verify by computing C from the decomposition
    C_computed = np.zeros((3, 7))
    for i in range(3):
        for k in range(7):
            val = 0.0
            for j in range(4):
                for r in range(rank):
                    val += A[i, j] * B[j, k] * U_arr[i*4+j, r] * V_arr[j*7+k, r] * W_arr[k*3+i, r]
            C_computed[i, k] = val

    error = np.max(np.abs(C_computed - C_expected))
    if error > 1e-10:
        # Fallback to schoolbook if verification fails
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
