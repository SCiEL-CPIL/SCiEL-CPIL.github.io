"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    schoolbook_rank = n * m * p  # 105

    # Laderman decomposition of 3x3x3 (rank 23)
    # Terms: (a_row, b_col, c_row, c_col, coeff)
    # We'll use the standard Laderman 23-term decomposition
    # A is 3x3, B is 3x3, C = A*B is 3x3
    # Each term is: C[i,j] += coeff * (sum over rows of A) * (sum over cols of B)
    
    # Laderman 23 terms for 3x3 matrix multiplication
    # Format: (list of (i, coeff) for A rows, list of (j, coeff) for B cols, (ci, cj) for C position)
    # Actually, let's use the standard form where each term is:
    # product = (linear combo of A entries) * (linear combo of B entries)
    # and contributes to C[i,j]
    
    # Standard Laderman decomposition:
    laderman_terms = [
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        ([(0,0,1)], [(0,0,1)], (0,0)),
        # Term 1: A[1,1]*B[1,1] -> C[1,1]
        ([(1,1,1)], [(1,1,1)], (1,1)),
        # Term 2: A[2,2]*B[2,2] -> C[2,2]
        ([(2,2,1)], [(2,2,1)], (2,2)),
        # Term 3: A[0,1]*B[1,0] -> C[0,0]
        ([(0,1,1)], [(1,0,1)], (0,0)),
        # Term 4: A[1,0]*B[0,1] -> C[0,1]
        ([(1,0,1)], [(0,1,1)], (0,1)),
        # Term 5: A[1,2]*B[2,1] -> C[1,1]
        ([(1,2,1)], [(2,1,1)], (1,1)),
        # Term 6: A[2,1]*B[1,2] -> C[1,2]
        ([(2,1,1)], [(1,2,1)], (1,2)),
        # Term 7: A[2,0]*B[0,2] -> C[2,2]
        ([(2,0,1)], [(0,2,1)], (2,2)),
        # Term 8: (A[0,0]+A[1,1])*B[2,2] -> C[2,2]
        ([(0,0,1),(1,1,1)], [(2,2,1)], (2,2)),
        # Term 9: (A[0,0]+A[2,2])*B[1,1] -> C[1,1]
        ([(0,0,1),(2,2,1)], [(1,1,1)], (1,1)),
        # Term 10: (A[1,1]+A[2,2])*B[0,0] -> C[0,0]
        ([(1,1,1),(2,2,1)], [(0,0,1)], (0,0)),
        # Term 11: (A[0,1]+A[1,0])*B[1,2] -> C[1,2]
        ([(0,1,1),(1,0,1)], [(1,2,1)], (1,2)),
        # Term 12: (A[1,2]+A[2,1])*B[2,0] -> C[2,0]
        ([(1,2,1),(2,1,1)], [(2,0,1)], (2,0)),
        # Term 13: (A[2,0]+A[0,2])*B[0,1] -> C[0,1]
        ([(2,0,1),(0,2,1)], [(0,1,1)], (0,1)),
        # Term 14: (A[0,2]+A[2,1])*B[1,0] -> C[1,0]
        ([(0,2,1),(2,1,1)], [(1,0,1)], (1,0)),
        # Term 15: (A[1,0]+A[0,1])*B[2,2] -> C[2,2] -- wait, let me be more careful
        # Let me use a verified Laderman decomposition
    ]
    
    # I'll use a well-known Laderman 23-term decomposition.
    # The terms are of the form: sum_{r} a_r * A[i_r, j_r] times sum_{c} b_c * B[i_c, j_c] -> C[p, q]
    # Let me use the standard published set.
    
    # Actually, let me just verify with a simple approach: 
    # I'll define the Laderman terms carefully and verify them numerically.
    
    # Standard Laderman 23 terms (one common formulation):
    # Each term: (A_factors, B_factors, C_pos)
    # where A_factors is list of (row, col, coeff), B_factors is list of (row, col, coeff)
    
    LADERMAN = [
        # 1
        ([(0,0,1)], [(0,0,1)], (0,0)),
        # 2
        ([(1,1,1)], [(1,1,1)], (1,1)),
        # 3
        ([(2,2,1)], [(2,2,1)], (2,2)),
        # 4
        ([(0,1,1)], [(1,0,1)], (0,0)),
        # 5
        ([(1,0,1)], [(0,1,1)], (0,1)),
        # 6
        ([(1,2,1)], [(2,1,1)], (1,1)),
        # 7
        ([(2,1,1)], [(1,2,1)], (1,2)),
        # 8
        ([(2,0,1)], [(0,2,1)], (2,2)),
        # 9
        ([(0,0,1),(1,1,1)], [(2,2,1)], (2,2)),
        # 10
        ([(0,0,1),(2,2,1)], [(1,1,1)], (1,1)),
        # 11
        ([(1,1,1),(2,2,1)], [(0,0,1)], (0,0)),
        # 12
        ([(0,1,1),(1,0,1)], [(1,2,1)], (1,2)),
        # 13
        ([(1,2,1),(2,1,1)], [(2,0,1)], (2,0)),
        # 14
        ([(2,0,1),(0,2,1)], [(0,1,1)], (0,1)),
        # 15
        ([(0,2,1),(2,1,1)], [(1,0,1)], (1,0)),
        # 16
        ([(1,0,1),(0,1,1)], [(2,2,1)], (2,2)),  # Need to verify
        # 17
        ([(0,2,1),(1,2,1)], [(2,0,1)], (2,0)),
        # 18
        ([(2,0,1),(2,1,1)], [(0,2,1)], (2,2)),
        # 19
        ([(0,0,1),(0,1,1)], [(2,2,1)], (2,2)),
        # 20
        ([(1,1,1),(1,2,1)], [(0,0,1)], (0,0)),
        # 21
        ([(2,2,1),(2,0,1)], [(1,1,1)], (1,1)),
        # 22
        ([(0,0,1),(1,0,1)], [(2,2,1)], (2,2)),
        # 23
        ([(1,1,1),(2,1,1)], [(0,0,1)], (0,0)),
    ]
    
    # This is getting complicated and error-prone. Let me use a different approach.
    # I'll use the known fact that rank(3x3x3) <= 23 and try to construct it programmatically
    # or use a simpler but still improved decomposition.
    
    # Alternative: Use Strassen for 2x2x2 blocks where possible.
    # Partition 3x5x7 as follows:
    # m=5 = 2+3, p=7 = 2+5, n=3 = 2+1
    # Or: m=5=3+2, p=7=2+2+3
    # 
    # Let's partition:
    # n=3: split into 2+1
    # m=5: split into 2+3  
    # p=7: split into 2+2+3
    #
    # Blocks:
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:5]*B[2:5, 0:2]
    # This is getting complex. Let me use a cleaner partition.
    
    # Simpler approach: Use Strassen for 2x2x2 sub-blocks
    # Partition m=5 into 2+2+1, p=7 into 2+2+2+1, n=3 into 2+1
    # 
    # Actually, let me just use a verified small-rank decomposition for the whole thing
    # by leveraging Strassen 2x2x2 where the sub-problems allow it.
    
    # Let me try: partition A as 3x5 into rows {0,1} and {2}, cols {0,1},{2,3},{4}
    # B as 5x7 into rows {0,1},{2,3},{4}, cols {0,1},{2,3},{4,5},{6}
    
    # This is too ad-hoc. Let me use the most reliable improvement:
    # Replace the 2x2x2 sub-blocks with Strassen (rank 7 vs 8)
    
    # Partition:
    # n=3 = 2+1
    # m=5 = 2+2+1
    # p=7 = 2+2+2+1
    
    # The 2x2x2 sub-problems can use Strassen.
    # Number of 2x2x2 blocks: for each (n_block, m_block, p_block) where all are 2-sized
    # n_blocks: 1 (the 2-sized one)
    # m_blocks: 2 (the two 2-sized ones)  
    # p_blocks: 3 (the three 2-sized ones)
    # So 1*2*3 = 6 blocks of 2x2x2, each can use Strassen (rank 7) instead of schoolbook (rank 8)
    # Savings: 6 * (8-7) = 6
    # Total rank: 105 - 6 = 99
    
    # But we also have mixed blocks that are 2x1x2, 1x2x2, etc.
    # Let me just implement the full decomposition with Strassen for 2x2x2 sub-blocks.
    
    # Actually, the simplest correct improvement: use Strassen for ALL 2x2x2 sub-multiplications
    # in the partitioned scheme.
    
    # Let me use a cleaner partition:
    # n=3: [0,1], [2]  (sizes 2, 1)
    # m=5: [0,1], [2,3], [4]  (sizes 2, 2, 1)
    # p=7: [0,1], [2,3], [4,5], [6]  (sizes 2, 2, 2, 1)
    
    # C[i_block, j_block] = sum over k_block of A[i_block, k_block] * B[k_block, j_block]
    
    # For 2x2x2 sub-problems, use Strassen.
    # For others, use schoolbook.
    
    deadline = time.time() + 110  # Leave margin before 120s kill
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        # Strassen 7 products for 2x2 matrix multiplication
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # p1 = (a11+a22)(b11+b22)
        # p2 = (a21+a22)b11
        # p3 = a11(b12-b22)
        # p4 = a22(b21-b11)
        # p5 = (a11+a12)b22
        # p6 = (a21-a11)(b11+b12)
        # p7 = (a12-a22)(b21+b22)
        # C11 = p1+p4-p5+p7
        # C12 = p3+p5
        # C21 = p2+p4
        # C22 = p1-p2+p3+p6
        
        rank = 7
        U = [[0]*rank for _ in range(4)]  # indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        V = [[0]*rank for _ in range(4)]  # indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        W = [[0]*rank for _ in range(4)]  # indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        
        # p1 = (a11+a22)(b11+b22)
        U[0][0] = 1  # a11
        U[3][0] = 1  # a22
        V[0][0] = 1  # b11
        V[1][0] = 1  # b12 -- wait, b11+b22, not b11+b12
        # Correction: b11 is V[0,0], b22 is V[3,3] in our indexing? 
        # V[j*2+k, c] where j is row of B, k is col of B
        # So V[0][0] = B[0,0], V[1][0] = B[0,1], V[2][0] = B[1,0], V[3][0] = B[1,1]
        # b11 = V[0], b12 = V[1], b21 = V[2], b22 = V[3]
        
        V[0][0] = 1  # b11
        V[3][0] = 1  # b22
        # C11 += p1, C22 += p1
        W[0][0] = 1  # C[0,0]
        W[3][0] = 1  # C[1,1]
        
        # p2 = (a21+a22)b11
        U[2][1] = 1  # a21
        U[3][1] = 1  # a22
        V[0][1] = 1  # b11
        # C21 += p2
        W[2][1] = 1  # C[1,0]
        
        # p3 = a11(b12-b22)
        U[0][2] = 1  # a11
        V[1][2] = 1  # b12
        V[3][2] = -1  # -b22
        # C12 += p3, C22 += p3
        W[1][2] = 1  # C[0,1]
        W[3][2] = 1  # C[1,1]
        
        # p4 = a22(b21-b11)
        U[3][3] = 1  # a22
        V[2][3] = 1  # b21
        V[0][3] = -1  # -b11
        # C11 += p4, C21 += p4
        W[0][3] = 1  # C[0,0]
        W[2][3] = 1  # C[1,0]
        
        # p5 = (a11+a12)b22
        U[0][4] = 1  # a11
        U[1][4] = 1  # a12
        V[3][4] = 1  # b22
        # C11 -= p5, C12 += p5
        W[0][4] = -1  # C[0,0]
        W[1][4] = 1  # C[0,1]
        
        # p6 = (a21-a11)(b11+b12)
        U[2][5] = 1  # a21
        U[0][5] = -1  # -a11
        V[0][5] = 1  # b11
        V[1][5] = 1  # b12
        # C22 += p6
        W[3][5] = 1  # C[1,1]
        
        # p7 = (a12-a22)(b21+b22)
        U[1][6] = 1  # a12
        U[3][6] = -1  # -a22
        V[2][6] = 1  # b21
        V[3][6] = 1  # b22
        # C11 += p7
        W[0][6] = 1  # C[0,0]
        
        return U, V, W
    
    def schoolbook_block(n_b, m_b, p_b, offset_U, offset_V, offset_W, U, V, W, col_start):
        """Add schoolbook terms for an n_b x m_b x p_b block."""
        col = col_start
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[offset_U[i]][col] = 1
                    V[offset_V[j]][col] = 1
                    W[offset_W[k]][col] = 1
                    col += 1
        return col - col_start
    
    def strassen_block(n_b, m_b, p_b, offset_U, offset_V, offset_W, U, V, W, col_start):
        """Add Strassen terms for a 2x2x2 block. Only works for 2x2x2."""
        if n_b != 2 or m_b != 2 or p_b != 2:
            raise ValueError("Strassen only for 2x2x2")
        SU, SV, SW = strassen_2x2x2()
        col = col_start
        for c in range(7):
            for r in range(4):
                if SU[r][c] != 0:
                    U[offset_U[r]][col] += SU[r][c]
                if SV[r][c] != 0:
                    V[offset_V[r]][col] += SV[r][c]
                if SW[r][c] != 0:
                    W[offset_W[r]][col] += SW[r][c]
            col += 1
        return col - col_start
    
    # Define partitions
    # n=3: blocks of sizes [2, 1], starting indices [0, 2]
    n_blocks = [(2, 0), (1, 2)]  # (size, start)
    # m=5: blocks of sizes [2, 2, 1], starting indices [0, 2, 4]
    m_blocks = [(2, 0), (2, 2), (1, 4)]
    # p=7: blocks of sizes [2, 2, 2, 1], starting indices [0, 2, 4, 6]
    p_blocks = [(2, 0), (2, 2), (2, 4), (1, 6)]
    
    # Build offset arrays
    # U is indexed by (i, j) where i in [0,3), j in [0,5)
    # V is indexed by (j, k) where j in [0,5), k in [0,7)
    # W is indexed by (k, i) where k in [0,7), i in [0,3)
    
    # For each (n_block, m_block, p_block) triple, add terms
    rank = 0
    # First, determine total rank
    for ni, (nsz, noff) in enumerate(n_blocks):
        for mi, (msz, moff) in enumerate(m_blocks):
            for pi, (psz, poff) in enumerate(p_blocks):
                if nsz == 2 and msz == 2 and psz == 2:
                    rank += 7
                else:
                    rank += nsz * msz * psz
    
    # Allocate matrices
    U = [[0] * rank for _ in range(n * m)]  # 15 x rank
    V = [[0] * rank for _ in range(m * p)]  # 35 x rank
    W = [[0] * rank for _ in range(p * n)]  # 21 x rank
    
    # Fill in terms
    col = 0
    for ni, (nsz, noff) in enumerate(n_blocks):
        for mi, (msz, moff) in enumerate(m_blocks):
            for pi, (psz, poff) in enumerate(p_blocks):
                if nsz == 2 and msz == 2 and psz == 2:
                    # Strassen block
                    # offset_U: for local (li, lj) -> global U[(noff+li)*5 + (moff+lj)]
                    offset_U = [(noff + li) * m + (moff + lj) for li in range(2) for lj in range(2)]
                    # offset_V: for local (lj, lk) -> global V[(moff+lj)*7 + (poff+lk)]
                    offset_V = [(moff + lj) * p + (poff + lk) for lj in range(2) for lk in range(2)]
                    # offset_W: for local (lk, li) -> global W[(poff+lk)*3 + (noff+li)]
                    offset_W = [(poff + lk) * n + (noff + li) for lk in range(2) for li in range(2)]
                    col += strassen_block(nsz, msz, psz, offset_U, offset_V, offset_W, U, V, W, col)
                else:
                    # Schoolbook block
                    offset_U = [(noff + li) * m + (moff + lj) for li in range(nsz) for lj in range(msz)]
                    offset_V = [(moff + lj) * p + (poff + lk) for lj in range(msz) for lk in range(psz)]
                    offset_W = [(poff + lk) * n + (noff + li) for lk in range(psz) for li in range(nsz)]
                    col += schoolbook_block(nsz, msz, psz, offset_U, offset_V, offset_W, U, V, W, col)
    
    # Verify with random test
    try:
        A_test = [[random.uniform(-1, 1) for _ in range(m)] for _ in range(n)]
        B_test = [[random.uniform(-1, 1) for _ in range(p)] for _ in range(m)]
        
        # Compute A*B directly
        C_direct = [[sum(A_test[i][j] * B_test[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Compute via decomposition
        C_decomp = [[0.0 for _ in range(p)] for _ in range(n)]
        for c in range(rank):
            u_val = [sum(U[i*m+j][c] * A_test[i][j] for j in range(m)) for i in range(n)]  # Wait, this isn't right
            # U[i*m+j, c] is the coefficient for A[i,j] in the c-th product
            # So product_c = sum_{i,j} U[i*m+j, c] * A[i,j] * sum_{j,k} V[j*p+k, c] * B[j,k]
            # Then C[i,k] += W[k*n+i, c] * product_c
            
            u_prod = sum(U[i * m + j][c] * A_test[i][j] for i in range(n) for j in range(m))
            v_prod = sum(V[j * p + k][c] * B_test[j][k] for j in range(m) for k in range(p))
            prod = u_prod * v_prod
            for i in range(n):
                for k in range(p):
                    C_decomp[i][k] += W[k * n + i][c] * prod
        
        # Check max error
        max_err = max(abs(C_direct[i][k] - C_decomp[i][k]) for i in range(n) for k in range(p))
        if max_err > 1e-9:
            # Fall back to schoolbook
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
    except Exception:
        # Fall back to schoolbook
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
