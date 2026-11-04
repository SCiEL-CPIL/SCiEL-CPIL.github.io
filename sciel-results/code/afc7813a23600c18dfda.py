import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    rank = 38
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    def add_term(u_entries, v_entries, w_entries):
        """Add one rank-1 term to the decomposition."""
        nonlocal rank
        col = rank
        rank += 1
        U.append([0.0] * rank)
        V.append([0.0] * rank)
        W.append([0.0] * rank)
        for i in range(n * m):
            U[i][col] = 0.0
        for i in range(m * p):
            V[i][col] = 0.0
        for i in range(p * n):
            W[i][col] = 0.0
        for idx, val in u_entries:
            U[idx][col] = val
        for idx, val in v_entries:
            V[idx][col] = val
        for idx, val in w_entries:
            W[idx][col] = val

    def add_strassen_2x2x2(a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22):
        """
        Add 7 Strassen terms for 2x2x2 matrix multiplication.
        U indices: a11, a12, a21, a22 (for A block)
        V indices: b11, b12, b21, b22 (for B block)
        W indices: c11, c12, c21, c22 (for C block)
        """
        # M1 = (A11+A22)(B11+B22) -> C11+1, C22+1
        add_term([(a11, 1), (a22, 1)], [(b11, 1), (b22, 1)], [(c11, 1), (c22, 1)])
        # M2 = (A21+A22)B11 -> C21+1, C22-1
        add_term([(a21, 1), (a22, 1)], [(b11, 1)], [(c21, 1), (c22, -1)])
        # M3 = A11(B12-B22) -> C12+1, C22+1
        add_term([(a11, 1)], [(b12, 1), (b22, -1)], [(c12, 1), (c22, 1)])
        # M4 = A22(B21-B11) -> C11+1, C21+1
        add_term([(a22, 1)], [(b21, 1), (b11, -1)], [(c11, 1), (c21, 1)])
        # M5 = (A11+A12)B22 -> C12+1, C11-1
        add_term([(a11, 1), (a12, 1)], [(b22, 1)], [(c12, 1), (c11, -1)])
        # M6 = (A21-A11)(B11+B12) -> C21+1, C22+1
        add_term([(a21, 1), (a11, -1)], [(b11, 1), (b12, 1)], [(c21, 1), (c22, 1)])
        # M7 = (A12-A22)(B21+B22) -> C11+1, C12+1
        add_term([(a12, 1), (a22, -1)], [(b21, 1), (b22, 1)], [(c11, 1), (c12, 1)])

    def add_schoolbook_1x2x2(a_row, b0, b1, c0, c1):
        """
        Add 2 schoolbook terms for 1x2x2 multiplication.
        A is 1x2 (one row), B is 2x2, C is 1x2.
        U index: a_row (single row of A)
        V indices: b0, b1 (columns of B)
        W indices: c0, c1 (columns of C)
        """
        # C[0] = A[0]*B[0]
        add_term([(a_row, 1)], [(b0, 1)], [(c0, 1)])
        # C[1] = A[0]*B[1]
        add_term([(a_row, 1)], [(b1, 1)], [(c1, 1)])

    # Part 1: First 2 columns of B (B[:, 0:2])
    # A[:, 0:2] is 3x2, B[0:2, :] is 2x4
    # Split into 2x2x2 (rows 0-1 of A) + 1x2x2 (row 2 of A)
    
    # U indices for A[:, 0:2]: A[i, 0] = i*4+0, A[i, 1] = i*4+1
    # V indices for B[0:2, :]: B[j, k] = j*4+k
    # W indices for C[:, 0:2]: C[k, i] = k*3+i (trace convention W[k*n+i])
    
    # 2x2x2 block: A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
    # A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # C[0,0]=0, C[0,1]=1, C[1,0]=3, C[1,1]=4
    add_strassen_2x2x2(0, 1, 4, 5, 0, 1, 4, 5, 0, 1, 3, 4)
    
    # 1x2x2 block: A[2, 0:2], B[0:2, 0:2], C[0:2, 2]
    # A[2,0]=8, A[2,1]=9
    # B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # C[0,2]=2, C[1,2]=5
    add_schoolbook_1x2x2(8, 0, 4, 2, 5)
    add_schoolbook_1x2x2(9, 0, 4, 2, 5)
    
    # Wait, that's wrong. Let me reconsider.
    # For 1x2x2: A is 1x2, B is 2x2, C is 1x2
    # C[0] = A[0]*B[0,0] + A[1]*B[1,0]
    # C[1] = A[0]*B[0,1] + A[1]*B[1,1]
    # So we need 4 terms, not 2.
    
    # Let me redo this. Actually for 1x2x2, schoolbook is 1*2*2 = 4 terms.
    # But I can use Strassen-like tricks? No, 1x2x2 doesn't benefit from Strassen.
    # Actually, 1x2x2 has rank at most 4 (schoolbook). Let me just use schoolbook.
    
    # Let me restart with a cleaner approach.
    
    # Reset
    rank = 38
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Clear and rebuild
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col_ptr = [0]
    
    def add_term2(u_entries, v_entries, w_entries):
        col = col_ptr[0]
        col_ptr[0] += 1
        for idx, val in u_entries:
            U[idx][col] = val
        for idx, val in v_entries:
            V[idx][col] = val
        for idx, val in w_entries:
            W[idx][col] = val
    
    def add_strassen2(a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22):
        add_term2([(a11, 1), (a22, 1)], [(b11, 1), (b22, 1)], [(c11, 1), (c22, 1)])
        add_term2([(a21, 1), (a22, 1)], [(b11, 1)], [(c21, 1), (c22, -1)])
        add_term2([(a11, 1)], [(b12, 1), (b22, -1)], [(c12, 1), (c22, 1)])
        add_term2([(a22, 1)], [(b21, 1), (b11, -1)], [(c11, 1), (c21, 1)])
        add_term2([(a11, 1), (a12, 1)], [(b22, 1)], [(c12, 1), (c11, -1)])
        add_term2([(a21, 1), (a11, -1)], [(b11, 1), (b12, 1)], [(c21, 1), (c22, 1)])
        add_term2([(a12, 1), (a22, -1)], [(b21, 1), (b22, 1)], [(c11, 1), (c12, 1)])
    
    # Part 1: C[:, 0:2] = A[:, 0:2] * B[0:2, :]
    # This is 3x2x4
    
    # Sub-part 1a: C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] (2x2x2)
    # A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # C[0,0]=0, C[0,1]=1, C[1,0]=3, C[1,1]=4
    add_strassen2(0, 1, 4, 5, 0, 1, 4, 5, 0, 1, 3, 4)
    
    # Sub-part 1b: C[0:2, 2:4] = A[0:2, 0:2] * B[0:2, 2:4] (2x2x2)
    # B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # C[0,2]=2, C[0,3]=3, C[1,2]=5, C[1,3]=6
    add_strassen2(0, 1, 4, 5, 2, 3, 6, 7, 2, 3, 5, 6)
    
    # Sub-part 1c: C[2, 0:2] = A[2, 0:2] * B[0:2, 0:2] (1x2x2, schoolbook rank 4)
    # A[2,0]=8, A[2,1]=9
    # B[0,0]=0, B[1,0]=4, B[0,1]=1, B[1,1]=5
    # C[2,0]=6, C[2,1]=7
    add_term2([(8, 1)], [(0, 1)], [(6, 1)])
    add_term2([(9, 1)], [(4, 1)], [(6, 1)])
    add_term2([(8, 1)], [(1, 1)], [(7, 1)])
    add_term2([(9, 1)], [(5, 1)], [(7, 1)])
    
    # Sub-part 1d: C[2, 2:4] = A[2, 0:2] * B[0:2, 2:4] (1x2x2, schoolbook rank 4)
    # B[0,2]=2, B[1,2]=6, B[0,3]=3, B[1,3]=7
    # C[2,2]=8, C[2,3]=9
    add_term2([(8, 1)], [(2, 1)], [(8, 1)])
    add_term2([(9, 1)], [(6, 1)], [(8, 1)])
    add_term2([(8, 1)], [(3, 1)], [(9, 1)])
    add_term2([(9, 1)], [(7, 1)], [(9, 1)])
    
    # Part 2: C[:, 2:4] = A[:, 2:4] * B[2:4, :]
    # This is 3x2x4
    
    # Sub-part 2a: C[0:2, 0:2] = A[0:2, 2:4] * B[2:4, 0:2] (2x2x2)
    # A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    # C[0,0]=0, C[0,1]=1, C[1,0]=3, C[1,1]=4
    add_strassen2(2, 3, 6, 7, 8, 9, 12, 13, 0, 1, 3, 4)
    
    # Sub-part 2b: C[0:2, 2:4] = A[0:2, 2:4] * B[2:4, 2:4] (2x2x2)
    # B[2,2]=10, B[2,3]=11, B[3,2]=14, B[3,3]=15
    # C[0,2]=2, C[0,3]=3, C[1,2]=5, C[1,3]=6
    add_strassen2(2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 5, 6)
    
    # Sub-part 2c: C[2, 0:2] = A[2, 2:4] * B[2:4, 0:2] (1x2x2, schoolbook rank 4)
    # A[2,2]=10, A[2,3]=11
    # B[2,0]=8, B[3,0]=12, B[2,1]=9, B[3,1]=13
    # C[2,0]=6, C[2,1]=7
    add_term2([(10, 1)], [(8, 1)], [(6, 1)])
    add_term2([(11, 1)], [(12, 1)], [(6, 1)])
    add_term2([(10, 1)], [(9, 1)], [(7, 1)])
    add_term2([(11, 1)], [(13, 1)], [(7, 1)])
    
    # Sub-part 2d: C[2, 2:4] = A[2, 2:4] * B[2:4, 2:4] (1x2x2, schoolbook rank 4)
    # B[2,2]=10, B[3,2]=14, B[2,3]=11, B[3,3]=15
    # C[2,2]=8, C[2,3]=9
    add_term2([(10, 1)], [(10, 1)], [(8, 1)])
    add_term2([(11, 1)], [(14, 1)], [(8, 1)])
    add_term2([(10, 1)], [(11, 1)], [(9, 1)])
    add_term2([(11, 1)], [(15, 1)], [(9, 1)])
    
    # Part 3: C[3, :] = A[3, :] * B (1x4x4, schoolbook rank 16)
    # A[3,0]=12, A[3,1]=13, A[3,2]=14, A[3,3]=15
    # For each k in 0..3: C[3,k] = sum_j A[3,j]*B[j,k]
    # W index for C[3,k] = k*3+3
    for j in range(4):
        for k in range(4):
            add_term2([(12 + j, 1)], [(j * 4 + k, 1)], [(k * 3 + 3, 1)])
    
    # Verify
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails
    rank_sb = n * m * p
    U_sb = [[0.0] * rank_sb for _ in range(n * m)]
    V_sb = [[0.0] * rank_sb for _ in range(m * p)]
    W_sb = [[0.0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1.0
                V_sb[j * p + k][col] = 1.0
                W_sb[k * n + i][col] = 1.0
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
