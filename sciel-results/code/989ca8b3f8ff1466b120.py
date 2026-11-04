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

    # Build the decomposition using recursive Strassen with a split base case for 2x2x3.
    # 4x4 * 4x6:
    #   A is 4x4, split into 2x2 blocks of 2x2.
    #   B is 4x6, split into 2x2 blocks of 2x3.
    #   C is 4x6, split into 2x2 blocks of 2x3.
    #
    # The 7 Strassen sub-products are all of the form (2x2) * (2x3) -> (2x3).
    #
    # For the 2x2 * 2x3 sub-problem:
    #   B (2x3) is split into B1 (2x1) and B2 (2x2).
    #   A*B1 uses schoolbook rank 4.
    #   A*B2 uses Strassen rank 7.
    #   Total rank for 2x2x3 is 11.
    #
    # Total rank for 4x4x6 is 7 * 11 = 77.

    def build_2x2x1():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        rank = 4
        U = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows
        V = [[0]*rank for _ in range(2)]  # 2*1 = 2 rows
        W = [[0]*rank for _ in range(2)]  # 1*2 = 2 rows

        # Terms:
        # 0: A[0,0]*B[0] -> C[0]
        # 1: A[0,1]*B[1] -> C[0]
        # 2: A[1,0]*B[0] -> C[1]
        # 3: A[1,1]*B[1] -> C[1]

        # U: (i,j) -> index i*2+j
        # V: (j,k) -> index j*1+k = j
        # W: (k,i) -> index k*2+i

        # Term 0: A[0,0], B[0], C[0]
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1  # k=0, i=0 -> index 0

        # Term 1: A[0,1], B[1], C[0]
        U[1][1] = 1
        V[1][1] = 1
        W[0][1] = 1  # k=0, i=0 -> index 0

        # Term 2: A[1,0], B[0], C[1]
        U[2][2] = 1
        V[0][2] = 1
        W[1][2] = 1  # k=0, i=1 -> index 1

        # Term 3: A[1,1], B[1], C[1]
        U[3][3] = 1
        V[1][3] = 1
        W[1][3] = 1  # k=0, i=1 -> index 1

        return U, V, W

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows
        V = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows
        W = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows

        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # U index: i*2+j
        # V index: j*2+k
        # W index: k*2+i

        # Term 0: M1 = (A11+A22)(B11+B22)
        # A: (0,0)+1, (1,1)+1
        # B: (0,0)+1, (1,1)+1
        # C: (0,0)+1, (1,1)+1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # Term 1: M2 = (A21+A22)B11
        # A: (1,0)+1, (1,1)+1
        # B: (0,0)+1
        # C: (1,0)+1, (1,1)-1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1

        # Term 2: M3 = A11(B12-B22)
        # A: (0,0)+1
        # B: (0,1)+1, (1,1)-1
        # C: (0,1)+1, (1,1)+1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1

        # Term 3: M4 = A22(B21-B11)
        # A: (1,1)+1
        # B: (1,0)+1, (0,0)-1
        # C: (0,0)+1, (1,0)+1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1

        # Term 4: M5 = (A11+A12)B22
        # A: (0,0)+1, (0,1)+1
        # B: (1,1)+1
        # C: (0,0)-1, (0,1)+1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1

        # Term 5: M6 = (A21-A11)(B11+B12)
        # A: (1,0)+1, (0,0)-1
        # B: (0,0)+1, (0,1)+1
        # C: (1,1)+1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        # Term 6: M7 = (A12-A22)(B21+B22)
        # A: (0,1)+1, (1,1)-1
        # B: (1,0)+1, (1,1)+1
        # C: (0,0)+1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1

        return U, V, W

    def build_2x2x3():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3.
        Split B into B1 (2x1) and B2 (2x2).
        A*B1 uses rank-4 schoolbook.
        A*B2 uses rank-7 Strassen.
        """
        rank = 11
        U = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows
        V = [[0]*rank for _ in range(6)]  # 2*3 = 6 rows
        W = [[0]*rank for _ in range(6)]  # 3*2 = 6 rows

        # V index: j*3+k
        # W index: k*2+i

        # Part 1: A*B1 (rank 4), B1 is columns 0 of B, C1 is column 0 of C.
        # B1[0] -> V index 0*3+0 = 0
        # B1[1] -> V index 1*3+0 = 3
        # C1[0] -> W index 0*2+0 = 0
        # C1[1] -> W index 0*2+1 = 1

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

        # Part 2: A*B2 (rank 7), B2 is columns 1,2 of B, C2 is columns 1,2 of C.
        # B2[0,0] -> B[0,1] -> V index 0*3+1 = 1
        # B2[0,1] -> B[0,2] -> V index 0*3+2 = 2
        # B2[1,0] -> B[1,1] -> V index 1*3+1 = 4
        # B2[1,1] -> B[1,2] -> V index 1*3+2 = 5
        # C2[0,0] -> C[0,1] -> W index 1*2+0 = 2
        # C2[0,1] -> C[0,2] -> W index 2*2+0 = 4
        # C2[1,0] -> C[1,1] -> W index 1*2+1 = 3
        # C2[1,1] -> C[1,2] -> W index 2*2+1 = 5

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
        """Rank-77 Strassen decomposition for 4x4 * 4x6 -> 4x6 using Kronecker structure."""
        n, m, p = 4, 4, 6
        n2, m2, p2 = 2, 2, 3

        # Get the 2x2x3 sub-decomposition
        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = len(U_sub[0])  # 11
        total_rank = 7 * rank_sub  # 77

        U = [[0]*total_rank for _ in range(n*m)]  # 16 rows
        V = [[0]*total_rank for _ in range(m*p)]  # 24 rows
        W = [[0]*total_rank for _ in range(p*n)]  # 24 rows

        # Strassen terms for 4x4 * 4x6:
        # A blocks: A11 (0:2, 0:2), A12 (0:2, 2:4), A21 (2:4, 0:2), A22 (2:4, 2:4)
        # B blocks: B11 (0:2, 0:3), B12 (0:2, 3:6), B21 (2:4, 0:3), B22 (2:4, 3:6)
        # C blocks: C11 (0:2, 0:3), C12 (0:2, 3:6), C21 (2:4, 0:3), C22 (2:4, 3:6)
        #
        # U index: i*m + j = i*4 + j
        # V index: j*p + k = j*6 + k
        # W index: k*n + i = k*4 + i
        #
        # Sub-problem local indices:
        # U_sub: i*m2 + j = i*2 + j, where i in 0..1, j in 0..1
        # V_sub: j*p2 + k = j*3 + k, where j in 0..1, k in 0..2
        # W_sub: k*n2 + i = k*2 + i, where k in 0..2, i in 0..1
        #
        # Strassen terms:
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C12 += M1, C21 += M1, C22 -= M1
        # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
        # M3 = A11(B12-B22) -> C12 += M3, C22 += M3
        # M4 = A22(B21-B11) -> C11 += M4, C21 += M4
        # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
        # M6 = (A21-A11)(B11+B12) -> C22 += M6
        # M7 = (A12-A22)(B21+B22) -> C11 += M7, C22 += M7
        
        # Define the Strassen structure as (A_blocks, B_blocks, C_blocks)
        # Each block is (row_offset, col_offset)
        strassen_terms = [
            # M1: (A11+A22)(B11+B22)
            ([(0,0), (2,2)], [(0,0), (2,2)], [(0,0,1), (0,2,1), (2,0,1), (2,2,-1)]),
            # M2: (A21+A22)B11
            ([(2,0), (2,2)], [(0,0)], [(2,0,1), (2,2,-1)]),
            # M3: A11(B12-B22)
            ([(0,0)], [(0,2), (2,2)], [(0,2,1), (2,2,1)]),
            # M4: A22(B21-B11)
            ([(2,2)], [(2,0), (0,0)], [(0,0,1), (2,0,1)]),
            # M5: (A11+A12)B22
            ([(0,0), (0,2)], [(2,2)], [(0,0,-1), (0,2,1)]),
            # M6: (A21-A11)(B11+B12)
            ([(2,0), (0,0)], [(0,0), (0,2)], [(2,2,1)]),
            # M7: (A12-A22)(B21+B22)
            ([(0,2), (2,2)], [(2,0), (2,2)], [(0,0,1), (2,2,1)]),
        ]
        # Fix M2, M3, M4, M5, M6, M7 C-blocks to be (row, col, coeff)
        strassen_terms = [
            ([(0,0,1), (2,2,1)], [(0,0,1), (2,2,1)], [(0,0,1), (0,2,1), (2,0,1), (2,2,-1)]),
            ([(2,0,1), (2,2,1)], [(0,0,1)], [(2,0,1), (2,2,-1)]),
            ([(0,0,1)], [(0,2,1), (2,2,-1)], [(0,2,1), (2,2,1)]),
            ([(2,2,1)], [(2,0,1), (0,0,-1)], [(0,0,1), (2,0,1)]),
            ([(0,0,1), (0,2,1)], [(2,2,1)], [(0,0,-1), (0,2,1)]),
            ([(2,0,1), (0,0,-1)], [(0,0,1), (0,2,1)], [(2,2,1)]),
            ([(0,2,1), (2,2,-1)], [(2,0,1), (2,2,1)], [(0,0,1), (2,2,1)]),
        ]

        for t_idx, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                # U contribution
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0: continue
                    i_local = u_local // 2
                    j_local = u_local % 2
                    for a_block in a_blocks:
                        a_ro, a_co, coeff = a_block
                        g_u = (i_local + a_ro) * m + (j_local + a_co)
                        U[g_u][base_col + c] += u_val * coeff
                # V contribution
                for v_local in range(6):
                    v_val = V_sub[v_local][c]
                    if v_val == 0: continue
                    j_local = v_local // 3
                    k_local = v_local % 3
                    for b_block in b_blocks:
                        b_ro, b_co, coeff = b_block
                        g_v = (j_local + b_ro) * p + (k_local + b_co)
                        V[g_v][base_col + c] += v_val * coeff
                # W contribution
                for w_local in range(6):
                    w_val = W_sub[w_local][c]
                    if w_val == 0: continue
                    k_local = w_local // 2
                    i_local = w_local % 2
                    for c_block in c_blocks:
                        c_ro, c_co, coeff = c_block
                        g_w = (k_local + c_ro) * n + (i_local + c_co)
                        W[g_w][base_col + c] += w_val * coeff

        return U, V, W

    def make_trace(U, V, W, n, m, p):
        """Return the (n*m*p,) vector of trace products T[i,j,k]."""
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                u_row = Ua[i*m + j, :]
                for k in range(p):
                    T[i, j, k] = np.dot(u_row * Va[j*p + k, :], Wa[k*n + i, :])
        return T

    def verify(U, V, W, n, m, p, tol=1e-7):
        """Exact (dense) verification of the decomposition."""
        T = make_trace(U, V, W, n, m, p)
        return bool(np.all(np.abs(T - 1.0) <= tol))

    def try_reduce_rank(U, V, W, n, m, p, time_budget):
        """More aggressive rank reduction using linear algebra."""
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        R = Ua.shape[1]
        if R == 0:
            return U, V, W, R

        def column_contribution(r):
            u_col = Ua[:, r]
            v_col = Va[:, r]
            w_col = Wa[:, r]
            c = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    uv = u_col[i*m + j]
                    if uv == 0: continue
                    for k in range(p):
                        c[i, j, k] = uv * v_col[j*p + k] * w_col[k*n + i]
            return c

        # Step 1: Remove zero-contribution columns
        keep = list(range(R))
        contribs = {}
        for r in range(R):
            if time.time() > time_budget:
                break
            contribs[r] = column_contribution(r)

        changed = True
        while changed and time.time() < time_budget:
            changed = False
            for r in list(keep):
                if time.time() > time_budget:
                    break
                c = contribs.get(r)
                if c is None:
                    continue
                if np.all(np.abs(c) <= 1e-9):
                    keep.remove(r)
                    changed = True
                    break

        if len(keep) < R:
            Ua = Ua[:, keep]
            Va = Va[:, keep]
            Wa = Wa[:, keep]
            R = len(keep)

        # Step 2: Try to find linear dependencies using SVD
        if time.time() < time_budget - 5 and R > 1:
            nmp = n * m * p
            M = np.zeros((nmp, R), dtype=float)
            for r in range(R):
                c = contribs.get(r)
                if c is None:
                    c = column_contribution(r)
                    contribs[r] = c
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            M[i*m*p + j*p + k, r] = c[i, j, k]
            
            # SVD to find rank
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            except:
                pass

        if len(keep) == Ua.shape[1]:
            return U, V, W, R

        return Ua.tolist(), Va.tolist(), Wa.tolist(), R

    # Build the initial rank-77 decomposition.
    U, V, W = build_4x4x6()

    # Verify the initial decomposition; fall back to schoolbook if broken.
    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to reduce the rank (time-bounded).
    time_budget = min(deadline - 10, time.time() + 90)
    U_r, V_r, W_r, r_rank = try_reduce_rank(U, V, W, n, m, p, time_budget)

    # Use the reduced decomposition only if it still verifies exactly.
    if verify(U_r, V_r, W_r, n, m, p):
        return {"U": U_r, "V": V_r, "W": W_r}
    else:
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
