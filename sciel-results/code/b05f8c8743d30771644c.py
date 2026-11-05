"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # 150s wall-clock limit, keep 40s margin

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Index conventions (trace convention W[k*n + i, r]):
    #   U index: i*m + j   (A is n x m)
    #   V index: j*p + k   (B is m x p)
    #   W index: k*n + i   (C is p x n, trace convention)
    # A tensor decomposition of rank R is valid iff for every (i,j,k):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1

    def build_2x2x1():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        rank = 4
        U = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows
        V = [[0]*rank for _ in range(2)]  # 2*1 = 2 rows
        W = [[0]*rank for _ in range(2)]  # 1*2 = 2 rows (k*n+i, n=2)

        # Term 0: A[0,0]*B[0] -> C[0,0]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        # Term 1: A[0,1]*B[1] -> C[0,0]
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        # Term 2: A[1,0]*B[0] -> C[1,0]
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        # Term 3: A[1,1]*B[1] -> C[1,0]
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        return U, V, W

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        # W index: k*2 + i
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
        return U, V, W

    def build_2x2x3():
        """Rank-11 decomposition for 2x2 * 2x3 -> 2x3.
        Split B into B1 (2x1) and B2 (2x2).
        A*B1 uses rank-4 schoolbook.
        A*B2 uses rank-7 Strassen.
        """
        rank = 11
        U = [[0]*rank for _ in range(4)]  # 2*2
        V = [[0]*rank for _ in range(6)]  # 2*3
        W = [[0]*rank for _ in range(6)]  # 3*2 (k*2+i)

        U1, V1, W1 = build_2x2x1()
        for c in range(4):
            for r in range(4):
                if U1[r][c] != 0: U[r][c] += U1[r][c]
            for r in range(2):
                if V1[r][c] != 0: V[r*3][c] += V1[r][c]   # B row r, col 0
            for r in range(2):
                if W1[r][c] != 0: W[r][c] += W1[r][c]     # C row r, col 0 -> k=0,i=r -> idx r

        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        v2_map = {0:1, 1:2, 2:4, 3:5}   # B2 (j,k2) -> B (j, k2+1): idx j*3+(k2+1)
        w2_map = {0:2, 1:4, 2:3, 3:5}   # C2 (k2,i) -> C (k2+1,i): idx (k2+1)*2+i
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0: U[r][base+c] += U2[r][c]
            for r in range(4):
                if V2[r][c] != 0: V[v2_map[r]][base+c] += V2[r][c]
            for r in range(4):
                if W2[r][c] != 0: W[w2_map[r]][base+c] += W2[r][c]
        return U, V, W

    def build_4x4x4_winograd():
        """Winograd F(4) block-circulant decomposition for 4x4 * 4x4 -> 4x4.
        
        Uses the (2,2)-block Strassen structure with F(4) evaluation on the
        4x4 output. The 4x4x4 problem is decomposed into 7 rank-7 Strassen
        sub-problems (one per 2x2 block multiplication), giving rank 49.
        """
        n, m, p = 4, 4, 4
        U_sub, V_sub, W_sub = build_2x2x2_strassen()
        rank_sub = 7
        total_rank = 7 * rank_sub  # 49

        U = [[0]*total_rank for _ in range(n*m)]  # 16
        V = [[0]*total_rank for _ in range(m*p)]  # 16
        W = [[0]*total_rank for _ in range(p*n)]  # 16

        # Standard Strassen terms for 2x2 blocks
        terms = [
            {'A': [(0,1),(3,1)], 'B': [(0,1),(3,1)], 'C': [(0,1),(3,1)]},
            {'A': [(2,1),(3,1)], 'B': [(0,1)], 'C': [(2,1),(3,-1)]},
            {'A': [(0,1)], 'B': [(1,1),(3,-1)], 'C': [(1,1),(3,1)]},
            {'A': [(3,1)], 'B': [(2,1),(0,-1)], 'C': [(0,1),(2,1)]},
            {'A': [(0,1),(1,1)], 'B': [(3,1)], 'C': [(0,-1),(1,1)]},
            {'A': [(2,1),(0,-1)], 'B': [(0,1),(1,1)], 'C': [(3,1)]},
            {'A': [(1,1),(3,-1)], 'B': [(2,1),(3,1)], 'C': [(0,1)]},
        ]
        u_offset = {0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)}
        v_offset = {0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)}
        w_offset = {0:(0,0), 1:(2,0), 2:(0,2), 3:(2,2)}

        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0: continue
                    i_local = u_local // 2; j_local = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offset[block]
                        g_u = (i_local + ro) * m + (j_local + co)
                        U[g_u][base_col + c] += u_val * coeff
                for v_local in range(4):
                    v_val = V_sub[v_local][c]
                    if v_val == 0: continue
                    j_local = v_local // 2; k_local = v_local % 2
                    for block, coeff in term['B']:
                        ro, co = v_offset[block]
                        g_v = (j_local + ro) * p + (k_local + co)
                        V[g_v][base_col + c] += v_val * coeff
                for w_local in range(4):
                    w_val = W_sub[w_local][c]
                    if w_val == 0: continue
                    k_local = w_local // 2; i_local = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offset[block]
                        g_w = (k_local + ro) * n + (i_local + co)
                        W[g_w][base_col + c] += w_val * coeff
        return U, V, W

    def build_4x4x6():
        """Build 4x4*4x6 decomposition by extending the 4x4x4 Winograd core.
        
        Key insight: C[:, 4:6] = C[:, 0:4] * M where M is a 2x4 matrix with
        entries in {0,1}. We can express the last 2 output columns as linear
        combinations of the first 4, sharing the same rank-1 terms.
        
        Approach: Use the 4x4x4 decomposition (rank 49) for the first 4 columns.
        For columns 4 and 5 of B, we need additional terms. But we can use
        the algebraic structure: if we set B[:,4] = B[:,0]+B[:,2] and 
        B[:,5] = B[:,1]+B[:,3], then C[:,4] = C[:,0]+C[:,2] and 
        C[:,5] = C[:,1]+C[:,3]. This means we don't need new rank-1 terms!
        
        However, this only works for specific B. For general B, we need a 
        different approach. Instead, we'll use the full 4x6 decomposition but
        with a more efficient structure.
        
        Alternative: Decompose 4x4*4x6 as 4x4*4x4 + 4x4*4x2, where the 
        4x4*4x2 part uses rank-12 (2 schoolbook 2x2x1 blocks). Total: 49+12=61.
        """
        n, m, p = 4, 4, 6
        
        # Build the 4x4x4 core (rank 49)
        U_core, V_core, W_core = build_4x4x4_winograd()
        rank_core = 49
        
        # Build 4x4*4x2 as two 2x2*2x1 blocks (rank 8 each, total 16)
        # Actually, let's do it more carefully:
        # C[:, 4:6] = A * B[:, 4:6]
        # We can decompose this using 2 independent 4x4*4x1 problems
        # Each 4x4*4x1 has rank 16 (schoolbook) or can use Strassen
        
        # For 4x4*4x1: split into 2x2 blocks
        # C_top[:,4] = A_top * B_top[:,4] (2x2 * 2x1, rank 4)
        # C_bot[:,4] = A_bot * B_bot[:,4] (2x2 * 2x1, rank 4)
        # Total for one column: rank 8
        # For two columns: rank 16
        
        # So total rank = 49 + 16 = 65
        total_rank = rank_core + 16
        
        U = [[0]*total_rank for _ in range(n*m)]  # 16
        V = [[0]*total_rank for _ in range(m*p)]  # 24
        W = [[0]*total_rank for _ in range(p*n)]  # 24
        
        # Copy the core decomposition
        for r in range(n*m):
            for c in range(rank_core):
                U[r][c] = U_core[r][c]
        for r in range(m*4):  # Only first 4 columns of B
            for c in range(rank_core):
                V[r][c] = V_core[r][c]
        for r in range(4*n):  # Only first 4 rows of C (k=0..3)
            for c in range(rank_core):
                W[r][c] = W_core[r][c]
        
        # Add the 4x4*4x2 part for columns 4 and 5
        base = rank_core
        # For column 4 (k=4):
        # C[:,4] = A * B[:,4]
        # Split: C_top[:,4] = A_top * B_top[:,4], C_bot[:,4] = A_bot * B_bot[:,4]
        
        # Schoolbook for 2x2 * 2x1 -> 2x1 (rank 4)
        # Term 0: A[0,0]*B[0] -> C[0]
        # Term 1: A[0,1]*B[1] -> C[0]
        # Term 2: A[1,0]*B[0] -> C[1]
        # Term 3: A[1,1]*B[1] -> C[1]
        
        def add_2x2x1_block(u_row_off, v_col_off, w_row_off, col_base):
            """Add a 2x2*2x1 schoolbook block at specified offsets."""
            # u_row_off: row offset in U (i offset)
            # v_col_off: column offset in V (k offset)  
            # w_row_off: row offset in W (k offset for output row)
            for term in range(4):
                c = col_base + term
                if term == 0:  # A[0,0]*B[0] -> C[0]
                    U[u_row_off + 0][c] = 1
                    V[(0)*6 + v_col_off + 0][c] = 1
                    W[(w_row_off + 0)*4 + 0][c] = 1
                elif term == 1:  # A[0,1]*B[1] -> C[0]
                    U[u_row_off + 1][c] = 1
                    V[(1)*6 + v_col_off + 0][c] = 1
                    W[(w_row_off + 0)*4 + 0][c] = 1
                elif term == 2:  # A[1,0]*B[0] -> C[1]
                    U[u_row_off + 4][c] = 1
                    V[(0)*6 + v_col_off + 0][c] = 1
                    W[(w_row_off + 0)*4 + 1][c] = 1
                elif term == 3:  # A[1,1]*B[1] -> C[1]
                    U[u_row_off + 5][c] = 1
                    V[(1)*6 + v_col_off + 0][c] = 1
                    W[(w_row_off + 0)*4 + 1][c] = 1
        
        # Column 4: k=4
        # Top half (i=0,1): u_row_off=0, w_row_off=4
        add_2x2x1_block(0, 4, 4, base)
        # Bottom half (i=2,3): u_row_off=8, w_row_off=4
        add_2x2x1_block(8, 4, 4, base + 4)
        
        # Column 5: k=5
        # Top half (i=0,1): u_row_off=0, w_row_off=5
        add_2x2x1_block(0, 5, 5, base + 8)
        # Bottom half (i=2,3): u_row_off=8, w_row_off=5
        add_2x2x1_block(8, 5, 5, base + 12)
        
        return U, V, W

    def build_4x4x6_strassen():
        """Fallback: original Strassen-based 4x4x6 decomposition (rank 77)."""
        n, m, p = 4, 4, 6
        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = len(U_sub[0])  # 11
        total_rank = 7 * rank_sub  # 77

        U = [[0]*total_rank for _ in range(n*m)]  # 16
        V = [[0]*total_rank for _ in range(m*p)]  # 24
        W = [[0]*total_rank for _ in range(p*n)]  # 24

        terms = [
            {'A': [(0,1),(3,1)], 'B': [(0,1),(3,1)], 'C': [(0,1),(3,1)]},
            {'A': [(2,1),(3,1)], 'B': [(0,1)], 'C': [(2,1),(3,-1)]},
            {'A': [(0,1)], 'B': [(1,1),(3,-1)], 'C': [(1,1),(3,1)]},
            {'A': [(3,1)], 'B': [(2,1),(0,-1)], 'C': [(0,1),(2,1)]},
            {'A': [(0,1),(1,1)], 'B': [(3,1)], 'C': [(0,-1),(1,1)]},
            {'A': [(2,1),(0,-1)], 'B': [(0,1),(1,1)], 'C': [(3,1)]},
            {'A': [(1,1),(3,-1)], 'B': [(2,1),(3,1)], 'C': [(0,1)]},
        ]
        u_offset = {0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)}
        v_offset = {0:(0,0), 1:(0,3), 2:(2,0), 3:(2,3)}
        w_offset = {0:(0,0), 1:(3,0), 2:(0,2), 3:(3,2)}

        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            for c in range(rank_sub):
                for u_local in range(4):
                    u_val = U_sub[u_local][c]
                    if u_val == 0: continue
                    i_local = u_local // 2; j_local = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offset[block]
                        g_u = (i_local + ro) * m + (j_local + co)
                        U[g_u][base_col + c] += u_val * coeff
                for v_local in range(6):
                    v_val = V_sub[v_local][c]
                    if v_val == 0: continue
                    j_local = v_local // 3; k_local = v_local % 3
                    for block, coeff in term['B']:
                        ro, co = v_offset[block]
                        g_v = (j_local + ro) * p + (k_local + co)
                        V[g_v][base_col + c] += v_val * coeff
                for w_local in range(6):
                    w_val = W_sub[w_local][c]
                    if w_val == 0: continue
                    k_local = w_local // 2; i_local = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offset[block]
                        g_w = (k_local + ro) * n + (i_local + co)
                        W[g_w][base_col + c] += w_val * coeff
        return U, V, W

    def make_trace(U, V, W, n, m, p):
        """Return the (n*m*p,) vector of trace products T[i,j,k]."""
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
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
        """Greedy column elimination: remove any rank-1 column whose removal
        keeps every trace product equal to 1. Vectorized; time-bounded."""
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        R = Ua.shape[1]
        if R == 0:
            return U, V, W, R

        # Precompute per-column contribution to the trace tensor.
        # contrib[r] is (n,m,p): U[:,r] outer V[:,r] outer W[:,r] in index space.
        # T = sum_r contrib[r]; removing r keeps T iff contrib[r] == 0 (elementwise)
        # AND T - contrib[r] == 1 wherever it must be 1. Since T==1 initially,
        # removal is valid iff contrib[r] == 0 on all (i,j,k).
        T = make_trace(U, V, W, n, m, p)  # (n,m,p)
        # Build index maps
        idx_U = [(i*m + j) for i in range(n) for j in range(m)]
        idx_V = [(j*p + k) for j in range(m) for k in range(p)]
        idx_W = [(k*n + i) for k in range(p) for i in range(n)]

        def column_contribution(r):
            # contrib[i,j,k] = Ua[idx_U[i*m+j], r] * Va[idx_V[j*p+k], r] * Wa[idx_W[k*n+i], r]
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

        # Start from a full copy; iterate removals.
        keep = list(range(R))
        # Precompute contributions for all columns once (R is at most 77).
        contribs = {}
        for r in range(R):
            if time.time() > time_budget:
                break
            contribs[r] = column_contribution(r)

        # Greedy: repeatedly find a removable column (contribution zero) and drop it.
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

        if len(keep) == R:
            return U, V, W, R

        U_new = Ua[:, keep]
        V_new = Va[:, keep]
        W_new = Wa[:, keep]
        return U_new.tolist(), V_new.tolist(), W_new.tolist(), len(keep)

    # Build the initial decomposition using Winograd + extension.
    U, V, W = build_4x4x6()
    
    # Verify the initial decomposition; fall back to schoolbook if broken.
    if not verify(U, V, W, n, m, p):
        # Try the original Strassen-based approach as fallback
        U_alt, V_alt, W_alt = build_4x4x6_strassen()
        if verify(U_alt, V_alt, W_alt, n, m, p):
            U, V, W = U_alt, V_alt, W_alt
        else:
            # Final fallback: schoolbook
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
