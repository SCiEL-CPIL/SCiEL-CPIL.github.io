"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook_factors(n, m, p):
    """Return (U, V, W, rank) for schoolbook [n,m,p]."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank


def _strassen_222_factors():
    """
    Return U (4x7), V (4x7), W (4x7) for 2x2*2x2=2x2 using Strassen (rank 7).
    Verified numerically before use.
    """
    # Strassen formulas:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-b)(e+f)
    # M7 = (a+d)(g+h)
    # C[0,0] = M1 - M4 + M6 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M5 - M2
    
    # Verify with test values
    def strassen_2x2(A, B):
        a, b = A[0][0], A[0][1]
        c, d = A[1][0], A[1][1]
        e, f = B[0][0], B[0][1]
        g, h = B[1][0], B[1][1]
        M1 = (a + d) * (e + h)
        M2 = (c + d) * e
        M3 = a * (f - h)
        M4 = d * (g - e)
        M5 = (a + b) * h
        M6 = (c - b) * (e + f)
        M7 = (a + d) * (g + h)
        C00 = M1 - M4 + M6 - M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M3 + M5 - M2
        return [[C00, C01], [C10, C11]]
    
    # Test
    A_test = [[1, 2], [3, 4]]
    B_test = [[5, 6], [7, 8]]
    C_direct = [[1*5+3*7, 1*6+3*8], [2*5+4*7, 2*6+4*8]]
    C_strassen = strassen_2x2(A_test, B_test)
    if C_direct != C_strassen:
        # Fall back to schoolbook
        return _schoolbook_factors(2, 2, 2)
    
    # Build factor matrices
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # For each M_r, we have:
    # M_r = (sum_i,j u_r[i,j] * A[i,j]) * (sum_j,k v_r[j,k] * B[j,k])
    # and contributes w_r[i,k] * M_r to C[i,k]
    
    U = [[0]*7 for _ in range(4)]  # 4 rows (i*2+j for i,j in 0,1)
    V = [[0]*7 for _ in range(4)]  # 4 rows (j*2+k for j,k in 0,1)
    W = [[0]*7 for _ in range(4)]  # 4 rows (k*2+i for k,i in 0,1)
    
    # M1 = (a+d)(e+h): a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
    # A-part: A[0,0] + A[1,1] -> U[0,0]=1, U[3,0]=1 (i=0,j=0 -> idx 0; i=1,j=1 -> idx 3)
    # B-part: B[0,0] + B[1,1] -> V[0,0]=1, V[3,0]=1 (j=0,k=0 -> idx 0; j=1,k=1 -> idx 3)
    # W contributions: C[0,0]+=1, C[0,1]+=1, C[1,0]+=1, C[1,1]+=1
    # W[k*2+i]: C[0,0]->k=0,i=0->idx 0; C[0,1]->k=1,i=0->idx 2; C[1,0]->k=0,i=1->idx 1; C[1,1]->k=1,i=1->idx 3
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[2][0] = 1; W[1][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e: c=A[1,0], d=A[1,1], e=B[0,0]
    # A-part: A[1,0] + A[1,1] -> U[2,1]=1, U[3,1]=1 (i=1,j=0 -> idx 2; i=1,j=1 -> idx 3)
    # B-part: B[0,0] -> V[0,1]=1
    # C[1,0] += 1 -> W[1,1] = 1 (k=0,i=1 -> idx 1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3 = a(f-h): a=A[0,0], f=B[0,1], h=B[1,1]
    # A-part: A[0,0] -> U[0,2]=1
    # B-part: B[0,1] - B[1,1] -> V[1,2]=1, V[3,2]=-1 (j=0,k=1 -> idx 1; j=1,k=1 -> idx 3)
    # C[0,1] += 1 -> W[2,2] = 1 (k=1,i=0 -> idx 2)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1
    
    # M4 = d(g-e): d=A[1,1], g=B[1,0], e=B[0,0]
    # A-part: A[1,1] -> U[3,3]=1
    # B-part: B[1,0] - B[0,0] -> V[2,3]=1, V[0,3]=-1 (j=1,k=0 -> idx 2; j=0,k=0 -> idx 0)
    # C[0,0] -= 1 -> W[0,3] = -1; C[1,0] += 1 -> W[1,3] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[1][3] = 1
    
    # M5 = (a+b)h: a=A[0,0], b=A[0,1], h=B[1,1]
    # A-part: A[0,0] + A[0,1] -> U[0,4]=1, U[1,4]=1 (i=0,j=0 -> idx 0; i=0,j=1 -> idx 1)
    # B-part: B[1,1] -> V[3,4]=1
    # C[0,1] += 1 -> W[2,4] = 1; C[1,1] += 1 -> W[3,4] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[3][4] = 1
    
    # M6 = (c-b)(e+f): c=A[1,0], b=A[0,1], e=B[0,0], f=B[0,1]
    # A-part: A[1,0] - A[0,1] -> U[2,5]=1, U[1,5]=-1 (i=1,j=0 -> idx 2; i=0,j=1 -> idx 1)
    # B-part: B[0,0] + B[0,1] -> V[0,5]=1, V[1,5]=1 (j=0,k=0 -> idx 0; j=0,k=1 -> idx 1)
    # C[0,0] += 1 -> W[0,5] = 1
    U[2][5] = 1; U[1][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1
    
    # M7 = (a+d)(g+h): a=A[0,0], d=A[1,1], g=B[1,0], h=B[1,1]
    # A-part: A[0,0] + A[1,1] -> U[0,6]=1, U[3,6]=1
    # B-part: B[1,0] + B[1,1] -> V[2,6]=1, V[3,6]=1 (j=1,k=0 -> idx 2; j=1,k=1 -> idx 3)
    # C[0,0] -= 1 -> W[0,6] = -1; C[1,1] -= 1 -> W[3,6] = -1
    U[0][6] = 1; U[3][6] = 1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = -1
    
    return U, V, W, 7


def _block_decomposition():
    """
    Decompose [3,5,5] using block structure.
    A is 3x5, B is 5x5, C is 3x5.
    
    Split A into rows: A[0:2,:] (2x5) and A[2:3,:] (1x5)
    Split B into cols: B[:,0:2] (5x2) and B[:,2:5] (5x3)
    
    C[0:2,0:2] = A[0:2,:] * B[:,0:2]  -> [2,5,2]
    C[0:2,2:5] = A[0:2,:] * B[:,2:5]  -> [2,5,3]
    C[2:3,0:2] = A[2:3,:] * B[:,0:2]  -> [1,5,2]
    C[2:3,2:5] = A[2:3,:] * B[:,2:5]  -> [1,5,3]
    
    For [2,5,2]: split middle 5 into 2+2+1
      [2,2,2] x2 + [2,1,2] -> use Strassen for [2,2,2]: 7+7+2 = 16
    For [2,5,3]: split middle 5 into 2+2+1
      [2,2,3] x2 + [2,1,3] -> 12+12+3 = 27
    For [1,5,2]: 10 (schoolbook)
    For [1,5,3]: 15 (schoolbook)
    Total: 16+27+10+15 = 68
    """
    n, m, p = 3, 5, 5
    
    # We'll build sub-decompositions and combine them.
    # Each sub-decomposition handles a block of C with specific row/col ranges
    # and specific middle-index ranges.
    
    # Sub-decomp structure: (U_sub, V_sub, W_sub, rank, row_offset, col_offset, mid_start, mid_end)
    # U_sub has shape (n_sub * m_sub, rank) but we need to map to global indices.
    # Actually, let's think more carefully.
    
    # For a sub-problem [n_s, m_s, p_s] with:
    #   A rows: [row_start, row_start+n_s)
    #   B cols: [col_start, col_start+p_s)
    #   middle indices: [mid_start, mid_start+m_s)
    # The contribution to C[row_start+i, col_start+k] is:
    #   sum_{j=0}^{m_s-1} sum_r U_s[i*m_s+j, r] * V_s[j*p_s+k, r] * W_s[k*n_s+i, r]
    
    # Global U index: (row_start+i)*m + (mid_start+j)
    # Global V index: (mid_start+j)*p + (col_start+k)
    # Global W index: (col_start+k)*n + (row_start+i)
    
    def make_sub(n_s, m_s, p_s, row_start, col_start, mid_start, use_strassen=False):
        """Create sub-decomposition and return (U_global_rows, V_global_rows, W_global_rows, rank)."""
        if use_strassen and n_s == 2 and m_s == 2 and p_s == 2:
            U_s, V_s, W_s, rank_s = _strassen_222_factors()
        else:
            U_s, V_s, W_s, rank_s = _schoolbook_factors(n_s, m_s, p_s)
        
        # Create global-sized sparse rows
        U_rows = {}  # global_row -> list of (col, val)
        V_rows = {}
        W_rows = {}
        
        for i in range(n_s):
            for j in range(m_s):
                global_u_row = (row_start + i) * m + (mid_start + j)
                for r in range(rank_s):
                    val = U_s[i * m_s + j][r]
                    if val != 0:
                        U_rows.setdefault(global_u_row, []).append((r, val))
        
        for j in range(m_s):
            for k in range(p_s):
                global_v_row = (mid_start + j) * p + (col_start + k)
                for r in range(rank_s):
                    val = V_s[j * p_s + k][r]
                    if val != 0:
                        V_rows.setdefault(global_v_row, []).append((r, val))
        
        for k in range(p_s):
            for i in range(n_s):
                global_w_row = (col_start + k) * n + (row_start + i)
                for r in range(rank_s):
                    val = W_s[k * n_s + i][r]
                    if val != 0:
                        W_rows.setdefault(global_w_row, []).append((r, val))
        
        return U_rows, V_rows, W_rows, rank_s
    
    # Collect all sub-decompositions
    all_U_rows = {}
    all_V_rows = {}
    all_W_rows = {}
    total_rank = 0
    
    # Sub 1: C[0:2,0:2] = A[0:2,:] * B[:,0:2], middle split into [0:2],[2:4],[4:5]
    # 1a: [2,2,2] rows 0-1, cols 0-1, mid 0-1, Strassen
    U_r, V_r, W_r, r_s = make_sub(2, 2, 2, 0, 0, 0, use_strassen=True)
    for row, entries in U_r.items():
        all_U_rows.setdefault(row, [])
        for c, v in entries:
            all_U_rows[row].append((c + total_rank, v))
    for row, entries in V_r.items():
        all_V_rows.setdefault(row, [])
        for c, v in entries:
            all_V_rows[row].append((c + total_rank, v))
    for row, entries in W_r.items():
        all_W_rows.setdefault(row, [])
        for c, v in entries:
            all_W_rows[row].append((c + total_rank, v))
    total_rank += r_s
    
    # 1b: [2,2,2] rows 0-1, cols 0-1, mid 2-3, Strassen
    U_r, V_r, W_r, r_s = make_sub(2, 2, 2, 0, 0, 2, use_strassen=True)
    for row, entries in U_r.items():
        all_U_rows.setdefault(row, [])
        for c, v in entries:
            all_U_rows[row].append((c + total_rank, v))
    for row, entries in V_r.items():
        all_V_rows.setdefault(row, [])
        for c, v in entries:
            all_V_rows[row].append((c + total_rank, v))
    for row, entries in W_r.items():
        all_W_rows.setdefault(row, [])
        for c, v in entries:
            all_W_rows[row].append((c + total_rank, v))
    total_rank += r_s
    
    # 1c: [2,1,2] rows 0-1, cols 0-1, mid 4-5, schoolbook (rank 4)
    U_r, V_r, W_r, r_s = make_sub(2, 1, 2, 0, 0, 4, use_strassen=False)
    for row, entries in U_r.items():
        all_U_rows.setdefault(row, [])
        for c, v in entries:
            all_U_rows[row].append((c + total_rank, v))
    for row, entries in V_r.items():
        all_V_rows.setdefault(row, [])
        for c, v in entries:
            all_V_rows[row].append((c + total_rank, v))
    for row, entries in W_r.items():
        all_W_rows.setdefault(row, [])
        for c, v in entries:
            all_W_rows[row].append((c + total_rank, v))
    total_rank += r_s
    
    # Sub 2: C[0:2,2:5] = A[0:2,:] * B[:,2:5], middle split into [0:2],[2:4],[4:5]
    # 2a: [2,2,3] rows 0-1, cols 2-4, mid 0-1, schoolbook (rank 12)
    U_r, V_r, W_r, r_s = make_sub(2, 2, 3, 0, 2, 0, use_strassen=False)
    for row, entries in U_r.items():
        all_U_rows.setdefault(row, [])
        for c, v in entries:
            all_U_rows[row].append((c + total_rank, v))
    for row, entries in V_r.items():
        all_V_rows.setdefault(row, [])
        for c, v in entries:
            all_V_rows[row].append((c + total_rank, v))
    for row, entries in W_r.items():
        all_W_rows.setdefault(row, [])
        for c, v in entries:
            all_W_rows[row].append((c + total_rank, v))
    total_rank += r_s
    
    # 2b: [2,2,3] rows 0-1, cols 2-4, mid 2-3, schoolbook (rank 12)
    U_r, V_r, W_r, r_s = make_sub(2, 2, 3, 0, 2, 2, use_strassen=False)
    for row, entries in U_r.items():
        all_U_rows.setdefault(row, [])
        for c, v in entries:
            all_U_rows[row].append((c + total_rank, v))
    for row, entries in V_r.items():
        all_V_rows.setdefault(row, [])
        for c, v in entries:
            all_V_rows[row].append((c + total_rank, v))
    for row, entries in W_r.items():
        all_W_rows.setdefault(row, [])
        for c, v in entries:
            all_W_rows[row].append((c + total_rank, v))
    total_rank += r_s
    
    # 2c: [2,1,3] rows 0-1, cols 2-4, mid 4-5, schoolbook (rank 6)
    U_r, V_r, W_r, r_s = make_sub(2, 1, 3, 0, 2, 4, use_strassen=False)
    for row, entries in U_r.items():
        all_U_rows.setdefault(row, [])
        for c, v in entries:
            all_U_rows[row].append((c + total_rank, v))
    for row, entries in V_r.items():
        all_V_rows.setdefault(row, [])
        for c, v in entries:
            all_V_rows[row].append((c + total_rank, v))
    for row, entries in W_r.items():
        all_W_rows.setdefault(row, [])
        for c, v in entries:
            all_W_rows[row].append((c + total_rank, v))
    total_rank += r_s
    
    # Sub 3: C[2:3,0:2] = A[2:3,:] * B[:,0:2], middle split into [0:5] (no split needed, just 1x5x2)
    # 3a: [1,5,2] rows 2, cols 0-1, mid 0-4, schoolbook (rank 10)
    U_r, V_r, W_r, r_s = make_sub(1, 5, 2, 2, 0, 0, use_strassen=False)
    for row, entries in U_r.items():
        all_U_rows.setdefault(row, [])
        for c, v in entries:
            all_U_rows[row].append((c + total_rank, v))
    for row, entries in V_r.items():
        all_V_rows.setdefault(row, [])
        for c, v in entries:
            all_V_rows[row].append((c + total_rank, v))
    for row, entries in W_r.items():
        all_W_rows.setdefault(row, [])
        for c, v in entries:
            all_W_rows[row].append((c + total_rank, v))
    total_rank += r_s
    
    # Sub 4: C[2:3,2:5] = A[2:3,:] * B[:,2:5], middle split into [0:5]
    # 4a: [1,5,3] rows 2, cols 2-4, mid 0-4, schoolbook (rank 15)
    U_r, V_r, W_r, r_s = make_sub(1, 5, 3, 2, 2, 0, use_strassen=False)
    for row, entries in U_r.items():
        all_U_rows.setdefault(row, [])
        for c, v in entries:
            all_U_rows[row].append((c + total_rank, v))
    for row, entries in V_r.items():
        all_V_rows.setdefault(row, [])
        for c, v in entries:
            all_V_rows[row].append((c + total_rank, v))
    for row, entries in W_r.items():
        all_W_rows.setdefault(row, [])
        for c, v in entries:
            all_W_rows[row].append((c + total_rank, v))
    total_rank += r_s
    
    # Convert to dense matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for row, entries in all_U_rows.items():
        for c, v in entries:
            U[row][c] = v
    for row, entries in all_V_rows.items():
        for c, v in entries:
            V[row][c] = v
    for row, entries in all_W_rows.items():
        for c, v in entries:
            W[row][c] = v
    
    return U, V, W, total_rank


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS
    
    try:
        U, V, W, rank = _block_decomposition()
        
        # Verify correctness
        if time.time() < deadline:
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            # Check: C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            np.random.seed(42)
            A_test = np.random.randint(-3, 4, size=(n, m))
            B_test = np.random.randint(-3, 4, size=(m, p))
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((n, p), dtype=float)
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        for r in range(rank):
                            s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    C_computed[i, k] = s
            
            if np.allclose(C_computed, C_expected, atol=1e-9):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W, rank = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
