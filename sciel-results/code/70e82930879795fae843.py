"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _strassen_2x2x2_factors():
    """
    Return U, V, W factor matrices for 2x2x2 matrix multiplication using Strassen's rank-7 algorithm.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indices: U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[i,k]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00 + A11
    V[0, 0] = 1; V[3, 0] = 1  # B00 + B11
    W[0, 0] = 1; W[3, 0] = 1  # C00 + C11
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1  # A10 + A11
    V[0, 1] = 1               # B00
    W[2, 1] = 1; W[3, 1] = -1  # C10 - C11
    
    # M3: A00(B01-B11)
    U[0, 2] = 1               # A00
    V[1, 2] = 1; V[3, 2] = -1  # B01 - B11
    W[1, 2] = 1; W[3, 2] = 1   # C01 + C11
    
    # M4: A11(B10-B00)
    U[3, 3] = 1               # A11
    V[2, 3] = 1; V[0, 3] = -1  # B10 - B00
    W[0, 3] = 1; W[2, 3] = 1   # C00 + C10
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1  # A00 + A01
    V[3, 4] = 1               # B11
    W[0, 4] = -1; W[1, 4] = 1  # -C00 + C01
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1  # A10 - A00
    V[0, 5] = 1; V[1, 5] = 1   # B00 + B01
    W[3, 5] = 1                # C11
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1  # A01 - A11
    V[2, 6] = 1; V[3, 6] = 1   # B10 + B11
    W[0, 6] = 1                # C00
    
    return U, V, W


def _build_strassen_decomposition(n, m, p):
    """
    Build a Strassen-based decomposition for n x m times m x p matrix multiplication.
    Uses 2x2x2 Strassen blocks where possible, and schoolbook for remaining parts.
    """
    # We'll decompose into blocks of size 2x2x2 where possible
    # For 3x5x6: 
    # - Rows of A: 3 (can do one 2-block + one 1-row)
    # - Cols of A / rows of B: 5 (can do two 2-blocks + one 1-col)
    # - Cols of B: 6 (can do three 2-blocks)
    
    # Strategy: partition the computation into blocks
    # A is 3x5, B is 5x6, C is 3x6
    # 
    # Let's partition:
    # A rows: [0,1] and [2]
    # A cols/B rows: [0,1], [2,3], [4]
    # B cols: [0,1], [2,3], [4,5]
    
    # This gives us blocks:
    # Block (0,0): A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]  (2x2x2, Strassen rank 7)
    # Block (0,1): A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]  (2x2x2, Strassen rank 7)
    # Block (0,2): A[0:2,4:5] * B[4:5,0:2] -> C[0:2,0:2]  (2x1x2, schoolbook rank 4)
    # Block (1,0): A[2:3,0:2] * B[0:2,0:2] -> C[2:3,0:2]  (1x2x2, schoolbook rank 4)
    # Block (1,1): A[2:3,2:4] * B[2:4,0:2] -> C[2:3,0:2]  (1x2x2, schoolbook rank 4)
    # Block (1,2): A[2:3,4:5] * B[4:5,0:2] -> C[2:3,0:2]  (1x1x2, schoolbook rank 2)
    # Block (2,0): A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]  (2x2x2, Strassen rank 7)
    # Block (2,1): A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]  (2x2x2, Strassen rank 7)
    # Block (2,2): A[0:2,4:5] * B[4:5,2:4] -> C[0:2,2:4]  (2x1x2, schoolbook rank 4)
    # Block (3,0): A[2:3,0:2] * B[0:2,2:4] -> C[2:3,2:4]  (1x2x2, schoolbook rank 4)
    # Block (3,1): A[2:3,2:4] * B[2:4,2:4] -> C[2:3,2:4]  (1x2x2, schoolbook rank 4)
    # Block (3,2): A[2:3,4:5] * B[4:5,2:4] -> C[2:3,2:4]  (1x1x2, schoolbook rank 2)
    # Block (4,0): A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]  (2x2x2, Strassen rank 7)
    # Block (4,1): A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]  (2x2x2, Strassen rank 7)
    # Block (4,2): A[0:2,4:5] * B[4:5,4:6] -> C[0:2,4:6]  (2x1x2, schoolbook rank 4)
    # Block (5,0): A[2:3,0:2] * B[0:2,4:6] -> C[2:3,4:6]  (1x2x2, schoolbook rank 4)
    # Block (5,1): A[2:3,2:4] * B[2:4,4:6] -> C[2:3,4:6]  (1x2x2, schoolbook rank 4)
    # Block (5,2): A[2:3,4:5] * B[4:5,4:6] -> C[2:3,4:6]  (1x1x2, schoolbook rank 2)
    
    # Total rank: 6*7 + 6*4 + 6*2 = 42 + 24 + 12 = 78
    # Wait, let me recount:
    # Strassen blocks (2x2x2): 6 blocks * 7 = 42
    # Schoolbook 2x1x2 or 1x2x2: 6 blocks * 4 = 24
    # Schoolbook 1x1x2: 6 blocks * 2 = 12
    # Total: 78
    
    # Let's build this decomposition
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a block
    def add_block(a_row_start, a_row_end, a_col_start, a_col_end, 
                  b_row_start, b_row_end, b_col_start, b_col_end,
                  c_row_start, c_row_end, c_col_start, c_col_end,
                  use_strassen=False):
        """Add a block to the decomposition."""
        a_rows = list(range(a_row_start, a_row_end))
        a_cols = list(range(a_col_start, a_col_end))
        b_rows = list(range(b_row_start, b_row_end))
        b_cols = list(range(b_col_start, b_col_end))
        c_rows = list(range(c_row_start, c_row_end))
        c_cols = list(range(c_col_start, c_col_end))
        
        ar = len(a_rows)
        ac = len(a_cols)
        br = len(b_rows)
        bc = len(b_cols)
        cr = len(c_rows)
        cc = len(c_cols)
        
        if use_strassen and ar == 2 and ac == 2 and br == 2 and bc == 2 and cr == 2 and cc == 2:
            # Use Strassen
            sU, sV, sW = _strassen_2x2x2_factors()
            for r in range(7):
                u_col = np.zeros(n * m, dtype=float)
                v_col = np.zeros(m * p, dtype=float)
                w_col = np.zeros(p * n, dtype=float)
                
                # Fill U
                for idx, ai in enumerate(a_rows):
                    for jdx, aj in enumerate(a_cols):
                        u_val = sU[idx * 2 + jdx, r]
                        if u_val != 0:
                            u_col[ai * m + aj] = u_val
                
                # Fill V
                for idx, bi in enumerate(b_rows):
                    for kdx, bk in enumerate(b_cols):
                        v_val = sV[idx * 2 + kdx, r]
                        if v_val != 0:
                            v_col[bi * p + bk] = v_val
                
                # Fill W
                for idx, ci in enumerate(c_rows):
                    for kdx, ck in enumerate(c_cols):
                        w_val = sW[idx * 2 + kdx, r]
                        if w_val != 0:
                            w_col[ck * n + ci] = w_val
                
                U_list.append(u_col)
                V_list.append(v_col)
                W_list.append(w_col)
        else:
            # Use schoolbook
            for ai in a_rows:
                for aj in a_cols:
                    for bk in b_cols:
                        # Note: b_rows should match a_cols for matrix multiplication
                        u_col = np.zeros(n * m, dtype=float)
                        v_col = np.zeros(m * p, dtype=float)
                        w_col = np.zeros(p * n, dtype=float)
                        
                        u_col[ai * m + aj] = 1
                        v_col[aj * p + bk] = 1
                        w_col[bk * n + ai] = 1
                        
                        U_list.append(u_col)
                        V_list.append(v_col)
                        W_list.append(w_col)
    
    # Define the blocks
    # A is 3x5, B is 5x6, C is 3x6
    # A rows: [0,1], [2]
    # A cols: [0,1], [2,3], [4]
    # B cols: [0,1], [2,3], [4,5]
    
    # Block (a_row_block, a_col_block, b_col_block)
    a_row_blocks = [(0, 2), (2, 3)]
    a_col_blocks = [(0, 2), (2, 4), (4, 5)]
    b_col_blocks = [(0, 2), (2, 4), (4, 6)]
    
    for ar_start, ar_end in a_row_blocks:
        for ac_start, ac_end in a_col_blocks:
            for bc_start, bc_end in b_col_blocks:
                ar = ar_end - ar_start
                ac = ac_end - ac_start
                bc = bc_end - bc_start
                cr = ar  # C rows match A rows
                cc = bc  # C cols match B cols
                
                use_strassen = (ar == 2 and ac == 2 and bc == 2)
                add_block(ar_start, ar_end, ac_start, ac_end,
                          ac_start, ac_end, bc_start, bc_end,
                          ar_start, ar_end, bc_start, bc_end,
                          use_strassen=use_strassen)
    
    # Convert to matrices
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    rank = U.shape[1]
    return U, V, W, rank


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    # Check: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k
    
    # Build the tensor from the decomposition
    rank = U.shape[1]
    
    # Check a few random entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    
    return True, None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # First, create the schoolbook decomposition as a fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try Strassen-based decomposition
    try:
        U_str, V_str, W_str, rank_str = _build_strassen_decomposition(n, m, p)
        
        # Verify
        valid, err = _verify_decomposition(U_str, V_str, W_str, n, m, p)
        
        if valid and rank_str < best_rank:
            best_rank = rank_str
            best_U = U_str
            best_V = V_str
            best_W = W_str
    except Exception as e:
        pass
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
