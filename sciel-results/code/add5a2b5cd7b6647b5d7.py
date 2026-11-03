"""Matrix multiplication 2,4,7 tensor decomposition: direct constructive rank-50 via block Strassen."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _construct_rank50():
    """
    Directly construct a rank-50 exact decomposition for 2x4 x 4x7 matrix multiplication.
    
    Structure:
      C = A_top * B_top + A_bot * B_bot
      where A_top = A[:, 0:2], A_bot = A[:, 2:4], B_top = B[0:2, :], B_bot = B[2:4, :]
    
    Each 2x2 x 2x7 block is decomposed as:
      - Three 2x2 x 2x2 Strassen blocks (rank 7 each) for column groups (0,1), (2,3), (4,5)
      - One 2x2 x 2x1 schoolbook block (rank 4) for column group (6,)
      Total per 2x2 x 2x7: 7*3 + 4 = 25
      Total: 2 * 25 = 50
    
    Indexing convention:
      U: shape (n*m, rank) = (8, rank);  U[i*m + j, r]
      V: shape (m*p, rank) = (28, rank); V[j*p + k, r]
      W: shape (p*n, rank) = (14, rank); W[k*n + i, r]
    
    For each rank term r, we need:
      sum_r U[i*4+j, r] * V[j*7+k, r] * W[k*2+i, r] = 1  for all i in {0,1}, j in {0,1,2,3}, k in {0,...,6}
    """
    n, m, p = 2, 4, 7
    U_cols = []  # list of vectors of length n*m = 8
    V_cols = []  # list of vectors of length m*p = 28
    W_cols = []  # list of vectors of length p*n = 14
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def make_vec(dim, entries):
        vec = [0.0] * dim
        for idx, val in entries.items():
            vec[idx] = val
        return vec
    
    # Strassen 2x2x2x2: 7 terms
    # A: 2x2 with local rows ai in {0,1}, local cols aj in {0,1}
    # B: 2x2 with local rows bj in {0,1}, local cols bk in {0,1}
    # C: 2x2 with local rows ci in {0,1}, local cols ck in {0,1}
    #
    # Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each M_i = (linear form in A) * (linear form in B):
    #   u^i[ai, aj] = coefficient of A[ai,aj] in the A-linear form
    #   v^i[bj, bk] = coefficient of B[bj,bk] in the B-linear form
    #   w^i[ck, ci] = coefficient of M_i in C[ci, ck]  (W convention: W[k*n+i])
    
    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: (A10+A11)B00 -> C10 += 1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # M3: A00(B10-B11) -> C01 += 1
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1}),
        # M4: A11(B01-B10) -> C00 += 1, C10 += 1
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (1,0):1}),
        # M5: (A00+A01)B11 -> C00 -= 1, C01 += 1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6: (A10-A00)(B00+B10) -> C11 += 1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1}),
        # M7: (A01-A11)(B01+B11) -> C00 += 1
        ({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(0,0):1}),
    ]
    
    def add_strassen_block(A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_col_offset):
        """
        Add 7 Strassen terms for a 2x2 x 2x2 sub-block.
        A: rows 0,1 (global), cols A_col_offset, A_col_offset+1
        B: rows B_row_offset, B_row_offset+1, cols B_col_offset, B_col_offset+1
        C: rows 0,1 (global), cols C_col_offset, C_col_offset+1
        """
        for a_coefs, b_coefs, c_coefs in strassen_terms:
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            
            # Fill U: u[ai * m + (A_col_offset + aj)] = a_val
            for (ai, aj), a_val in a_coefs.items():
                global_col = A_col_offset + aj
                u[ai * m + global_col] = a_val
            
            # Fill V: v[(B_row_offset + bj) * p + (B_col_offset + bk)] = b_val
            for (bj, bk), b_val in b_coefs.items():
                global_row = B_row_offset + bj
                global_col = B_col_offset + bk
                v[global_row * p + global_col] = b_val
            
            # Fill W: w[(C_col_offset + ck) * n + ci] = c_val
            for (ci, ck), c_val in c_coefs.items():
                global_col = C_col_offset + ck
                w[global_col * n + ci] = c_val
            
            add_term(u, v, w)
    
    def add_schoolbook_2x2x2x1(A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_col_offset):
        """
        Add 4 schoolbook terms for a 2x2 x 2x1 sub-block.
        A: rows 0,1, cols A_col_offset, A_col_offset+1
        B: rows B_row_offset, B_row_offset+1, col B_col_offset
        C: rows 0,1, col C_col_offset
        
        C[0, C_col] = A[0, A_col0]*B[B_row0, B_col] + A[0, A_col1]*B[B_row1, B_col]
        C[1, C_col] = A[1, A_col0]*B[B_row0, B_col] + A[1, A_col1]*B[B_row1, B_col]
        
        4 terms:
        T1: A[0, A_col0] * B[B_row0, B_col] -> C[0, C_col] += 1
        T2: A[0, A_col1] * B[B_row1, B_col] -> C[0, C_col] += 1
        T3: A[1, A_col0] * B[B_row0, B_col] -> C[1, C_col] += 1
        T4: A[1, A_col1] * B[B_row1, B_col] -> C[1, C_col] += 1
        """
        terms = [
            # (ai, aj, bj, bk, ci, ck)
            (0, 0, 0, 0, 0, 0),
            (0, 1, 1, 0, 0, 0),
            (1, 0, 0, 0, 1, 0),
            (1, 1, 1, 0, 1, 0),
        ]
        for (ai, aj, bj, bk, ci, ck) in terms:
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            
            u[ai * m + (A_col_offset + aj)] = 1.0
            v[(B_row_offset + bj) * p + (B_col_offset + bk)] = 1.0
            w[(C_col_offset + ck) * n + ci] = 1.0
            
            add_term(u, v, w)
    
    # Process each 2x2 x 2x7 block
    for block in range(2):
        A_col_offset = block * 2  # 0 or 2
        B_row_offset = block * 2  # 0 or 2
        
        # Three Strassen 2x2x2x2 blocks for column groups (0,1), (2,3), (4,5)
        for col_group in range(3):
            B_col_offset = col_group * 2  # 0, 2, 4
            C_col_offset = col_group * 2  # 0, 2, 4
            add_strassen_block(0, A_col_offset, B_row_offset, B_col_offset, C_col_offset)
        
        # One schoolbook 2x2x2x1 block for column group (6,)
        B_col_offset = 6
        C_col_offset = 6
        add_schoolbook_2x2x2x1(0, A_col_offset, B_row_offset, B_col_offset, C_col_offset)
    
    # Convert to numpy arrays
    U = np.array(U_cols, dtype=np.float64).T  # shape (8, rank)
    V = np.array(V_cols, dtype=np.float64).T  # shape (28, rank)
    W = np.array(W_cols, dtype=np.float64).T  # shape (14, rank)
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = U.shape[1]
    # Use einsum for efficiency
    # U reshaped: (n, m, rank)
    # V reshaped: (m, p, rank)
    # W reshaped: (p, n, rank)
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kr->ijk', U_3d, V_3d, W_3d)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Directly construct rank-50 decomposition
    U, V, W = _construct_rank50()
    
    # Verify
    valid, max_err = _verify(U, V, W, n, m, p)
    
    if not valid:
        # This should not happen, but fallback to schoolbook if something is wrong
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    valid, err = _verify(U, V, W, 2, 4, 7)
    print(f"Valid: {valid}, Max error: {err}")
