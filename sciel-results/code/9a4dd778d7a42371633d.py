"""Matrix multiplication 6x6x6 - structural improvement via rank reduction."""
import numpy as np
import time
import itertools

def construct():
    """
    Construct a low-rank tensor decomposition for 6x6 matrix multiplication.
    """
    n = m = p = 6
    deadline = time.time() + 110
    
    # Start with Strassen decomposition (rank 189)
    U, V, W = _strassen_6x6x6()
    rank = len(U[0])
    
    # Verify initial decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        rank = len(U[0])
    
    # Attempt rank reduction through linear dependency removal
    U, V, W, rank = _reduce_rank(U, V, W, n, m, p, deadline)
    
    return {'U': U, 'V': V, 'W': W}


def _strassen_6x6x6():
    """
    Strassen 2x2 block decomposition for 6x6x6 matrix multiplication.
    Split into 3x3 blocks. Each 3x3*3x3*3x3 has rank 27 (schoolbook).
    7 Strassen steps -> rank 7*27 = 189.
    """
    n = m = p = 6
    nh = mh = ph = 3
    block_rank = nh * mh * ph
    total_rank = 7 * block_rank
    
    # Strassen steps: (A_coeffs, B_coeffs)
    # A and B are 2x2 block matrices
    steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),  # M1 = (A00+A11)(B00+B11)
        ({2: 1, 3: 1}, {0: 1}),          # M2 = (A10+A11)B00
        ({0: 1}, {1: 1, 3: -1}),         # M3 = A00(B10-B00)
        ({3: 1}, {2: 1, 0: -1}),         # M4 = A11(B01-B11)
        ({0: 1, 1: 1}, {3: 1}),          # M5 = (A00+A01)B11
        ({2: 1, 0: -1}, {0: 1, 1: 1}),   # M6 = (A10-A00)(B00+B01)
        ({1: 1, 3: -1}, {2: 1, 3: 1}),   # M7 = (A01-A11)(B10+B11)
    ]
    
    # C block combinations
    c_combos = [
        {0: 1, 3: -1, 5: 1, 6: -1},  # C00
        {2: 1, 4: 1},                 # C10
        {1: 1, 3: 1},                 # C01
        {0: 1, 1: -1, 2: 1, 6: 1},    # C11
    ]
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for s in range(7):
        a_coeffs, b_coeffs = steps[s]
        for ai, a_sign in a_coeffs.items():
            for bi, b_sign in b_coeffs.items():
                # ai is block row index in A (0 or 1)
                # bi is block col index in B (0 or 1)
                a_row_start = (ai // 2) * nh
                a_col_start = (ai % 2) * mh
                b_row_start = (bi // 2) * mh
                b_col_start = (bi % 2) * ph
                
                for i in range(nh):
                    for j in range(mh):
                        for k in range(ph):
                            gi = a_row_start + i
                            gj = a_col_start + j
                            gk = b_col_start + k
                            
                            c_block_row = gi // nh
                            c_block_col = gk // ph
                            c_block_idx = c_block_row * 2 + c_block_col
                            
                            c_sign = c_combos[c_block_idx].get(s, 0)
                            if c_sign == 0:
                                continue
                            
                            col = s * block_rank + i * (mh * ph) + j * ph + k
                            
                            U[gi * m + gj][col] = a_sign
                            V[gj * p + gk][col] = b_sign
                            W[gk * n + gi][col] = c_sign
    
    return U, V, W


def _schoolbook(n, m, p):
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
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _reduce_rank(U, V, W, n, m, p, deadline, tol=1e-10):
    """
    Reduce rank by removing linearly dependent columns.
    """
    rank = len(U[0])
    
    # Convert to numpy for efficiency
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Find independent columns using SVD-based approach
    # We need to find which columns are linearly dependent
    # in the tensor sense
    
    # Strategy: iteratively check if a column can be removed
    # while maintaining the decomposition property
    
    cols_to_keep = list(range(rank))
    
    for col in sorted(cols_to_keep, key=lambda c: _column_norm(U_np[:, c], V_np[:, c], W_np[:, c])):
        if time.time() > deadline - 5:
            break
        
        # Try removing this column
        new_cols = [c for c in cols_to_keep if c != col]
        if len(new_cols) == 0:
            continue
        
        U_test = U_np[:, new_cols]
        V_test = V_np[:, new_cols]
        W_test = W_np[:, new_cols]
        
        if _verify_fast(U_test, V_test, W_test, n, m, p, tol):
            # Successfully removed column
            U_np = U_test
            V_np = V_test
            W_np = W_test
            cols_to_keep = new_cols
    
    # Convert back to lists
    U = U_np.tolist()
    V = V_np.tolist()
    W = W_np.tolist()
    
    return U, V, W, len(U[0])


def _column_norm(U_col, V_col, W_col):
    """Compute a norm-like value for a column."""
    return np.linalg.norm(U_col) * np.linalg.norm(V_col) * np.linalg.norm(W_col)


def _verify_fast(U, V, W, n, m, p, tol=1e-10):
    """Fast verification using numpy."""
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True
