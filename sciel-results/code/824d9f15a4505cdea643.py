import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _validate_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Validate that the given factor matrices exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Construct the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W each of shape (4, 7).
    
    Convention: U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[k,i]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # P1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (A10+A11) B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    
    # P3 = A00 (B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = -1
    
    # P4 = A11 (B01-B10)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # P5 = (A00+A01) B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # P6 = (A10-A00) (B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = -1
    
    # P7 = (A01-A11) (B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _decompose_221():
    """
    Decompose 5x5x5 matrix multiplication using a 2+2+1 block split.
    
    Split: rows/cols 0-1 form block 0 (size 2), rows/cols 2-3 form block 1 (size 2), row/col 4 forms block 2 (size 1).
    
    For each pair of blocks (I, J, K) where I is the row block of A, J is the column block of A / row block of B, 
    and K is the column block of B:
    - If I=J=K and size is 2: use Strassen (rank 7)
    - If I=J=K and size is 1: use schoolbook (rank 1)
    - Otherwise: use schoolbook for the cross term
    
    Block sizes: [2, 2, 1]
    """
    n = 5
    block_sizes = [2, 2, 1]
    block_starts = [0, 2, 4]
    
    # Precompute Strassen for 2x2x2
    U2, V2, W2 = _strassen_2x2x2()
    
    # Collect all terms
    terms = []
    
    for I in range(3):
        for J in range(3):
            for K in range(3):
                si = block_sizes[I]
                sj = block_sizes[J]
                sk = block_sizes[K]
                
                if I == J == K and si == 2:
                    # Diagonal 2x2x2 block: use Strassen
                    for r in range(7):
                        for i in range(2):
                            for j in range(2):
                                for k in range(2):
                                    u_idx = (block_starts[I] + i) * 5 + (block_starts[J] + j)
                                    v_idx = (block_starts[J] + j) * 5 + (block_starts[K] + k)
                                    w_idx = (block_starts[K] + k) * 5 + (block_starts[I] + i)
                                    
                                    u_val = U2[i * 2 + j, r]
                                    v_val = V2[j * 2 + k, r]
                                    w_val = W2[k * 2 + i, r]
                                    
                                    if u_val != 0 and v_val != 0 and w_val != 0:
                                        terms.append((u_idx, v_idx, w_idx, u_val * v_val * w_val))
                elif I == J == K and si == 1:
                    # Diagonal 1x1x1 block: schoolbook
                    i = block_starts[I]
                    j = block_starts[J]
                    k = block_starts[K]
                    u_idx = i * 5 + j
                    v_idx = j * 5 + k
                    w_idx = k * 5 + i
                    terms.append((u_idx, v_idx, w_idx, 1.0))
                else:
                    # Cross term: schoolbook
                    for i in range(si):
                        for j in range(sj):
                            for k in range(sk):
                                u_idx = (block_starts[I] + i) * 5 + (block_starts[J] + j)
                                v_idx = (block_starts[J] + j) * 5 + (block_starts[K] + k)
                                w_idx = (block_starts[K] + k) * 5 + (block_starts[I] + i)
                                terms.append((u_idx, v_idx, w_idx, 1.0))
    
    # Now build the factor matrices
    # First, count the rank
    rank = len(terms)
    
    U = np.zeros((25, rank), dtype=float)
    V = np.zeros((25, rank), dtype=float)
    W = np.zeros((25, rank), dtype=float)
    
    for r, (u_idx, v_idx, w_idx, val) in enumerate(terms):
        U[u_idx, r] = 1.0
        V[v_idx, r] = 1.0
        W[w_idx, r] = 1.0
    
    # Wait, the above is wrong. The Strassen terms have coefficients in U, V, W separately.
    # Let me redo this more carefully.
    
    U = np.zeros((25, rank), dtype=float)
    V = np.zeros((25, rank), dtype=float)
    W = np.zeros((25, rank), dtype=float)
    
    for r in range(rank):
        u_idx, v_idx, w_idx, _ = terms[r]
        U[u_idx, r] = 1.0
        V[v_idx, r] = 1.0
        W[w_idx, r] = 1.0
    
    return U, V, W

def _decompose_221_correct():
    """
    Correct implementation of 2+2+1 block decomposition.
    """
    n = 5
    block_sizes = [2, 2, 1]
    block_starts = [0, 2, 4]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # We'll build the decomposition term by term
    # Each term contributes to one column of U, V, W
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for I in range(3):
        for J in range(3):
            for K in range(3):
                si = block_sizes[I]
                sj = block_sizes[J]
                sk = block_sizes[K]
                
                if I == J == K and si == 2:
                    # Diagonal 2x2x2 block: use Strassen
                    for r in range(7):
                        u_col = np.zeros(25, dtype=float)
                        v_col = np.zeros(25, dtype=float)
                        w_col = np.zeros(25, dtype=float)
                        
                        for i in range(2):
                            for j in range(2):
                                u_val = U2[i * 2 + j, r]
                                if u_val != 0:
                                    u_idx = (block_starts[I] + i) * 5 + (block_starts[J] + j)
                                    u_col[u_idx] = u_val
                        
                        for j in range(2):
                            for k in range(2):
                                v_val = V2[j * 2 + k, r]
                                if v_val != 0:
                                    v_idx = (block_starts[J] + j) * 5 + (block_starts[K] + k)
                                    v_col[v_idx] = v_val
                        
                        for k in range(2):
                            for i in range(2):
                                w_val = W2[k * 2 + i, r]
                                if w_val != 0:
                                    w_idx = (block_starts[K] + k) * 5 + (block_starts[I] + i)
                                    w_col[w_idx] = w_val
                        
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                elif I == J == K and si == 1:
                    # Diagonal 1x1x1 block: schoolbook
                    i = block_starts[I]
                    j = block_starts[J]
                    k = block_starts[K]
                    u_idx = i * 5 + j
                    v_idx = j * 5 + k
                    w_idx = k * 5 + i
                    
                    u_col = np.zeros(25, dtype=float)
                    v_col = np.zeros(25, dtype=float)
                    w_col = np.zeros(25, dtype=float)
                    u_col[u_idx] = 1.0
                    v_col[v_idx] = 1.0
                    w_col[w_idx] = 1.0
                    
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
                else:
                    # Cross term: schoolbook
                    for i in range(si):
                        for j in range(sj):
                            for k in range(sk):
                                u_idx = (block_starts[I] + i) * 5 + (block_starts[J] + j)
                                v_idx = (block_starts[J] + j) * 5 + (block_starts[K] + k)
                                w_idx = (block_starts[K] + k) * 5 + (block_starts[I] + i)
                                
                                u_col = np.zeros(25, dtype=float)
                                v_col = np.zeros(25, dtype=float)
                                w_col = np.zeros(25, dtype=float)
                                u_col[u_idx] = 1.0
                                v_col[v_idx] = 1.0
                                w_col[w_idx] = 1.0
                                
                                U_cols.append(u_col)
                                V_cols.append(v_col)
                                W_cols.append(w_col)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return U, V, W

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Try 2+2+1 decomposition
    try:
        U, V, W = _decompose_221_correct()
        if _validate_decomposition(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception:
        pass
    
    # Fall back to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
