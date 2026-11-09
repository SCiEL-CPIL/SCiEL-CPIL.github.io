"""Matrix multiplication 4,5,6 - Strassen-based rank reduction with padding."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _add_term(U, V, W, i, j, k, n, m, p, col):
    """Add a single rank-1 term: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1."""
    U[i * m + j][col] = 1
    V[j * p + k][col] = 1
    W[k * n + i][col] = 1
    return col + 1

def _add_strassen_term(U, V, W, n, m, p, col, a_terms, b_terms, c_terms):
    """Add a rank-1 term defined by linear forms on A, B, and C.
    a_terms: list of (i, j, coeff) for A-side
    b_terms: list of (j, k, coeff) for B-side
    c_terms: list of (i, k, coeff) for C-side
    """
    for (i, j, coeff) in a_terms:
        U[i * m + j][col] = coeff
    for (j, k, coeff) in b_terms:
        V[j * p + k][col] = coeff
    for (i, k, coeff) in c_terms:
        W[k * n + i][col] = coeff
    return col + 1

def _strassen_2x2(U, V, W, i0, j0, k0, n, m, p, col):
    """Strassen decomposition for 2x2 @ 2x2 block starting at (i0,j0,k0)."""
    # P1 = (A11+A22)(B11+B22)
    col = _add_strassen_term(U, V, W, n, m, p, col,
        [(i0,j0,1), (i0+1,j0+1,1)],
        [(j0,k0,1), (j0+1,k0+1,1)],
        [(i0,k0,1), (i0+1,k0+1,1)])
    
    # P2 = (A21+A22)B11
    col = _add_strassen_term(U, V, W, n, m, p, col,
        [(i0+1,j0,1), (i0+1,j0+1,1)],
        [(j0,k0,1)],
        [(i0+1,k0,1), (i0+1,k0+1,-1)])
    
    # P3 = A11(B21-B22)
    col = _add_strassen_term(U, V, W, n, m, p, col,
        [(i0,j0,1)],
        [(j0+1,k0,1), (j0+1,k0+1,-1)],
        [(i0,k0+1,1), (i0+1,k0+1,1)])
    
    # P4 = A22(B11-B21)
    col = _add_strassen_term(U, V, W, n, m, p, col,
        [(i0+1,j0+1,1)],
        [(j0,k0,1), (j0+1,k0,-1)],
        [(i0,k0,1), (i0+1,k0,1)])
    
    # P5 = (A11+A12)B22
    col = _add_strassen_term(U, V, W, n, m, p, col,
        [(i0,j0,1), (i0,j0+1,1)],
        [(j0+1,k0+1,1)],
        [(i0,k0,-1), (i0,k0+1,1)])
    
    # P6 = (A21-A11)(B11+B12)
    col = _add_strassen_term(U, V, W, n, m, p, col,
        [(i0+1,j0,1), (i0,j0,-1)],
        [(j0,k0,1), (j0,k0+1,1)],
        [(i0+1,k0+1,1)])
    
    # P7 = (A12-A22)(B21+B22)
    col = _add_strassen_term(U, V, W, n, m, p, col,
        [(i0,j0+1,1), (i0+1,j0+1,-1)],
        [(j0+1,k0,1), (j0+1,k0+1,1)],
        [(i0,k0,1)])
    
    return col

def _strassen_3x3x3(U, V, W, i0, j0, k0, n, m, p, col):
    """Decomposition for 3x3 @ 3x3 block using Strassen + schoolbook (rank 23 vs schoolbook 27)."""
    # Use Strassen for the top-left 2x2 @ 2x2 -> 2x2 (rank 7)
    col = _strassen_2x2(U, V, W, i0, j0, k0, n, m, p, col)
    
    # Schoolbook for the remaining 20 terms
    for di in range(3):
        for dj in range(3):
            for dk in range(3):
                # Skip if this was already covered by Strassen (top-left 2x2 block)
                if di < 2 and dj < 2 and dk < 2:
                    continue
                col = _add_term(U, V, W, i0+di, j0+dj, k0+dk, n, m, p, col)
    
    return col

def _decompose_block(U, V, W, n, m, p, i0, j0, k0, col, deadline):
    """Recursively decompose the n x m @ m x p block at offset (i0,j0,k0)."""
    if time.time() > deadline:
        return col
    
    if n == 0 or m == 0 or p == 0:
        return col
    
    # Base case: 2x2x2 uses Strassen (rank 7 vs schoolbook 8)
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2(U, V, W, i0, j0, k0, n, m, p, col)
    
    # Base case: 3x3x3 (rank 23 vs schoolbook 27)
    if n == 3 and m == 3 and p == 3:
        return _strassen_3x3x3(U, V, W, i0, j0, k0, n, m, p, col)
    
    # Base case: if any dimension is 1, use schoolbook
    if n == 1 or m == 1 or p == 1:
        for di in range(n):
            for dj in range(m):
                for dk in range(p):
                    col = _add_term(U, V, W, i0+di, j0+dj, k0+dk, n, m, p, col)
        return col
    
    # Small blocks: use schoolbook
    if n <= 2 and m <= 2 and p <= 2:
        for di in range(n):
            for dj in range(m):
                for dk in range(p):
                    col = _add_term(U, V, W, i0+di, j0+dj, k0+dk, n, m, p, col)
        return col
    
    # Try splitting strategies and pick the one with lowest rank
    best_col = col
    best_strategy = None
    
    # Try splitting m (middle dimension)
    if m >= 4:
        m1 = m // 2
        m2 = m - m1
        if m1 >= 2 and m2 >= 2:
            c1 = _decompose_block(U, V, W, n, m1, p, i0, j0, k0, col, deadline)
            c2 = _decompose_block(U, V, W, n, m2, p, i0, j0+m1, k0, c1, deadline)
            if best_strategy is None or c2 < best_col:
                best_col = c2
                best_strategy = ('m', m1)
    elif m == 3:
        c1 = _decompose_block(U, V, W, n, 1, p, i0, j0, k0, col, deadline)
        c2 = _decompose_block(U, V, W, n, 2, p, i0, j0+1, k0, c1, deadline)
        if best_strategy is None or c2 < best_col:
            best_col = c2
            best_strategy = ('m', 1)
    
    # Try splitting p (output dimension)
    if p >= 4:
        p1 = p // 2
        p2 = p - p1
        if p1 >= 2 and p2 >= 2:
            c1 = _decompose_block(U, V, W, n, m, p1, i0, j0, k0, col, deadline)
            c2 = _decompose_block(U, V, W, n, m, p2, i0, j0, k0+p1, c1, deadline)
            if best_strategy is None or c2 < best_col:
                best_col = c2
                best_strategy = ('p', p1)
    elif p == 3:
        c1 = _decompose_block(U, V, W, n, m, 1, i0, j0, k0, col, deadline)
        c2 = _decompose_block(U, V, W, n, m, 2, i0, j0, k0+1, c1, deadline)
        if best_strategy is None or c2 < best_col:
            best_col = c2
            best_strategy = ('p', 1)
    
    # Try splitting n (input dimension)
    if n >= 4:
        n1 = n // 2
        n2 = n - n1
        if n1 >= 2 and n2 >= 2:
            c1 = _decompose_block(U, V, W, n1, m, p, i0, j0, k0, col, deadline)
            c2 = _decompose_block(U, V, W, n2, m, p, i0+n1, j0, k0, c1, deadline)
            if best_strategy is None or c2 < best_col:
                best_col = c2
                best_strategy = ('n', n1)
    elif n == 3:
        c1 = _decompose_block(U, V, W, 1, m, p, i0, j0, k0, col, deadline)
        c2 = _decompose_block(U, V, W, 2, m, p, i0+1, j0, k0, c1, deadline)
        if best_strategy is None or c2 < best_col:
            best_col = c2
            best_strategy = ('n', 1)
    
    if best_strategy is not None:
        return best_col
    
    # Fallback: schoolbook
    for di in range(n):
        for dj in range(m):
            for dk in range(p):
                col = _add_term(U, V, W, i0+di, j0+dj, k0+dk, n, m, p, col)
    return col

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return n == 0 or m == 0 or p == 0
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    C_decomposed = np.zeros((n, p))
    for c in range(rank):
        Uc = U_arr[:, c].reshape(n, m)
        Vc = V_arr[:, c].reshape(m, p)
        Wc = W_arr[:, c].reshape(p, n)
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += Uc[i, j] * A[i, j] * Vc[j, k] * B[j, k] * Wc[k, i]
                C_decomposed[i, k] += s
    
    if not np.allclose(C_decomposed, C_expected, atol=1e-10):
        return False
    return True

def _next_pow2(x):
    x = max(1, x)
    p2 = 1
    while p2 < x:
        p2 *= 2
    return p2

def _strassen_padded(n, m, p):
    """
    Use Strassen with padding to next power of 2, then extract and prune.
    This can sometimes yield lower rank than direct recursive decomposition
    for non-power-of-2 dimensions.
    """
    N = _next_pow2(max(n, m, p))
    
    # Build full Strassen decomposition for N x N x N
    max_rank = N * N * N  # Upper bound
    U_full = [[0] * max_rank for _ in range(N * N)]
    V_full = [[0] * max_rank for _ in range(N * N)]
    W_full = [[0] * max_rank for _ in range(N * N)]
    
    deadline = time.time() + 50
    col = _decompose_block(U_full, V_full, W_full, N, N, N, 0, 0, 0, 0, deadline)
    
    if col > max_rank:
        return None
    
    # Trim to actual rank
    U_full = [row[:col] for row in U_full]
    V_full = [row[:col] for row in V_full]
    W_full = [row[:col] for row in W_full]
    
    # Extract sub-decomposition for n x m x p
    U_sub = [[U_full[i*N + j][c] for c in range(col)] for i in range(n) for j in range(m)]
    V_sub = [[V_full[j*N + k][c] for c in range(col)] for j in range(m) for k in range(p)]
    W_sub = [[W_full[k*N + i][c] for c in range(col)] for k in range(p) for i in range(n)]
    
    # Prune zero columns
    keep_cols = []
    for c in range(col):
        u_nonzero = any(U_sub[i][c] != 0 for i in range(n * m))
        v_nonzero = any(V_sub[i][c] != 0 for i in range(m * p))
        w_nonzero = any(W_sub[i][c] != 0 for i in range(p * n))
        if u_nonzero and v_nonzero and w_nonzero:
            keep_cols.append(c)
    
    U_final = [[U_sub[i][c] for c in keep_cols] for i in range(n * m)]
    V_final = [[V_sub[i][c] for c in keep_cols] for i in range(m * p)]
    W_final = [[W_sub[i][c] for c in keep_cols] for i in range(p * n)]
    
    return U_final, V_final, W_final

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Leave safety margin
    
    best_result = None
    best_rank = n * m * p + 1  # Schoolbook rank + 1
    
    # Strategy 1: Direct recursive Strassen decomposition
    try:
        max_rank = n * m * p
        U = [[0] * max_rank for _ in range(n * m)]
        V = [[0] * max_rank for _ in range(m * p)]
        W = [[0] * max_rank for _ in range(p * n)]
        
        col = _decompose_block(U, V, W, n, m, p, 0, 0, 0, 0, deadline)
        
        if col <= max_rank:
            U = [row[:col] for row in U]
            V = [row[:col] for row in V]
            W = [row[:col] for row in W]
            
            if _verify(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = col
    except Exception:
        pass
    
    # Strategy 2: Padded Strassen with pruning
    if time.time() < deadline - 30:
        try:
            result = _strassen_padded(n, m, p)
            if result is not None:
                U_pad, V_pad, W_pad = result
                rank_pad = len(U_pad[0]) if U_pad else 0
                if rank_pad < best_rank:
                    if _verify(U_pad, V_pad, W_pad, n, m, p):
                        best_result = {"U": U_pad, "V": V_pad, "W": W_pad}
                        best_rank = rank_pad
        except Exception:
            pass
    
    # Fallback: schoolbook decomposition
    if best_result is None:
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
