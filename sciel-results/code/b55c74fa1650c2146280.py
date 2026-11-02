"""Matrix multiplication 3,3,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return schoolbook decomposition for 3x3 x 3x6. Rank = 54."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=3, m=3, p=6, tol=1e-9):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    # Check off-diagonal entries
    for i in range(n):
        for j in range(m):
            for i2 in range(n):
                for j2 in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j2 * p + k
                        c = k * n + i2
                        expected = 1.0 if (i == i2 and j == j2) else 0.0
                        val = float(np.sum(U_arr[a] * V_arr[b] * W_arr[c]))
                        err = abs(val - expected)
                        if err > max_err:
                            max_err = err
    return max_err < tol

def _strassen_2x2():
    """
    Return rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    This is Strassen's algorithm with exact integer coefficients.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    n = 2
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # U indices: A00=0, A01=1, A10=2, A11=3
    # V indices: B00=0, B01=1, B10=2, B11=3
    # W indices: C00=0, C01=1, C10=2, C11=3
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1; W[2][0] = -1; W[1][0] = -1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[3][3] = 1; W[2][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[0][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _embed_strassen_3x3(U2, V2, W2):
    """
    Embed 2x2 Strassen decomposition into 3x3 by adding identity terms for the 3rd row/col.
    Returns U3, V3, W3 of shapes (9, 13), (9, 13), (9, 13)
    Rank = 7 + 3 + 3 = 13 (not minimal but exact)
    Actually, we can do better. Let's just use schoolbook for the extra parts.
    """
    n = 3
    # Start with the 2x2 part embedded in top-left
    U3 = [[0]*13 for _ in range(9)]
    V3 = [[0]*13 for _ in range(9)]
    W3 = [[0]*13 for _ in range(9)]
    
    # Copy Strassen 2x2 into top-left (indices 0..3 for each)
    for i in range(4):
        for j in range(7):
            U3[i][j] = U2[i][j]
            V3[i][j] = V2[i][j]
            W3[i][j] = W2[i][j]
    
    # Now we need to handle:
    # C[i][2] for i=0,1,2: involves A[i][0..2] * B[0..2][2]
    # C[2][j] for j=0,1,2: involves A[2][0..2] * B[0..2][j]
    # C[2][2]: involves A[2][0..2] * B[0..2][2]
    
    # Terms 7-9: C[0][2] = A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2]
    # Terms 10-12: C[1][2] = A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2]
    # Terms 13-15: C[2][0] = A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0]
    # Terms 16-18: C[2][1] = A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1]
    # Terms 19-21: C[2][2] = A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2]
    
    # This is getting complex. Let's just use a different approach:
    # Use schoolbook for 3x3 (rank 27) since finding an exact rank-23 is hard.
    # But we can try to use Strassen + some terms.
    
    # Actually, let's just return schoolbook for 3x3 for now.
    # We'll focus on the 3x3x6 structure.
    raise NotImplementedError("Use schoolbook instead")

def _construct_3x3x6():
    """
    Construct a decomposition for 3x3 x 3x6.
    Use the structure: B is 3x6, which is two 3x3 blocks side by side.
    C = A @ B = [A @ B1, A @ B2]
    
    If we have a rank-r decomposition for 3x3 x 3x3:
    U1, V1, W1 of shapes (9, r), (9, r), (9, r)
    
    Then for 3x3 x 3x6:
    U_new shape (9, 2r): [U1, U1]
    V_new shape (18, 2r): [V1_top; V2_top] where V1_top is V1 for first 9 rows, 0 for last 9, etc.
    Actually:
    V_new = [[V1], [0]] for first r columns (B1 part)
    V_new = [[0], [V1]] for next r columns (B2 part)
    W_new shape (18, 2r): [W1, W1]
    
    Wait, let me think more carefully.
    
    For 3x3 x 3x3:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    For 3x3 x 3x6:
    T[i, j, k] where k is 0..5
    Let k = 3 + k' for k' = 0..2 (second block)
    
    For first block (k=0,1,2):
    T[i, j, k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    For second block (k=3,4,5):
    T[i, j, 3+k'] = sum_c U[i,j,c] * V2[j,k',c] * W2[k',i,c]
    
    So we can use:
    U_new = [U1, U1] (same U for both blocks)
    V_new: for first r columns, V[j,k,c] = V1[j,k,c] for k<3, 0 for k>=3
           for next r columns, V[j,k,c] = 0 for k<3, V1[j,k-3,c] for k>=3
    W_new: for first r columns, W[k,i,c] = W1[k,i,c] for k<3, 0 for k>=3
           for next r columns, W[k,i,c] = 0 for k<3, W1[k-3,i,c] for k>=3
    
    This gives rank 2r.
    """
    n, m, p = 3, 3, 6
    
    # Use schoolbook for 3x3 x 3x3 (rank 27)
    U1 = [[0]*27 for _ in range(9)]
    V1 = [[0]*27 for _ in range(9)]
    W1 = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U1[i*3+j][col] = 1
                V1[j*3+k][col] = 1
                W1[k*3+i][col] = 1
                col += 1
    
    # Now embed into 3x3 x 3x6
    r = 27
    U_new = [[0]*(2*r) for _ in range(9)]
    V_new = [[0]*(2*r) for _ in range(18)]
    W_new = [[0]*(2*r) for _ in range(18)]
    
    for c in range(r):
        # First block
        for i in range(9):
            U_new[i][c] = U1[i][c]
        for j in range(3):
            for k in range(3):
                V_new[j*6+k][c] = V1[j*3+k][c]
        for k in range(3):
            for i in range(3):
                W_new[k*3+i][c] = W1[k*3+i][c]
        
        # Second block
        for i in range(9):
            U_new[i][r+c] = U1[i][c]
        for j in range(3):
            for k in range(3):
                V_new[j*6+(k+3)][r+c] = V1[j*3+k][c]
        for k in range(3):
            for i in range(3):
                W_new[(k+3)*3+i][r+c] = W1[k*3+i][c]
    
    return {"U": U_new, "V": V_new, "W": W_new}

def _try_improve(U, V, W, n, m, p, deadline):
    """
    Try to improve the rank using numerical optimization.
    We'll use a simple gradient-free approach: try to zero out some entries
    while maintaining the decomposition.
    """
    start_rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For now, just return the input unchanged if we can't improve
    # A proper implementation would use more sophisticated methods
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Start with the structured decomposition
    best = _construct_3x3x6()
    best_rank = len(best["U"][0])
    
    # Verify
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fall back to schoolbook
        best = _schoolbook()
    
    elapsed = time.time() - start_time
    if elapsed < 80:
        # Try to improve
        improved_U, improved_V, improved_W = _try_improve(
            best["U"], best["V"], best["W"], n, m, p, deadline
        )
        if len(improved_U[0]) < best_rank:
            if _verify(improved_U, improved_V, improved_W, n, m, p):
                best = {"U": improved_U, "V": improved_V, "W": improved_W}
    
    return best
# EVOLVE-BLOCK-END
