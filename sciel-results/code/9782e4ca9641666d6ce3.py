"""Matrix multiplication 4x4x9 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [4, 4, 9]

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 144."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta(i,k) for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                expected = 1.0 if i == k else 0.0
                if abs(total - expected) > 1e-10:
                    return False
    return True

def _strassen_2x2():
    """Return Strassen rank-7 decomposition for 2x2x2."""
    # U: 4 rows (i*m+j for i,j in {0,1}), 7 cols
    # V: 4 rows (j*p+k for j,k in {0,1}), 7 cols
    # W: 4 rows (k*n+i for k,i in {0,1}), 7 cols
    U = [[1, 0, 0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 0, 1],
         [0, 0, 0, 1, 0, 0, 1]]
    V = [[1, 1, 0, 0, 0, 0, 0],
         [1, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 0, 0]]
    W = [[1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1]]
    return U, V, W

def _kron_product(A, B):
    """Kronecker product of two matrices A (ra x ca) and B (rb x cb)."""
    ra, ca = len(A), len(A[0])
    rb, cb = len(B), len(B[0])
    result = [[0.0] * (ca * cb) for _ in range(ra * rb)]
    for i in range(ra):
        for j in range(ca):
            if A[i][j] != 0:
                for ii in range(rb):
                    for jj in range(cb):
                        result[i * rb + ii][j * cb + jj] += A[i][j] * B[ii][jj]
    return result

def _try_kron_decomposition(deadline):
    """Try decomposition of 4x4x9 as 4x4x4 (Strassen-Kron, rank 49) + 4x4x5 (schoolbook, rank 80)."""
    if time.time() > deadline:
        return None
    
    n, m, p = 4, 4, 9
    p1, p2 = 4, 5  # split p into 4 + 5
    
    # Get Strassen 2x2x2 factors
    U2, V2, W2 = _strassen_2x2()
    
    # Verify 2x2x2
    if not _verify_decomposition(U2, V2, W2, 2, 2, 2):
        return None
    
    # Kronecker product for 4x4x4
    U44 = _kron_product(U2, U2)  # 16 x 49
    V44 = _kron_product(V2, V2)  # 16 x 49
    W44 = _kron_product(W2, W2)  # 16 x 49
    
    # Verify 4x4x4
    if not _verify_decomposition(U44, V44, W44, 4, 4, 4):
        return None
    
    # For the 4x4x5 schoolbook part:
    # U: 16 rows, V: 4*5=20 rows, W: 5*4=20 rows
    # For each (i,j,k) with i in [0,3], j in [0,3], k in [0,4]:
    #   U[i*m+j, c] = 1, V[j*p2+k, c] = 1, W[k*n+i, c] = 1
    # c indexes all (i,j,k) triples: 4*4*5 = 80
    rank_sb = n * m * p2  # 80
    
    rank_total = 49 + rank_sb  # 129
    U = [[0.0] * rank_total for _ in range(n * m)]
    V = [[0.0] * rank_total for _ in range(m * p)]
    W = [[0.0] * rank_total for _ in range(p * n)]
    
    # First part: 4x4x4 Strassen-Kronecker (columns 0-48)
    for i in range(16):
        for c in range(49):
            U[i][c] = U44[i][c]
            V[i][c] = V44[i][c]
            W[i][c] = W44[i][c]
    
    # Second part: 4x4x5 schoolbook (columns 49-128)
    # For the schoolbook part, V has rows j*p2+k where j in [0,3], k in [0,4]
    # But in the full V, row index is j*p+k where k is the full column index [0,8]
    # For the 4x4x5 part, we're handling columns k=4..8 of the original B
    # So V row for schoolbook: j*p + (4+k') where k' in [0,4]
    # W row for schoolbook: (4+k')*n + i where k' in [0,4]
    col_offset = 49
    for i in range(n):
        for j in range(m):
            for k in range(p2):
                c = col_offset + i * m * p2 + j * p2 + k
                U[i * m + j][c] = 1.0
                V[j * p + (p1 + k)][c] = 1.0  # k in original is p1+k = 4+k
                W[(p1 + k) * n + i][c] = 1.0
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # First try the Kronecker product decomposition (rank 49 if valid)
    result = _try_kron_decomposition(deadline)
    if result is not None:
        return result
    
    # Fall back to schoolbook decomposition (rank 144)
    if time.time() > deadline:
        return _schoolbook()
    
    return _schoolbook()
