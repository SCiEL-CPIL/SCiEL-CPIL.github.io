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

def _kron_product(A, B):
    """Kronecker product of two matrices."""
    ra, ca = len(A), len(A[0])
    rb, cb = len(B), len(B[0])
    result = [[0.0] * (ca * cb) for _ in range(ra * rb)]
    for i in range(ra):
        for j in range(ca):
            if A[i][j] != 0:
                for k in range(rb):
                    for l in range(cb):
                        result[i * rb + k][j * cb + l] = A[i][j] * B[k][l]
    return result

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
    """Kronecker product of two matrices."""
    ra, ca = len(A), len(A[0])
    rb, cb = len(B), len(B[0])
    result = [[0.0] * (ca * cb) for _ in range(ra * rb)]
    for i in range(ra):
        for j in range(ca):
            if A[i][j] != 0:
                for k in range(rb):
                    for l in range(cb):
                        result[i * rb + k][j * cb + l] = A[i][j] * B[k][l]
    return result

def _concat_decomp(d1, d2):
    """Concatenate two decompositions (same row shapes, different ranks)."""
    U1, V1, W1 = d1
    U2, V2, W2 = d2
    r1 = len(U1[0])
    r2 = len(U2[0])
    U = [row[:r1] + row[:r2] for row in U1]  # This won't work directly
    # Actually need to pad
    U = [row + [0.0]*r2 for row in U1]
    V = [row + [0.0]*r2 for row in V1]
    W = [row + [0.0]*r2 for row in W1]
    U2p = [row[:r1] + row for row in U2]
    V2p = [row[:r1] + row for row in V2]
    W2p = [row[:r1] + row for row in W2]
    # Combine
    U_final = [a + b for a, b in zip(U, U2p)]
    V_final = [a + b for a, b in zip(V, V2p)]
    W_final = [a + b for a, b in zip(W, W2p)]
    return U_final, V_final, W_final

def _try_kron_decomposition(deadline):
    """Try decomposition of 4x4x9 as 4x4x4 (Strassen-Kron) + 4x4x4 (Strassen-Kron) + 4x4x1 (schoolbook)."""
    if time.time() > deadline:
        return None
    
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
    
    # Now build 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1
    # The 4x4x4 part uses columns 0..48
    # The second 4x4x4 part uses columns 49..97
    # The 4x4x1 part uses columns 98..113
    
    # For 4x4x1: multiply 4x4 by 4x1 to get 4x1
    # U: 16 rows, V: 4 rows, W: 4 rows
    # Schoolbook: rank 16
    # U[i*4+j, c] = 1, V[j*1+k, c] = 1, W[k*4+i, c] = 1
    # where i,k in {0,1,2,3}, j in {0,1,2,3}
    # c = i*4+j+k... wait, let me think again.
    # For 4x4x1: n=4, m=4, p=1
    # U has n*m = 16 rows, V has m*p = 4 rows, W has p*n = 4 rows
    # Schoolbook: for each (i,j,k) with i in [0,3], j in [0,3], k in [0,0]:
    #   U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # So c = i*4 + j (since k is always 0)
    # U[i*4+j, i*4+j] = 1
    # V[j*1+0, i*4+j] = 1 -> V[j, i*4+j] = 1
    # W[0*4+i, i*4+j] = 1 -> W[i, i*4+j] = 1
    
    rank_total = 49 + 49 + 16
    U = [[0.0] * rank_total for _ in range(16)]
    V = [[0.0] * rank_total for _ in range(16)]
    W = [[0.0] * rank_total for _ in range(16)]
    
    # First 4x4x4 block (columns 0-48)
    for i in range(16):
        for c in range(49):
            U[i][c] = U44[i][c]
            V[i][c] = V44[i][c]
            W[i][c] = W44[i][c]
    
    # Second 4x4x4 block (columns 49-97)
    for i in range(16):
        for c in range(49):
            U[i][49+c] = U44[i][c]
            V[i][49+c] = V44[i][c]
            W[i][49+c] = W44[i][c]
    
    # 4x4x1 block (columns 98-113)
    for i in range(4):
        for j in range(4):
            c = 98 + i*4 + j
            U[i*4+j][c] = 1.0
            V[j][c] = 1.0
            W[i][c] = 1.0
    
    # Verify
    if _verify_decomposition(U, V, W, 4, 4, 9):
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
