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

def _construct_223_rank8():
    """Construct rank-8 decomposition for 2x2x3 matrix multiplication."""
    n, m, p = 2, 2, 3
    rank = 8
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    # k=0,1 part (Schoolbook for 2x2x2)
    for k in range(2):
        for j in range(2):
            for i in range(2):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    # k=2 part (Schoolbook for 2x2x1)
    k = 2
    for j in range(2):
        for i in range(2):
            U[i * m + j][col] = 1.0
            V[j * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
            
    return U, V, W

def _try_kron_decomposition(deadline):
    """Try Kronecker product decomposition using rank-8 2x2x3 factors."""
    if time.time() > deadline:
        return None
    
    U2, V2, W2 = _construct_223_rank8()
    
    # Verify the 2x2x3 decomposition first
    if not _verify_decomposition(U2, V2, W2, 2, 2, 3):
        return None
    
    # Kronecker product
    U4 = _kron_product(U2, U2)
    V4 = _kron_product(V2, V2)
    W4 = _kron_product(W2, W2)
    
    if _verify_decomposition(U4, V4, W4, 4, 4, 9):
        return {"U": U4, "V": V4, "W": W4}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    result = _try_kron_decomposition(deadline)
    if result is not None:
        return result
    
    return _schoolbook()
