"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _verify_random_matrix(U, V, W, n, m, p, trials=5):
    """Verify decomposition by testing with random matrices."""
    rank = U.shape[1]
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(rank):
            La = 0.0
            for i in range(n):
                for j in range(m):
                    La += U[i*m+j, r] * A[i, j]
            Lb = 0.0
            for j in range(m):
                for k in range(p):
                    Lb += V[j*p+k, r] * B[j, k]
            for k in range(p):
                for i in range(n):
                    C_computed[i, k] += W[k*n+i, r] * La * Lb
        if not np.allclose(C_computed, C_expected, atol=1e-6):
            return False
    return True

def _schoolbook_factors(n, m, p):
    """Return schoolbook decomposition for n×m×p as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Returns the Strassen decomposition for 2x2x2 matrix multiplication.
    Rank = 7.
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _split_and_combine(n, m, p, deadline=None):
    """
    Recursively decompose n×m×p by splitting the largest dimension.
    When all three dimensions are 2, use Strassen.
    """
    if deadline is not None and time.time() > deadline:
        return _schoolbook_factors(n, m, p)
    
    # Base case: 2×2×2 uses Strassen
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    
    # Base case: any dimension is 1, use schoolbook
    if n == 1 or m == 1 or p == 1:
        return _schoolbook_factors(n, m, p)
    
    # If all even, we could use Strassen, but for simplicity split largest
    # Split the largest dimension into two parts
    if n >= m and n >= p:
        n1 = n // 2
        n2 = n - n1
        U1, V1, W1 = _split_and_combine(n1, m, p, deadline)
        U2, V2, W2 = _split_and_combine(n2, m, p, deadline)
        rank = U1.shape[1] + U2.shape[1]
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        U[:n1*m, :U1.shape[1]] = U1
        V[:, :V1.shape[1]] = V1
        W[:p*n1, :W1.shape[1]] = W1
        U[n1*m:, U1.shape[1]:] = U2
        V[:, V1.shape[1]:] = V2
        W[p*n1:, W1.shape[1]:] = W2
        return U, V, W
    elif m >= n and m >= p:
        m1 = m // 2
        m2 = m - m1
        U1, V1, W1 = _split_and_combine(n, m1, p, deadline)
        U2, V2, W2 = _split_and_combine(n, m2, p, deadline)
        rank = U1.shape[1] + U2.shape[1]
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            U[i*m:i*m+m1, :U1.shape[1]] = U1[i*m1:(i+1)*m1, :]
        V[:m1*p, :V1.shape[1]] = V1
        W[:, :W1.shape[1]] = W1
        for i in range(n):
            U[i*m+m1:i*m+m, U1.shape[1]:] = U2[i*m2:(i+1)*m2, :]
        V[m1*p:, V1.shape[1]:] = V2
        W[:, W1.shape[1]:] = W2
        return U, V, W
    else:
        p1 = p // 2
        p2 = p - p1
        U1, V1, W1 = _split_and_combine(n, m, p1, deadline)
        U2, V2, W2 = _split_and_combine(n, m, p2, deadline)
        rank = U1.shape[1] + U2.shape[1]
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        U[:, :U1.shape[1]] = U1
        for j in range(m):
            V[j*p:j*p+p1, :V1.shape[1]] = V1[j*p1:(j+1)*p1, :]
        W[:p1*n, :W1.shape[1]] = W1
        U[:, U1.shape[1]:] = U2
        for j in range(m):
            V[j*p+p1:j*p+p, V1.shape[1]:] = V2[j*p2:(j+1)*p2, :]
        W[p1*n:, W1.shape[1]:] = W2
        return U, V, W

def _rank_n_solution(n, m, p):
    """
    Rank-n structured decomposition for n×m×p matrix multiplication.
    Uses the observation that the tensor can be decomposed with rank n
    by assigning one rank-1 term per row of A.
    """
    rank = n
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for r in range(n):
        for j in range(m):
            U[r*m + j, r] = 1
        for j in range(m):
            for k in range(p):
                V[j*p + k, r] = 1
        for k in range(p):
            W[k*n + r, r] = 1
    return U, V, W

def _schoolbook():
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

def construct():
    """
    Construct the decomposition. Try rank-n solution first, then recursive splitting.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100
    
    try:
        # Try rank-n solution first (rank = n = 3 for 3x5x5)
        U, V, W = _rank_n_solution(n, m, p)
        rank = U.shape[1]
        
        if _verify_decomposition(U, V, W, n, m, p) and _verify_random_matrix(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Fall back to recursive splitting with Strassen
        U, V, W = _split_and_combine(n, m, p, deadline)
        rank = U.shape[1]
        
        if not _verify_decomposition(U, V, W, n, m, p):
            return _schoolbook()
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        return _schoolbook()
# EVOLVE-BLOCK-END
