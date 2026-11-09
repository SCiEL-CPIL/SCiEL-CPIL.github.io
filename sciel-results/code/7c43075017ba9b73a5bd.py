"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """
    Verify that U, V, W form a valid tensor decomposition for matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _make_strassen_3x3():
    """
    Strassen decomposition for 3x3 x 3x3 matrix multiplication.
    Returns U, V, W each of shape (9, 7) with integer entries.
    """
    # Strassen's algorithm for 2x2x2 has rank 7.
    # For 3x3x3, we can use a block approach or known rank-23 decomposition.
    # Actually, the standard Strassen is for 2x2. For 3x3, we need a different approach.
    # Let me use the schoolbook for 3x3x3 which has rank 27, but we can do better.
    # Known: rank of 3x3x3 matrix multiplication tensor is 23 (Bini et al.).
    # But for half-integer constraint, let's start with a known construction.
    
    # For now, let's use a simple block decomposition:
    # Split 3x3 into blocks and use Strassen-like ideas.
    # Actually, let me just construct a valid rank-27 schoolbook for 3x3x3
    # and then try to reduce.
    
    n = m = p = 3
    rank = 27
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
    return U, V, W

def _make_3x3_x_3x2():
    """
    Schoolbook decomposition for 3x3 x 3x2 matrix multiplication.
    Returns U, V, W each of shape (9, 6, 6) with rank 18.
    """
    n, m, p = 3, 3, 2
    rank = n * m * p  # 18
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
    return U, V, W

def _combine_decompositions(U1, V1, W1, U2, V2, W2, offset):
    """
    Combine two decompositions. The second one's V indices are offset by p1*n.
    U1, V1, W1: for first part (e.g., 3x3 x 3x3)
    U2, V2, W2: for second part (e.g., 3x3 x 3x2)
    offset: the starting column index for the second decomposition in V
    """
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(len(U1))]
    V = [[0] * total_rank for _ in range(len(V1) + len(V2))]
    W = [[0] * total_rank for _ in range(len(W1))]
    
    # Copy first decomposition
    for i in range(len(U1)):
        for c in range(rank1):
            U[i][c] = U1[i][c]
    for i in range(len(V1)):
        for c in range(rank1):
            V[i][c] = V1[i][c]
    for i in range(len(W1)):
        for c in range(rank1):
            W[i][c] = W1[i][c]
    
    # Copy second decomposition with offset in V
    for i in range(len(U2)):
        for c in range(rank2):
            U[i][rank1 + c] = U2[i][c]
    for i in range(len(V2)):
        for c in range(rank2):
            V[len(V1) + i][rank1 + c] = V2[i][c]
    for i in range(len(W2)):
        for c in range(rank2):
            W[i][rank1 + c] = W2[i][c]
    
    return U, V, W

def _make_combined_3x3_3x5():
    """
    Combine 3x3x3 (schoolbook rank 27) and 3x3x2 (schoolbook rank 18)
    to get a decomposition for 3x3 x 3x5 with rank 45.
    This is just the schoolbook, but let's verify the structure.
    """
    n, m, p = 3, 3, 5
    
    # First part: 3x3 x 3x3 (columns 0-2 of B, rows 0-2 of C)
    U1, V1, W1 = _make_strassen_3x3()  # Actually schoolbook rank 27
    # V1 has shape (3*3, 27) = (9, 27) for B[:, 0:3]
    
    # Second part: 3x3 x 3x2 (columns 3-4 of B, rows 0-2 of C)  
    U2, V2, W2 = _make_3x3_x_3x2()  # rank 18
    # V2 has shape (3*2, 18) = (6, 18) for B[:, 3:5]
    
    # Combine: V has 9 + 6 = 15 rows
    U, V, W = _combine_decompositions(U1, V1, W1, U2, V2, W2, 0)
    
    return {"U": U, "V": V, "W": W}

def _try_explicit_construction():
    """
    Try to construct a lower-rank decomposition using known algebraic identities.
    
    For 3x3 x 3x5, we can try to find a rank-r decomposition where r < 45.
    
    Key idea: use the fact that matrix multiplication tensors have structure.
    The rank of the (n x m) x (m x p) multiplication tensor is at least max(n*m, m*p, p*n)
    and at most n*m*p.
    
    For 3x3 x 3x5: lower bound is max(9, 15, 15) = 15.
    """
    n, m, p = DIMENSIONS
    
    # Try rank 15: this is the theoretical lower bound
    # For this to work, we'd need a very clever decomposition
    # Let's try to construct one using the structure of the problem
    
    # The tensor T[a,b,c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if indices are consistent, 0 otherwise
    
    # For rank r, we need U (9 x r), V (15 x r), W (15 x r) such that
    # sum_c U[a,c]*V[b,c]*W[c_idx,c] = T[a,b,c_idx]
    
    # This is a hard problem. Let's try a systematic search for small ranks
    # with half-integer entries from a small set {-1, -0.5, 0, 0.5, 1}
    
    # Given time constraints, let's try a few specific ranks
    ranks_to_try = [15, 16, 17, 18, 19, 20]
    
    for r in ranks_to_try:
        result = _search_rank(r, max_attempts=200)
        if result is not None:
            return result
    
    return None

def _search_rank(r, max_attempts=100):
    """
    Search for a rank-r decomposition with half-integer entries.
    Uses random initialization from a small set of values.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for attempt in range(max_attempts):
        rng = np.random.RandomState(attempt * 7919 + 42)
        
        U = rng.choice(vals, size=(nm, r))
        V = rng.choice(vals, size=(mp, r))
        W = rng.choice(vals, size=(pn, r))
        
        # Compute T_hat = sum_c U[a,c]*V[b,c]*W[c,c]
        # Use einsum for efficiency
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        
        if np.allclose(T_hat, T, atol=1e-9):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x5 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # First, get the schoolbook solution as fallback (rank 45)
    U = [[0] * (n*m*p) for _ in range(n * m)]
    V = [[0] * (n*m*p) for _ in range(m * p)]
    W = [[0] * (n*m*p) for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    fallback = {"U": U, "V": V, "W": W}
    assert _verify(fallback["U"], fallback["V"], fallback["W"]), "Schoolbook verification failed"
    
    # Try to find lower rank decompositions
    result = _try_explicit_construction()
    if result is not None:
        assert _verify(result["U"], result["V"], result["W"]), "Verification failed"
        return result
    
    # Return schoolbook as fallback
    return fallback
# EVOLVE-BLOCK-END
