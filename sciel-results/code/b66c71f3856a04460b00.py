import time
import numpy as np

DIMENSIONS = [4, 4, 8]
_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _strassen_222():
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _kron_decomposition(U1, V1, W1, U2, V2, W2):
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    m1, _ = V1.shape
    m2, _ = V2.shape
    p1, _ = W1.shape
    p2, _ = W2.shape
    
    R = r1 * r2
    U = np.zeros((n1 * n2, R), dtype=float)
    V = np.zeros((m1 * m2, R), dtype=float)
    W = np.zeros((p1 * p2, R), dtype=float)
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r_idx = r1_idx * r2 + r2_idx
            U[:, r_idx] = np.kron(U1[:, r1_idx], U2[:, r2_idx])
            V[:, r_idx] = np.kron(V1[:, r1_idx], V2[:, r2_idx])
            W[:, r_idx] = np.kron(W1[:, r1_idx], W2[:, r2_idx])
            
    return U, V, W

def _direct_sum_decomposition(U1, V1, W1):
    n, r = U1.shape
    m, _ = V1.shape
    p, _ = W1.shape
    
    R = 2 * r
    U = np.zeros((n, R), dtype=float)
    V = np.zeros((2 * m, R), dtype=float)
    W = np.zeros((2 * p, R), dtype=float)
    
    U[:, :r] = U1
    V[:m, :r] = V1
    W[:p, :r] = W1
    
    U[:, r:] = U1
    V[m:, r:] = V1
    W[p:, r:] = W1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    R = U.shape[1]
    tol = 1e-9
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _remove_redundant_columns(U, V, W, n, m, p):
    """
    Deterministically remove columns that are linear combinations of others.
    Condition: U[:, k] in span(U[:, others]), V[:, k] in span(V[:, others]), W[:, k] in span(W[:, others]).
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    current_rank = U.shape[1]
    
    # We iterate and try to remove the last column if it's dependent on the first current_rank-1
    # Then repeat.
    # This is O(R^2 * N) roughly.
    # R is 98, N is ~32. 98^2 * 32 ~ 300k. Fast.
    
    changed = True
    while changed:
        changed = False
        current_rank = U.shape[1]
        if current_rank <= 1:
            break
            
        # Try to remove the last column
        # Check if U[:, -1] is in the span of U[:, :-1]
        # And V[:, -1] in span(V[:, :-1])
        # And W[:, -1] in span(W[:, :-1])
        
        # Use SVD or rank check
        # Rank of [U[:, :-1], U[:, -1]] should equal Rank of U[:, :-1]
        
        U_rest = U[:, :-1]
        V_rest = V[:, :-1]
        W_rest = W[:, :-1]
        
        U_test = np.hstack([U_rest, U[:, -1:]])
        V_test = np.hstack([V_rest, V[:, -1:]])
        W_test = np.hstack([W_rest, W[:, -1:]])
        
        r_U_rest = np.linalg.matrix_rank(U_rest)
        r_V_rest = np.linalg.matrix_rank(V_rest)
        r_W_rest = np.linalg.matrix_rank(W_rest)
        
        r_U_test = np.linalg.matrix_rank(U_test)
        r_V_test = np.linalg.matrix_rank(V_test)
        r_W_test = np.linalg.matrix_rank(W_test)
        
        if r_U_rest == r_U_test and r_V_rest == r_V_test and r_W_rest == r_W_test:
            # The last column is redundant
            U = U[:, :-1]
            V = V[:, :-1]
            W = W[:, :-1]
            changed = True
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    U2, V2, W2 = _strassen_222()
    U444, V444, W444 = _kron_decomposition(U2, V2, W2, U2, V2, W2)
    U, V, W = _direct_sum_decomposition(U444, V444, W444)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback
        schoolbook_rank = n * m * p
        U_school = np.zeros((n*m, schoolbook_rank))
        V_school = np.zeros((m*p, schoolbook_rank))
        W_school = np.zeros((p*n, schoolbook_rank))
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_school[ii*m + jj, col] = 1
                    V_school[jj*p + kk, col] = 1
                    W_school[kk*n + ii, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Deterministic reduction
    U, V, W = _remove_redundant_columns(U, V, W, n, m, p)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback
        schoolbook_rank = n * m * p
        U_school = np.zeros((n*m, schoolbook_rank))
        V_school = np.zeros((m*p, schoolbook_rank))
        W_school = np.zeros((p*n, schoolbook_rank))
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_school[ii*m + jj, col] = 1
                    V_school[jj*p + kk, col] = 1
                    W_school[kk*n + ii, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
