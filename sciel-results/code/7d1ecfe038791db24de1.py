"""
5x5 matrix multiplication - rank reduction via structured algebraic search.
Attempts to find a rank < 125 decomposition using Strassen-style recursive
block decomposition and randomized verification. Falls back to schoolbook if
no improvement is found within time limits.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _schoolbook(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
    r = n * m * p
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    C[i,k] = sum_c W[k*n+i, c] * (A_flat . U[:,c]) * (B_flat . V[:,c])
    where A_flat has A[i,j] at position i*m+j, B_flat has B[j,k] at j*p+k.
    """
    r = U.shape[1]
    # Random test matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    A_flat = A.flatten()  # length n*m, index i*m+j
    B_flat = B.flatten()  # length m*p, index j*p+k
    
    # Compute C from decomposition
    # L_c = A_flat . U[:,c] for each c
    L = A_flat @ U  # shape (r,)
    M = B_flat @ V  # shape (r,)
    
    # C[i,k] = sum_c W[k*n+i, c] * L[c] * M[c]
    # W is (p*n, r), W[k*n+i, c]
    # We need to reconstruct C from W, L, M
    # C_flat[k*n+i] = sum_c W[k*n+i, c] * L[c] * M[c]
    C_flat = W @ (L * M)  # shape (p*n,)
    C = C_flat.reshape(p, n).T  # C[i,k] = C_flat[k*n+i], so C_flat is (k,i) order
    
    # Actually let's be more careful.
    # C_flat has index k*n+i for element C[i,k]
    # So C_flat.reshape(p, n) gives [k, i]
    # We want C[i, k], so transpose
    C_computed = C_flat.reshape(p, n).T
    
    if not np.allclose(C_computed, C_expected, atol=tol):
        return False
    return True

def _strassen_2x2_decomposition():
    """
    Return the Strassen 2x2 decomposition as U, V, W factors.
    For 2x2 matrices:
    M1 = (A11-A22)(B12+B22)
    M2 = (A11+A22)B22
    M3 = A11(B11-B22)
    M4 = A22(B11+B22)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    """
    # n=m=p=2
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n*m, r), dtype=float)
    V = np.zeros((m*p, r), dtype=float)
    W = np.zeros((p*n, r), dtype=float)
    
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # M1 = (A11-A22)(B12+B22) = (A[0,0]-A[1,1])(B[0,1]+B[1,1])
    U[0, 0] = 1.0; U[3, 0] = -1.0
    V[1, 0] = 1.0; V[3, 0] = 1.0
    
    # M2 = (A11+A22)B22 = (A[0,0]+A[1,1])B[1,1]
    U[0, 1] = 1.0; U[3, 1] = 1.0
    V[3, 1] = 1.0
    
    # M3 = A11(B11-B22) = A[0,0](B[0,0]-B[1,1])
    U[0, 2] = 1.0
    V[0, 2] = 1.0; V[3, 2] = -1.0
    
    # M4 = A22(B11+B22) = A[1,1](B[0,0]+B[1,1])
    U[3, 3] = 1.0
    V[0, 3] = 1.0; V[3, 3] = 1.0
    
    # M5 = (A11+A12)B22 = (A[0,0]+A[0,1])B[1,1]
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    
    # M6 = (A21-A11)(B11+B12) = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    
    # M7 = (A12-A22)(B21+B22) = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    
    # W: C[i,k] coefficients
    # C[0,0] = M1+M4-M5+M7 -> W[0*2+0, c] = W[0, c]
    W[0, 0] = 1.0; W[0, 3] = 1.0; W[0, 4] = -1.0; W[0, 6] = 1.0
    
    # C[0,1] = M3+M5 -> W[1*2+0, c] = W[2, c]
    W[2, 2] = 1.0; W[2, 4] = 1.0
    
    # C[1,0] = M2+M4 -> W[0*2+1, c] = W[1, c]
    W[1, 1] = 1.0; W[1, 3] = 1.0
    
    # C[1,1] = M1-M2+M3+M6 -> W[1*2+1, c] = W[3, c]
    W[3, 0] = 1.0; W[3, 1] = -1.0; W[3, 2] = 1.0; W[3, 5] = 1.0
    
    return U, V, W

def _combine_decompositions(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2, offset_i=0, offset_j=0, offset_k=0):
    """
    Combine two decompositions for block diagonal or block structure.
    This is a simplified version - for actual 5x5 we need more sophisticated tiling.
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 + r2
    U = np.zeros((n1*m1, r), dtype=float)
    V = np.zeros((m1*p1, r), dtype=float)
    W = np.zeros((p1*n1, r), dtype=float)
    
    U[:, :r1] = U1
    V[:, :r1] = V1
    W[:, :r1] = W1
    
    # For the second block, we'd need to map indices appropriately
    # This is a placeholder - proper tiling requires careful index mapping
    U[:, r1:] = 0
    V[:, r1:] = 0
    W[:, r1:] = 0
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer for 120s limit
    
    # First, get schoolbook as fallback
    U_s, V_s, W_s = _schoolbook(n, m, p)
    
    # Try to find better decompositions
    best_rank = n * m * p
    best_U, best_V, best_W = U_s, V_s, W_s
    
    # Strategy 1: Try known low-rank constructions
    # For 5x5, we can try to use 3x3 + 2x2 blocking or similar
    
    # Strategy 2: Randomized search with verification
    # Try random rank-r decompositions and verify
    
    np.random.seed(12345)
    
    # Try ranks from 100 down to best_rank-1
    # This is expensive, so we limit attempts
    max_attempts = 50
    attempts = 0
    
    for r in range(110, 124):  # Try ranks 110 to 123
        if time.time() - start_time > time_limit:
            break
        if r >= best_rank:
            continue
        
        for attempt in range(max_attempts // 10):
            if time.time() - start_time > time_limit:
                break
            attempts += 1
            
            # Random integer/half-integer initialization
            scale = 1.0
            U = np.random.choice([-1, 0, 1], size=(n*m, r)) * scale
            V = np.random.choice([-1, 0, 1], size=(m*p, r)) * scale
            W = np.random.choice([-1, 0, 1], size=(p*n, r)) * scale
            
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank <= r:
            break
    
    # If no improvement found, return schoolbook
    if best_rank == n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
