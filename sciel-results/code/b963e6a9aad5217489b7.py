import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 138.0  # seconds; leave 12s margin before the 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    T = np.einsum('at,bt,ct->abc', U, V, W)
    # Expected: T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i*m+j, j*p+k, k*n+i] - 1.0) > tol:
                    return False
    # All other entries should be 0
    expected_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected_mask[i*m+j, j*p+k, k*n+i] = True
    if np.max(np.abs(T[~expected_mask])) > tol:
        return False
    return True

def _build_rank_55():
    """
    Constructs the rank-55 decomposition.
    Split m=3 into m1=2 (Strassen) and m2=1 (Schoolbook).
    Split p=10 into 5 blocks of 2 for the Strassen part.
    """
    n, m, p = 2, 3, 10
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Indices for A (2x3)
    # A[i,j] -> U[i*m + j]
    # A00 -> 0, A01 -> 1, A02 -> 2
    # A10 -> 3, A11 -> 4, A12 -> 5
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5
    
    # Process 5 blocks of 2 columns for the Strassen part (A_L * B_L)
    # A_L is 2x2 (cols 0,1), B_L is 2x10 (rows 0,1)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Indices for B (3x10)
        # B[j,k] -> V[j*p + k]
        # B00 -> V[0], B01 -> V[1]
        # B10 -> V[10], B11 -> V[11]
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        
        # Indices for C (2x10 output)
        # C[i,k] -> W[k*n + i]
        # C00 -> W[0], C01 -> W[2]
        # C10 -> W[1], C11 -> W[3]
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        
        # Strassen's algorithm for 2x2 matrix multiplication
        # M1 = (A00 + A11)(B00 + B11)
        U[a00, col] += 1.0
        U[a11, col] += 1.0
        V[b00, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M2 = (A10 + A11)B00
        U[a10, col] += 1.0
        U[a11, col] += 1.0
        V[b00, col] += 1.0
        W[c10, col] += 1.0
        W[c11, col] -= 1.0
        col += 1
        
        # M3 = A00(B01 - B11)
        U[a00, col] += 1.0
        V[b01, col] += 1.0
        V[b11, col] -= 1.0
        W[c01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M4 = A11(B10 - B00)
        U[a11, col] += 1.0
        V[b10, col] += 1.0
        V[b00, col] -= 1.0
        W[c00, col] += 1.0
        W[c10, col] += 1.0
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a00, col] += 1.0
        U[a01, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] -= 1.0
        W[c01, col] += 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a10, col] += 1.0
        U[a00, col] -= 1.0
        V[b00, col] += 1.0
        V[b01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a01, col] += 1.0
        U[a11, col] -= 1.0
        V[b10, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] += 1.0
        col += 1
    
    # Schoolbook part for A_R * B_R (2x1 * 1x10)
    # A_R is 2x1: A02 (idx 2), A12 (idx 5)
    # B_R is 1x10: B20..B29 (idx 20..29)
    # C contributions: C00..C09 (idx 0,2..18), C10..C19 (idx 1,3..19)
    
    for k in range(p):
        b2k = 2 * p + k  # Index for B[2, k]
        
        # Term 1: A02 * B2k -> C0k
        # C0k is at W[k*n + 0] = W[2*k]
        U[a02, col] += 1.0
        V[b2k, col] += 1.0
        W[2 * k, col] += 1.0
        col += 1
        
        # Term 2: A12 * B2k -> C1k
        # C1k is at W[k*n + 1] = W[2*k + 1]
        U[a12, col] += 1.0
        V[b2k, col] += 1.0
        W[2 * k + 1, col] += 1.0
        col += 1
        
    return U, V, W

def construct():
    start_time = time.time()
    
    # Build the rank-55 decomposition
    U, V, W = _build_rank_55()
    
    # Verify
    n, m, p = 2, 3, 10
    if time.time() - start_time < DEADLINE:
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback (should not happen)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
