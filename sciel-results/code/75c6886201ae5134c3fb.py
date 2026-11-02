"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook_decomp(n, m, p):
    """Return schoolbook decomposition for n x m @ m x p."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2 @ 2x2 (rank 7)."""
    # Strassen's algorithm: 7 multiplications instead of 8
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices: A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A @ B
    # U rows: a11, a12, a21, a22
    # V rows: b11, b12, b21, b22
    # W rows: c11, c12, c21, c22
    
    # M1 = a11 * (b21 - b11)
    U[0, 0] = 1.0; V[2, 0] = 1.0; V[0, 0] = -1.0
    W[2, 0] = 1.0; W[3, 0] = 1.0
    
    # M2 = (a11 + a12) * b11
    U[0, 1] = 1.0; U[1, 1] = 1.0; V[0, 1] = 1.0
    W[0, 1] = 1.0; W[2, 1] = 1.0
    
    # M3 = (a21 + a22) * b22
    U[2, 2] = 1.0; U[3, 2] = 1.0; V[3, 2] = 1.0
    W[1, 2] = 1.0; W[3, 2] = 1.0
    
    # M4 = a11 * (b12 - b22)
    U[0, 3] = 1.0; V[1, 3] = 1.0; V[3, 3] = -1.0
    W[0, 3] = 1.0; W[1, 3] = 1.0
    
    # M5 = (a12 - a22) * (b21 + b22)
    U[1, 4] = 1.0; U[3, 4] = -1.0; V[2, 4] = 1.0; V[3, 4] = 1.0
    W[1, 4] = 1.0; W[3, 4] = 1.0
    
    # M6 = (a11 - a21) * (b11 + b21)
    U[0, 5] = 1.0; U[2, 5] = -1.0; V[0, 5] = 1.0; V[2, 5] = 1.0
    W[0, 5] = 1.0; W[2, 5] = 1.0
    
    # M7 = (a22 - a11) * (b12 + b22)
    U[3, 6] = 1.0; U[0, 6] = -1.0; V[1, 6] = 1.0; V[3, 6] = 1.0
    W[1, 6] = 1.0; W[3, 6] = 1.0
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    # The tensor T[i,j,k] should equal sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For each (i,j,k) triple, check if the sum matches 1.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(U.shape[1]):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _block_decompose(n, m, p, max_depth=3):
    """
    Recursively decompose n x m @ m x p using block methods.
    Returns (U, V, W) arrays.
    """
    # Base case: use schoolbook for small dimensions
    if n <= 2 and m <= 2 and p <= 2:
        if n == 2 and m == 2 and p == 2:
            return _strassen_2x2x2()
        else:
            return _schoolbook_decomp(n, m, p)
    
    # Try to split along the smallest dimension
    # We'll try splitting n, m, or p and see which gives the best rank
    
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Option 1: Split n into two parts
    if n >= 2:
        for n1 in range(1, n):
            n2 = n - n1
            if n1 < 1 or n2 < 1:
                continue
            U1, V1, W1 = _block_decompose(n1, m, p, max_depth - 1)
            U2, V2, W2 = _block_decompose(n2, m, p, max_depth - 1)
            
            # Combine: U has rows for both blocks
            rank = U1.shape[1] + U2.shape[1]
            U = np.vstack([U1, U2])
            V = np.hstack([V1, V2])
            W = np.hstack([W1, W2])
            
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
    
    # Option 2: Split m into two parts
    if m >= 2:
        for m1 in range(1, m):
            m2 = m - m1
            if m1 < 1 or m2 < 1:
                continue
            U1, V1, W1 = _block_decompose(n, m1, p, max_depth - 1)
            U2, V2, W2 = _block_decompose(n, m2, p, max_depth - 1)
            
            rank = U1.shape[1] + U2.shape[1]
            U = np.hstack([U1, U2])
            V = np.vstack([V1, V2])
            W = np.hstack([W1, W2])
            
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
    
    # Option 3: Split p into two parts
    if p >= 2:
        for p1 in range(1, p):
            p2 = p - p1
            if p1 < 1 or p2 < 1:
                continue
            U1, V1, W1 = _block_decompose(n, m, p1, max_depth - 1)
            U2, V2, W2 = _block_decompose(n, m, p2, max_depth - 1)
            
            rank = U1.shape[1] + U2.shape[1]
            U = np.hstack([U1, U2])
            V = np.hstack([V1, V2])
            W = np.vstack([W1, W2])
            
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
    
    if best_U is None:
        return _schoolbook_decomp(n, m, p)
    
    # Verify the decomposition
    if _verify_decomp(best_U, best_V, best_W, n, m, p):
        return best_U, best_V, best_W
    
    # If verification fails, fall back to schoolbook
    return _schoolbook_decomp(n, m, p)

def construct():
    deadline = time.time() + 120
    
    n, m, p = 4, 5, 6
    U_rows = n * m  # 20
    V_rows = m * p  # 30
    W_rows = p * n  # 24
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomp(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    # Try block decomposition with Strassen
    try:
        U2, V2, W2 = _block_decompose(n, m, p, max_depth=4)
        if U2.shape[1] < best_rank:
            if _verify_decomp(U2, V2, W2, n, m, p):
                best_rank = U2.shape[1]
                best_U, best_V, best_W = U2, V2, W2
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5:
        # Convert to lists for return
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try additional optimizations if time permits
    # For now, return the best found
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
