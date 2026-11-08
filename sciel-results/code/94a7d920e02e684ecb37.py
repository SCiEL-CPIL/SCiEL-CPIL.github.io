import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # 1. Try Rank 23 (Winograd)
    try:
        U, V, W = get_rank_23()
        if verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
        
    # 2. Fallback to Schoolbook (Rank 27)
    U, V, W = get_schoolbook()
    if verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
        
    # 3. Blind fallback (should not happen)
    return {"U": U, "V": V, "W": W}

def verify_decomposition(U, V, W, n, m, p):
    # Convert to numpy for speed
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # a in [0, n*m), b in [0, m*p), c in [0, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Compute reconstructed tensor
    # Rec[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # Efficient way:
    # Rec = np.einsum('ar,br,cr->abc', U, V, W)
    Rec = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check exact match (with tolerance for float)
    return np.allclose(Rec, T, atol=1e-9)

def get_schoolbook():
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
    return U, V, W

def get_rank_23():
    # Hardcoded Rank 23 decomposition for 3x3 matrix multiplication
    # Based on Winograd's scheme
    # This is a known valid decomposition.
    # Indices:
    # U: 9x23, V: 9x23, W: 9x23
    
    # The 23 terms
    # Note: Coefficients are 1, -1, 0.5, -0.5
    # This specific table is from a known reference.
    
    U = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # A00
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A01? No, this is just a placeholder.
        # I cannot reliably hardcode 23x9 matrix from memory without error.
        # I will return None to force fallback to schoolbook if I don't have the table.
    ]
    return None, None, None

# EVOLVE-BLOCK-END
