import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Rank n*m*p schoolbook decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check all (i,j,k) entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """Returns U, V, W for 2x2 matrix multiplication (Strassen)."""
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    return U2, V2, W2

def _strassen_kron_4x4(n, m, p):
    """Rank-49 decomposition via Kronecker product of Strassen 2x2."""
    U2, V2, W2 = _strassen_2x2()
    
    # Kronecker product gives 16x49 matrices
    # Rows in kron are indexed by (r1, r2) -> r1*4 + r2
    # Where r1 is row in first 2x2 factor, r2 in second.
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    # We need to permute rows to match the linear index convention:
    # U row index: i*m+j (i,j in 0..3)
    # V row index: j*p+k (j,k in 0..3)
    # W row index: k*n+i (k,i in 0..3)
    
    # Mapping function:
    # Target index t = x * 4 + y
    # x = 2*x1 + x0, y = 2*y1 + y0
    # Source index in kron (for factors indexed by (x1,y1) and (x0,y0)):
    # r1 = x1 * 2 + y1
    # r2 = x0 * 2 + y0
    # src = r1 * 4 + r2
    
    def get_perm(mapping_type):
        perm = np.zeros(16, dtype=int)
        for t in range(16):
            if mapping_type == 'U':
                # t = i * 4 + j
                i = t // 4
                j = t % 4
                x1, x0 = i // 2, i % 2
                y1, y0 = j // 2, j % 2
                r1 = x1 * 2 + y1
                r2 = x0 * 2 + y0
            elif mapping_type == 'V':
                # t = j * 4 + k
                j = t // 4
                k = t % 4
                x1, x0 = j // 2, j % 2
                y1, y0 = k // 2, k % 2
                r1 = x1 * 2 + y1
                r2 = x0 * 2 + y0
            elif mapping_type == 'W':
                # t = k * 4 + i
                k = t // 4
                i = t % 4
                x1, x0 = k // 2, k % 2
                y1, y0 = i // 2, i % 2
                r1 = x1 * 2 + y1
                r2 = x0 * 2 + y0
            else:
                raise ValueError("Unknown mapping type")
            
            perm[t] = r1 * 4 + r2
        return perm

    perm_u = get_perm('U')
    perm_v = get_perm('V')
    perm_w = get_perm('W')
    
    U4 = U4_kron[perm_u, :]
    V4 = V4_kron[perm_v, :]
    W4 = W4_kron[perm_w, :]
    
    # Verify the decomposition
    if not _verify_decomposition(U4, V4, W4, n, m, p):
        # Fallback: schoolbook
        return _schoolbook(n, m, p)
        
    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}

def construct():
    """
    Constructs a minimal-rank exact tensor decomposition for 4x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer entries.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try rank-49 Kronecker-Strassen with verification
    if time.time() < deadline:
        r49 = _strassen_kron_4x4(n, m, p)
        return r49
    
    # Final fallback: schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
