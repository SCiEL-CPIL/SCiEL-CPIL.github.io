"""Matrix multiplication 3x4 by 4x7: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly reproduces the multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rng = np.random.default_rng(0)
    for _ in range(5):
        A = rng.integers(-5, 6, size=(n, m)).astype(float)
        B = rng.integers(-5, 6, size=(m, p)).astype(float)
        C = A @ B
        Crec = np.zeros((n, p))
        for c in range(U.shape[1]):
            L = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    L[i] += U[i*m+j, c] * A[i, j]
            R = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    R[k] += V[j*p+k, c] * B[j, k]
            for k in range(p):
                for i in range(n):
                    Crec[i, k] += L[i] * R[k] * W[k*n+i, c]
        if not np.allclose(Crec, C, atol=1e-9):
            return False
    return True

def _strassen_block(i_off, j_off, k_off, U, V, W, col, n, m, p):
    """
    Add Strassen decomposition for a 2x2x2 block at given offsets.
    i indices: i_off, i_off+1
    j indices: j_off, j_off+1
    k indices: k_off, k_off+1
    
    Returns the next available column index.
    """
    i0, i1 = i_off, i_off + 1
    j0, j1 = j_off, j_off + 1
    k0, k1 = k_off, k_off + 1
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    U[i0*m + j0, col] += 1
    U[i1*m + j1, col] += 1
    V[j0*p + k0, col] += 1
    V[j1*p + k1, col] += 1
    W[k0*n + i0, col] += 1
    W[k1*n + i1, col] += 1
    col += 1
    
    # Term 2: M2 = (A10+A11)B00
    # C10 += M2, C11 -= M2  (Wait, standard Strassen: C10 += M2, C11 += M2? No.)
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's re-map carefully to our indices:
    # A indices: (i, j). Block i in {i0, i1}, j in {j0, j1}.
    # A00 -> A[i0, j0], A01 -> A[i0, j1], A10 -> A[i1, j0], A11 -> A[i1, j1]
    # B indices: (j, k). Block j in {j0, j1}, k in {k0, k1}.
    # B00 -> B[j0, k0], B01 -> B[j0, k1], B10 -> B[j1, k0], B11 -> B[j1, k1]
    # C indices: (i, k). Block i in {i0, i1}, k in {k0, k1}.
    # C00 -> C[i0, k0], C01 -> C[i0, k1], C10 -> C[i1, k0], C11 -> C[i1, k1]
    
    # M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    # (Already added above)
    
    # M2 = (A10+A11)B00
    # C10 += M2, C11 -= M2  (Wait, C22 = M1 - M2 + ... so C11 (which is C22 in 1-based) gets -M2)
    # Let's stick to 0-based local indices for C: C00, C01, C10, C11 correspond to (i0,k0), (i0,k1), (i1,k0), (i1,k1).
    # Standard formulas:
    # C00 (local 0,0) = M1 - M4 + M5 + M7
    # C01 (local 0,1) = M3 + M5
    # C10 (local 1,0) = M2 + M4
    # C11 (local 1,1) = M1 - M2 + M3 + M6
    
    # My previous Term 1 was M1.
    # Term 2: M2 = (A10+A11)B00
    U[i1*m + j0, col] += 1
    U[i1*m + j1, col] += 1
    V[j0*p + k0, col] += 1
    W[k0*n + i1, col] += 1   # C10 += M2
    W[k1*n + i1, col] += -1  # C11 -= M2
    col += 1
    
    # Term 3: M3 = A00(B10-B11)
    U[i0*m + j0, col] += 1
    V[j1*p + k0, col] += 1
    V[j1*p + k1, col] += -1
    W[k1*n + i0, col] += 1   # C01 += M3
    W[k1*n + i1, col] += 1   # C11 += M3
    col += 1
    
    # Term 4: M4 = A11(B01-B00)
    U[i1*m + j1, col] += 1
    V[j0*p + k1, col] += 1
    V[j0*p + k0, col] += -1
    W[k0*n + i0, col] += -1  # C00 -= M4
    W[k0*n + i1, col] += 1   # C10 += M4
    col += 1
    
    # Term 5: M5 = (A00+A10)B11
    U[i0*m + j0, col] += 1
    U[i1*m + j0, col] += 1
    V[j1*p + k1, col] += 1
    W[k0*n + i0, col] += 1   # C00 += M5
    W[k1*n + i0, col] += 1   # C01 += M5
    col += 1
    
    # Term 6: M6 = (A01-A00)(B00+B10)
    U[i0*m + j1, col] += 1
    U[i0*m + j0, col] += -1
    V[j0*p + k0, col] += 1
    V[j1*p + k0, col] += 1
    W[k1*n + i1, col] += 1   # C11 += M6
    col += 1
    
    # Term 7: M7 = (A10-A11)(B01+B11)
    U[i1*m + j0, col] += 1
    U[i1*m + j1, col] += -1
    V[j0*p + k1, col] += 1
    V[j1*p + k1, col] += 1
    W[k0*n + i0, col] += 1   # C00 += M7
    col += 1
    
    return col

def construct():
    """Construct a rank-minimal exact decomposition for 3x4 * 4x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Fallback to schoolbook if we can't find a better one
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to construct a decomposition using Strassen for 2x2x2 blocks.
    # We can fit 3 non-overlapping 2x2x2 blocks in 3x4x7.
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
    # Note: j in {2,3} is valid since m=4.
    # i in {0,1} is valid since n=3.
    # k in {0,1} and {2,3} are valid since p=7.
    
    # Rank calculation:
    # Total triples = 3*4*7 = 84.
    # Each block covers 2*2*2 = 8 triples.
    # 3 blocks cover 24 triples.
    # Remaining triples: 84 - 24 = 60.
    # Rank = 3 * 7 (Strassen) + 60 (Schoolbook) = 21 + 60 = 81.
    
    rank = 3 * 7 + (n * m * p - 3 * 8)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Block 1
    col = _strassen_block(0, 0, 0, U, V, W, col, n, m, p)
    # Block 2
    col = _strassen_block(0, 2, 0, U, V, W, col, n, m, p)
    # Block 3
    col = _strassen_block(0, 0, 2, U, V, W, col, n, m, p)
    
    # Mark covered triples
    covered = set()
    for i_off in [0]:
        for j_off in [0, 2]:
            for k_off in [0]:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered.add((i_off + di, j_off + dj, k_off + dk))
    for i_off in [0]:
        for j_off in [0]:
            for k_off in [2]:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered.add((i_off + di, j_off + dj, k_off + dk))
        
    # Fill remaining with schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify col matches rank
    if col != rank:
        return schoolbook()
        
    # Verify correctness
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
