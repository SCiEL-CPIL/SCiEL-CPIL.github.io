import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is m * p
    rank = m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Indexing conventions:
    # U: (n*m, rank). Index i in [0, n*m) corresponds to A[i // m][i % m]
    # V: (m*p, rank). Index j in [0, m*p) corresponds to B[j // p][j % p]
    # W: (p*n, rank). Index k in [0, p*n) corresponds to C[k % n][k // n] ? 
    # Let's check the task description: "W[k*n+i,r]"
    # This implies W index is k*n + i where k is row of C? No, C is n x p.
    # Usually C[i,j] = sum_k A[i,k] B[k,j].
    # Task says W[k*n+i, r].
    # If C is n x p, indices are (i, j) with i in 0..n-1, j in 0..p-1.
    # The description "W[k*n+i]" suggests the first index of W is `k` and second is `i`?
    # Or is it a flat index? "W[k*n+i,r]" looks like accessing a 2D array W where row is `k*n+i`? No, that's flat.
    # Let's assume standard tensor decomposition:
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Here i is row of A (0..n-1), j is col of A/row of B (0..m-1), k is col of B (0..p-1).
    # Wait, matrix multiplication C = A B.
    # A is n x m. B is m x p. C is n x p.
    # T[i, k, j] ? No.
    # Standard: C[i, j] = sum_{k=0}^{m-1} A[i, k] B[k, j].
    # Tensor T has indices (i, j, k) corresponding to A[i,k], B[k,j], C[i,j].
    # The factors are:
    # U: shape (n*m, rank). Indices (i, k).
    # V: shape (m*p, rank). Indices (k, j).
    # W: shape (p*n, rank). Indices (j, i)?
    # Let's re-read carefully: "output factor follows the official trace convention W[k*n+i,r]"
    # This phrasing is tricky. "W[k*n+i,r]" usually implies W is a matrix with rows indexed by `k*n+i`.
    # If W corresponds to C, and C is n x p, then the indices of C are (i, j) where i is row (0..n-1) and j is col (0..p-1).
    # If the index is `k*n + i`, maybe k is the column index (0..p-1) and i is the row index (0..n-1)?
    # Then `k*n + i` ranges from 0 to p*n - 1. This matches the shape (p*n, rank).
    # So W index `idx` corresponds to C[i, j] where `idx = j*n + i`.
    # Let's verify U and V shapes.
    # U shape (n*m, rank). Indices (i, k) from A. Flat index `i*m + k`.
    # V shape (m*p, rank). Indices (k, j) from B. Flat index `k*p + j`.
    
    # Let's construct the schoolbook decomposition.
    # For each k in 0..m-1 and j in 0..p-1, we have a term A[i, k] * B[k, j] contributing to C[i, j].
    # We can assign one rank dimension for each pair (k, j).
    # Total rank = m * p.
    
    for k in range(m):
        for j in range(p):
            r = k * p + j
            
            # U part: A[i, k] is non-zero only for this k.
            # U index for A[i, k] is i * m + k.
            for i in range(n):
                U[i * m + k, r] = 1.0
            
            # V part: B[k, j] is non-zero only for this k, j.
            # V index for B[k, j] is k * p + j.
            V[k * p + j, r] = 1.0
            
            # W part: C[i, j] receives contribution.
            # W index for C[i, j] is j * n + i (based on "W[k*n+i]" where k->j, i->i).
            for i in range(n):
                W[j * n + i, r] = 1.0
                
    return {"U": U, "V": V, "W": W}
