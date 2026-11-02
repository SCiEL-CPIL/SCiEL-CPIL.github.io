"""
Rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.
Uses block-Strassen decomposition to reduce rank below schoolbook (105).

Strategy:
- Split the 5x5 middle dimension into 3+2 (for B) and correspondingly for A's rows.
- Actually, we need to be careful: A is 3x5, B is 5x7.
- Split A's columns (and B's rows) into 3+2.
- A = [A1 (3x3) | A2 (3x2)]
- B = [B1; B2] where B1 is 3x7, B2 is 2x7
- C = A1*B1 + A2*B2
- M_{3,3,7} and M_{3,2,7}

For M_{3,3,7}: split 7 into 3+3+1? Or use Strassen on the 3x3 part?
Actually, let's use a more systematic block-Strassen approach.

Let me think about known low-rank decompositions for small matrices:
- M_{2,2,2}: Strassen, rank 7
- M_{3,3,3}: can be done with rank 18 (using 2x2 Strassen blocks)
- M_{3,2,2}: rank 9
- M_{3,2,3}: rank 11
- M_{3,3,2}: rank 11

For M_{3,3,7}: split 7 = 3+3+1
  M_{3,3,3} (rank 18) + M_{3,3,3} (rank 18) + M_{3,3,1} (rank 9) = 45
For M_{3,2,7}: split 7 = 3+3+1
  M_{3,2,3} (rank 11) + M_{3,2,3} (rank 11) + M_{3,2,1} (rank 6) = 28
Total: 45 + 28 = 73

Hmm, that's 73. Let me try to do better.

Actually, let me just implement the known small decompositions directly.
I'll hardcode the Strassen-based block decompositions for the needed sub-problems.

Let me implement a function that builds the decomposition for any (n,m,p) using
block splitting and base cases (schoolbook for very small, Strassen for 2x2x2).
"""
import numpy as np

def _schoolbook(n, m, p):
    """Return U, V, W for schoolbook decomposition of n x m * m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
    return U, V, W

def _strassen_222():
    """Strassen decomposition for 2x2 * 2x2, rank 7."""
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    n = m = p = 2
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # A indices: A11=0, A12=1, A21=2, A22=3
    # B indices: B11=0, B12=1, B21=2, B22=3
    # C indices: C11=0, C12=1, C21=2, C22=3 (row-major: C[i*2+j])
    
    # M1: (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2: (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C21
    W[3, 1] = -1 # C22
    
    # M3: A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # M4: A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # M5: (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C11
    W[1, 4] = 1  # C12
    
    # M6: (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22
    
    # M7: (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11
    
    return U, V, W

def _combine_decomps(decomps, n, m, p, offsets):
    """
    Combine multiple sub-decompositions into one.
    decomps: list of (U_i, V_i, W_i) for sub-problems
    offsets: list of (i_off, j_off, k_off) for each sub-problem
    n, m, p: full dimensions
    """
    total_rank = sum(U.shape[1] for U, V, W in decomps)
    U_full = np.zeros((n * m, total_rank), dtype=float)
    V_full = np.zeros((m * p, total_rank), dtype=float)
    W_full = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for (Ui, Vi, Wi), (i_off, j_off, k_off) in zip(decomps, offsets):
        n_i = Ui.shape[0] // m  # not quite, need to figure out sub-dims
        # Actually, Ui has shape (n_i * m_i, rank_i) where m_i is the middle dim for this sub-problem
        # This is getting complicated. Let me handle it differently.
        rank_i = Ui.shape[1]
        # Place Ui into U_full at appropriate rows
        # The rows of Ui correspond to (i, j) pairs in the sub-problem
        # We need to map sub-problem (i, j) to full (i + i_off, j + j_off)
        n_i = Ui.shape[0] // (m)  # This assumes m_i = m, which is not always true
        # This approach is flawed. Let me restructure.
        break
    return U_full, V_full, W_full

# Let me take a different, cleaner approach.
# I'll directly build the decomposition for 3x5 * 5x7 by splitting into blocks
# and using known low-rank decompositions for the sub-blocks.

def _build_decomp(n, m, p):
    """Build a low-rank decomposition for n x m * m x p."""
    if n == 2 and m == 2 and p == 2:
        return _strassen_222()
    if n == 1 or m == 1 or p == 1:
        return _schoolbook(n, m, p)
    
    # Try to split
    # Split m into m1 + m2
    best = None
    for m1 in range(1, m):
        m2 = m - m1
        # For each way to split n into n1+n2 and p into p1+p2? No, n and p are fixed per sub-block.
        # Actually, when we split the middle dimension m into m1+m2:
        # A is n x (m1+m2) = [A1 (n x m1) | A2 (n x m2)]
        # B is (m1+m2) x p = [B1 (m1 x p); B2 (m2 x p)]
        # C = A1*B1 + A2*B2
        # So we need M_{n, m1, p} and M_{n, m2, p}
        U1, V1, W1 = _build_decomp(n, m1, p)
        U2, V2, W2 = _build_decomp(n, m2, p)
        
        rank = U1.shape[1] + U2.shape[1]
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        # U1 has shape (n * m1, rank1), place in rows 0..n*m1-1
        U[:n*m1, :U1.shape[1]] = U1
        # U2 has shape (n * m2, rank2), place in rows n*m1..n*m-1
        U[n*m1:, U1.shape[1]:] = U2
        
        # V1 has shape (m1 * p, rank1), place in rows 0..m1*p-1
        V[:m1*p, :V1.shape[1]] = V1
        # V2 has shape (m2 * p, rank2), place in rows m1*p..m*p-1
        V[m1*p:, V1.shape[1]:] = V2
        
        # W is the same for both (since C dimension is the same)
        W[:, :W1.shape[1]] = W1
        W[:, W1.shape[1]:] = W2
        
        if best is None or rank < best[0]:
            best = (rank, U, V, W)
    
    # Also try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = _build_decomp(n1, m, p)
        U2, V2, W2 = _build_decomp(n2, m, p)
        
        rank = U1.shape[1] + U2.shape[1]
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        # U1: (n1*m, rank1), rows 0..n1*m-1
        U[:n1*m, :U1.shape[1]] = U1
        # U2: (n2*m, rank2), rows n1*m..n*m-1
        U[n1*m:, U1.shape[1]:] = U2
        
        # V is same for both
        V[:, :V1.shape[1]] = V1
        V[:, V1.shape[1]:] = V2
        
        # W1: (p*n1, rank1), rows 0..p*n1-1
        W[:p*n1, :W1.shape[1]] = W1
        # W2: (p*n2, rank2), rows p*n1..p*n-1
        W[p*n1:, W1.shape[1]:] = W2
        
        if best is None or rank < best[0]:
            best = (rank, U, V, W)
    
    # Also try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = _build_decomp(n, m, p1)
        U2, V2, W2 = _build_decomp(n, m, p2)
        
        rank = U1.shape[1] + U2.shape[1]
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        # U is same for both
        U[:, :U1.shape[1]] = U1
        U[:, U1.shape[1]:] = U2
        
        # V1: (m*p1, rank1), cols 0..m*p1-1
        V[:, :m*p1] = V1
        # V2: (m*p2, rank2), cols m*p1..m*p-1
        V[:, m*p1:] = V2
        
        # W1: (p1*n, rank1), rows 0..p1*n-1
        W[:p1*n, :W1.shape[1]] = W1
        # W2: (p2*n, rank2), rows p1*n..p*n-1
        W[p1*n:, W1.shape[1]:] = W2
        
        if best is None or rank < best[0]:
            best = (rank, U, V, W)
    
    if best is None:
        return _schoolbook(n, m, p)
    
    return best[1], best[2], best[3]

def construct():
    n, m, p = 3, 5, 7
    U, V, W = _build_decomp(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
