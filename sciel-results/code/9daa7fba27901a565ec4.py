"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build expected tensor Texp
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    
    # Compute actual tensor T = sum_r U[:,r] V[:,r] W[:,r]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    err = np.max(np.abs(T - Texp))
    return err < tol

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using tensor contraction."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build expected tensor as a flat vector
    Texp_flat = np.zeros(n*m*m*p*p*n)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp_flat[a * (m*p) * (p*n) + b * (p*n) + c] = 1.0
    
    # Compute T flat
    T_flat = np.zeros(n*m*m*p*p*n)
    for r in range(rank):
        T_flat += np.outer(np.outer(U[:,r], V[:,r]), W[:,r]).flatten()
    
    err = np.max(np.abs(T_flat - Texp_flat))
    return err < tol

def _schoolbook(n, m, p):
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Rank 7 instead of 8.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21 + A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A11(B21 - B22)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A22(B12 - B11)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A11 + A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _block_decomposition(n, m, p):
    """
    Use block decomposition to reduce rank.
    
    For 2x4 x 4x7:
    Partition the middle dimension m=4 into two parts of size 2.
    A = [A1 A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x7
    C = A1*B1 + A2*B2
    
    For each 2x2 * 2x7 block, we can further partition the 7 columns into 4+3.
    B1 = [B11 B12] where B11 is 2x4, B12 is 2x3
    B2 = [B21 B22] where B21 is 2x4, B22 is 2x3
    
    C1 = A1*B11 + A2*B21  (2x4)
    C2 = A1*B12 + A2*B22  (2x3)
    
    For 2x2 * 2x2 blocks, use Strassen (rank 7)
    For 2x2 * 2x3 blocks, use schoolbook (rank 6)
    
    Total rank = 2*7 + 2*6 = 26
    """
    n, m, p = 2, 4, 7
    
    # Build the decomposition
    rank = 2*7 + 2*6  # 26
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Get Strassen 2x2x2
    strassen_222 = _strassen_2x2x2()
    
    # Block 1: A1 * B11 (2x2 * 2x2 -> 2x2)
    # A1: rows 0-1 of A, cols 0-1 -> indices 0,1,2,3 in A (row-major: i*m+j)
    # B11: rows 0-1 of B, cols 0-3 -> indices 0..7 in B (j*p+k)
    # C11: rows 0-1 of C, cols 0-1 -> indices 0..3 in C (k*n+i)
    for r in range(7):
        for i in range(4):
            U[i, col] += strassen_222['U'][i][r]
        for i in range(8):
            V[i, col] += strassen_222['V'][i // 2 * 2 + i % 2][r]  # This mapping is wrong
        col += 1
    
    # This indexing is getting complex. Let me use a cleaner approach.
    # Let me rebuild with explicit index tracking.
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper: apply a 2x2x2 Strassen decomposition to specific index ranges
    def apply_strassen_222(a_indices, b_indices, c_indices):
        """Apply Strassen 2x2x2 to given index sets."""
        nonlocal col
        s = _strassen_2x2x2()
        for r in range(7):
            for i in range(4):
                U[a_indices[i], col] += s['U'][i][r]
            for i in range(4):
                V[b_indices[i], col] += s['V'][i][r]
            for i in range(4):
                W[c_indices[i], col] += s['W'][i][r]
            col += 1
    
    # Helper: apply schoolbook 2x2x2
    def apply_schoolbook_222(a_indices, b_indices, c_indices):
        """Apply schoolbook 2x2x2 to given index sets."""
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[a_indices[i*2+j], col] = 1
                    V[b_indices[j*2+k], col] = 1
                    W[c_indices[k*2+i], col] = 1
                    col += 1
    
    # Helper: apply schoolbook 2x2x3 (A is 2x2, B is 2x3, C is 2x3)
    def apply_schoolbook_223(a_indices, b_indices, c_indices):
        """Apply schoolbook 2x2x3."""
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[a_indices[i*2+j], col] = 1
                    V[b_indices[j*3+k], col] = 1
                    W[c_indices[k*2+i], col] = 1
                    col += 1
    
    # Index mappings:
    # A is 2x4: A[i,j] at index i*4+j
    #   A1 (cols 0-1): [0, 1, 4, 5] for (i,j) in [(0,0),(0,1),(1,0),(1,1)]
    #   A2 (cols 2-3): [2, 3, 6, 7]
    
    # B is 4x7: B[j,k] at index j*7+k
    #   B11 (rows 0-1, cols 0-3): [0,1,2,3, 7,8,9,10]
    #   B12 (rows 0-1, cols 4-6): [4,5,6, 11,12,13]
    #   B21 (rows 2-3, cols 0-3): [14,15,16,17, 21,22,23,24]
    #   B22 (rows 2-3, cols 4-6): [18,19,20, 25,26,27]
    
    # C is 2x7: C[i,k] at index k*2+i
    #   C11 (cols 0-1): [0,1, 2,3]
    #   C12 (cols 2-3): [4,5, 6,7]
    #   C21 (cols 4-6): [8,9, 10,11, 12,13]
    
    A1_indices = [0, 1, 4, 5]  # A[i,j] for i in 0-1, j in 0-1
    A2_indices = [2, 3, 6, 7]  # A[i,j] for i in 0-1, j in 2-3
    
    B11_indices = [0, 1, 2, 3, 7, 8, 9, 10]  # B[j,k] for j in 0-1, k in 0-3
    B12_indices = [4, 5, 6, 11, 12, 13]      # B[j,k] for j in 0-1, k in 4-6
    B21_indices = [14, 15, 16, 17, 21, 22, 23, 24]  # B[j,k] for j in 2-3, k in 0-3
    B22_indices = [18, 19, 20, 25, 26, 27]    # B[j,k] for j in 2-3, k in 4-6
    
    C11_indices = [0, 1, 2, 3]  # C[i,k] for i in 0-1, k in 0-1
    C12_indices = [4, 5, 6, 7]  # C[i,k] for i in 0-1, k in 2-3
    C21_indices = [8, 9, 10, 11, 12, 13]  # C[i,k] for i in 0-1, k in 4-6
    
    # But Strassen needs 2x2 blocks with specific index orderings.
    # For 2x2x2 Strassen:
    #   A indices: A11, A12, A21, A22
    #   B indices: B11, B12, B21, B22
    #   C indices: C11, C12, C21, C22
    
    # For A1 (2x2): A11=A[0,0]=0, A12=A[0,1]=1, A21=A[1,0]=4, A22=A[1,1]=5
    A1_strassen = [0, 1, 4, 5]
    # For A2 (2x2): A11=A[0,2]=2, A12=A[0,3]=3, A21=A[1,2]=6, A22=A[1,3]=7
    A2_strassen = [2, 3, 6, 7]
    
    # For B11 (2x2, rows 0-1, cols 0-1): B11=B[0,0]=0, B12=B[0,1]=1, B21=B[1,0]=7, B22=B[1,1]=8
    B11_strassen = [0, 1, 7, 8]
    # For B11_right (2x2, rows 0-1, cols 2-3): B11=B[0,2]=2, B12=B[0,3]=3, B21=B[1,2]=9, B22=B[1,3]=10
    B11_right_strassen = [2, 3, 9, 10]
    
    # For B21 (2x2, rows 2-3, cols 0-1): B11=B[2,0]=14, B12=B[2,1]=15, B21=B[3,0]=21, B22=B[3,1]=22
    B21_strassen = [14, 15, 21, 22]
    # For B21_right (2x2, rows 2-3, cols 2-3): B11=B[2,2]=16, B12=B[2,3]=17, B21=B[3,2]=23, B22=B[3,3]=24
    B21_right_strassen = [16, 17, 23, 24]
    
    # For C11 (2x2, rows 0-1, cols 0-1): C11=C[0,0]=0, C12=C[0,1]=1, C21=C[1,0]=2, C22=C[1,1]=3
    C11_strassen = [0, 1, 2, 3]
    # For C12 (2x2, rows 0-1, cols 2-3): C11=C[0,2]=4, C12=C[0,3]=5, C21=C[1,2]=6, C22=C[1,3]=7
    C12_strassen = [4, 5, 6, 7]
    
    # Now apply:
    # C11 = A1*B11_left + A2*B21_left  (2x2 = 2x2*2x2 + 2x2*2x2)
    # C12 = A1*B11_right + A2*B21_right (2x2 = 2x2*2x2 + 2x2*2x2)
    # C21 = A1*B12 + A2*B22 (2x3 = 2x2*2x3 + 2x2*2x3)
    
    # Block 1: A1 * B11_left -> C11
    apply_strassen_222(A1_strassen, B11_strassen, C11_strassen)
    
    # Block 2: A2 * B21_left -> C11
    apply_strassen_222(A2_strassen, B21_strassen, C11_strassen)
    
    # Block 3: A1 * B11_right -> C12
    apply_strassen_222(A1_strassen, B11_right_strassen, C12_strassen)
    
    # Block 4: A2 * B21_right -> C12
    apply_strassen_222(A2_strassen, B21_right_strassen, C12_strassen)
    
    # Block 5: A1 * B12 -> C21 (2x2 * 2x3 -> 2x3)
    apply_schoolbook_223(A1_strassen, B12_indices, C21_indices)
    
    # Block 6: A2 * B22 -> C21 (2x2 * 2x3 -> 2x3)
    apply_schoolbook_223(A2_strassen, B22_indices, C21_indices)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_optimized_decomposition(n, m, p):
    """
    Try to find a better decomposition using various strategies.
    """
    n, m, p = 2, 4, 7
    
    # Start with block decomposition
    block = _block_decomposition(n, m, p)
    
    # Verify it
    if _verify(block['U'], block['V'], block['W'], n, m, p):
        return block
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Try block decomposition
    if time.time() < deadline - 20:
        block_result = _try_optimized_decomposition(n, m, p)
        if block_result['U'] is not None:
            # Verify the result
            if _verify(block_result['U'], block_result['V'], block_result['W'], n, m, p):
                result = block_result
    
    return result

# EVOLVE-BLOCK-END
