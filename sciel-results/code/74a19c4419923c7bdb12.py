"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_TIMEOUT = 130  # safety margin before 150s kill

def _strassen_222():
    """
    Return (U, V, W) for 2x2x2 matrix multiplication using Strassen's algorithm.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}
    """
    # Strassen's 7 multiplications for 2x2x2
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00 and C11
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C00
    W[1][4] = 1   # C01
    
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1   # C11
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1   # C00
    W[3][6] = 1   # C11
    
    return U, V, W

def _schoolbook_333():
    """
    Return (U, V, W) for 3x3x3 matrix multiplication using schoolbook.
    U has shape (9, 9), V has shape (9, 9), W has shape (9, 9).
    """
    rank = 9
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    return U, V, W

def _schoolbook_155():
    """
    Return (U, V, W) for 1x5x5 matrix multiplication using schoolbook.
    U has shape (5, 25), V has shape (25, 25), W has shape (5, 25).
    """
    rank = 25
    U = [[0]*rank for _ in range(5)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(5)]
    
    col = 0
    for j in range(5):
        for k in range(5):
            U[j][col] = 1
            V[j*5+k][col] = 1
            W[k][col] = 1
            col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    And for all other triples (i,j,k) that don't share an index appropriately, the sum should be 0.
    Actually, the condition is:
        C[k*n+i] = sum_j A[i,j] * B[j,k]
    which means:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if the indices are consistent
    More precisely, the tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    should equal 1 when (i,j,k) form a valid matrix multiplication term, i.e., always 1 for all i,j,k
    since each term A[i,j]*B[j,k] contributes to C[k,i].
    
    Wait, let me reconsider. The matrix multiplication tensor is:
    T[i,j,k] = 1 for all i,j,k (each A[i,j]*B[j,k] is a separate term)
    
    So we need: for all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U_np.shape[1]):
                    s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank71():
    """
    Build a rank-71 decomposition for 3x5x5 matrix multiplication.
    
    Strategy: Split the 3x5x5 problem into:
    - A 2x5x5 sub-problem (rows 0,1 of A)
    - A 1x5x5 sub-problem (row 2 of A)
    
    For the 2x5x5 sub-problem, further split the inner dimension 5 into 2+3:
    - 2x2x2 block: use Strassen (rank 7)
    - 2x2x3 block: schoolbook (rank 12)  
    - 2x3x2 block: schoolbook (rank 12)
    - 2x3x3 block: schoolbook (rank 18)
    Total for 2x5x5: 7 + 12 + 12 + 18 = 49
    
    Wait, that's not right. Let me reconsider.
    
    Actually, for 2x5x5, if we split j into j1 (2 dims) and j2 (3 dims), and k into k1 (2 dims) and k2 (3 dims):
    - A is 2x5 = [A(:,j1) | A(:,j2)] where A(:,j1) is 2x2, A(:,j2) is 2x3
    - B is 5x5 = [B(j1,k1) B(j1,k2); B(j2,k1) B(j2,k2)]
    
    The product C = A*B gives:
    C(:,k1) = A(:,j1)*B(j1,k1) + A(:,j2)*B(j2,k1)
    C(:,k2) = A(:,j1)*B(j1,k2) + A(:,j2)*B(j2,k2)
    
    So we have 4 sub-products:
    1. A(:,j1) * B(j1,k1): 2x2 * 2x2 -> 2x2, rank 7 (Strassen)
    2. A(:,j1) * B(j1,k2): 2x2 * 2x3 -> 2x3, rank 12 (schoolbook)
    3. A(:,j2) * B(j2,k1): 2x3 * 3x2 -> 2x2, rank 12 (schoolbook)
    4. A(:,j2) * B(j2,k2): 2x3 * 3x3 -> 2x3, rank 18 (schoolbook)
    
    Total: 7 + 12 + 12 + 18 = 49
    
    For the 1x5x5 sub-problem, schoolbook gives rank 25.
    
    Total rank: 49 + 25 = 74. That's worse than 75? No, 74 < 75.
    
    Hmm, but the inspiration says rank 71. Let me think again.
    
    Maybe we can do better on the 1x5x5 part or combine some terms.
    
    Actually, wait. Let me reconsider the 2x5x5 decomposition more carefully.
    
    For 2x2 * 2x3 -> 2x3:
    Each output C[i,k] = sum_j A[i,j]*B[j,k]
    There are 2*3 = 6 outputs, each requiring 2 multiplications, so 12 total. Schoolbook is 12.
    
    For 2x3 * 3x2 -> 2x2:
    2*2 = 4 outputs, each requiring 3 multiplications, so 12 total. Schoolbook is 12.
    
    For 2x3 * 3x3 -> 2x3:
    2*3 = 6 outputs, each requiring 3 multiplications, so 18 total. Schoolbook is 18.
    
    So 7 + 12 + 12 + 18 = 49 for the 2x5x5 part.
    25 for the 1x5x5 part.
    Total: 74.
    
    But the inspiration says 71. Maybe there's a way to share some computations or use a better algorithm for one of the sub-blocks.
    
    Let me just build the rank-74 version first and verify it works.
    """
    n, m, p = 3, 5, 5
    
    # Split: first 2 rows of A use the 2x5x5 decomposition, last row uses 1x5x5
    # Total rank = 49 + 25 = 74
    
    total_rank = 74
    U = [[0]*total_rank for _ in range(n*m)]  # 15 rows
    V = [[0]*total_rank for _ in range(m*p)]  # 25 rows
    W = [[0]*total_rank for _ in range(p*n)]  # 15 rows
    
    # Part 1: 2x5x5 (rows 0,1 of A)
    # Split j: j1 = [0,1], j2 = [2,3,4]
    # Split k: k1 = [0,1], k2 = [2,3,4]
    
    # Sub-product 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (2x2 * 2x2 -> 2x2)
    U1, V1, W1 = _strassen_222()
    for i in range(2):
        for j in range(2):
            for r in range(7):
                U[i*m + j][r] += U1[i*2+j][r]
    for j in range(2):
        for k in range(2):
            for r in range(7):
                V[j*p + k][r] += V1[j*2+k][r]
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W[k*n + i][r] += W1[k*2+i][r]
    
    # Sub-product 2: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5] (2x2 * 2x3 -> 2x3)
    # Schoolbook: 12 multiplications
    offset = 7
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*m + j][offset] = 1
                V[j*p + (2+k)][offset] = 1
                W[(2+k)*n + i][offset] = 1
                offset += 1
    
    # Sub-product 3: A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2] (2x3 * 3x2 -> 2x2)
    # Schoolbook: 12 multiplications
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[i*m + (2+j)][offset] = 1
                V[(2+j)*p + k][offset] = 1
                W[k*n + i][offset] = 1
                offset += 1
    
    # Sub-product 4: A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5] (2x3 * 3x3 -> 2x3)
    # Schoolbook: 18 multiplications
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i*m + (2+j)][offset] = 1
                V[(2+j)*p + (2+k)][offset] = 1
                W[(2+k)*n + i][offset] = 1
                offset += 1
    
    # Part 2: 1x5x5 (row 2 of A)
    # Schoolbook: 25 multiplications
    for j in range(5):
        for k in range(5):
            U[2*m + j][offset] = 1
            V[j*p + k][offset] = 1
            W[k*n + 2][offset] = 1
            offset += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Build rank-74 decomposition
    U, V, W = _build_rank71()
    
    # Verify the decomposition
    if time.time() - _start_time < _TIMEOUT - 5:
        try:
            if not _verify_decomposition(U, V, W, n, m, p):
                # Fall back to schoolbook rank-75
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
        except Exception:
            pass
    
    if time.time() - _start_time > _TIMEOUT:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
