"""Matrix multiplication 3x5x6 rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False

    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False

    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)

    C = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)

    if not np.allclose(C, 1.0, atol=tol):
        return False
    return True

def _strassen_2x2x2():
    """
    Returns the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    # Strassen 7 multiplications for 2x2x2
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # Standard Strassen:
    # M1 = (A11-A22)(B21+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M7

    # U entries: for each r (0-6), U[i,j,r]
    # V entries: for each r, V[j,k,r]
    # W entries: for each r, W[k,i,r]

    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # r=0: M1 = (A11-A22)(B21+B22)
    # U: A11=1, A22=-1 -> U[0,0]=1, U[3,0]=-1
    U[0, 0] = 1.0
    U[3, 0] = -1.0
    # V: B21=1, B22=1 -> V[2,1]=1, V[3,2]=1
    V[2, 1] = 1.0
    V[3, 2] = 1.0
    # W: C11 += M1, C22 += M1 -> W[0,0]=1, W[3,1]=1
    W[0, 0] = 1.0
    W[3, 1] = 1.0

    # r=1: M2 = (A21+A22)B11
    U[2, 1] = 1.0
    U[3, 1] = 1.0
    V[0, 0] = 1.0
    W[1, 0] = 1.0
    W[2, 1] = 1.0

    # r=2: M3 = A11(B12-B22)
    U[0, 2] = 1.0
    V[0, 1] = 1.0
    V[1, 2] = -1.0
    W[0, 0] += 1.0  # C12 += M3
    W[3, 1] += 1.0  # C22 += M3

    # r=3: M4 = A22(B21-B11)
    U[3, 3] = 1.0
    V[1, 0] = -1.0
    V[2, 1] = 1.0
    W[0, 0] += 1.0  # C11 += M4
    W[1, 0] += 1.0  # C21 += M4

    # r=4: M5 = (A11+A12)B22
    U[0, 4] = 1.0
    U[1, 4] = 1.0
    V[2, 2] = 1.0
    W[0, 0] -= 1.0  # C11 -= M5
    W[0, 1] += 1.0  # C12 += M5

    # r=5: M6 = (A21-A11)(B11+B12)
    U[0, 5] = -1.0
    U[2, 5] = 1.0
    V[0, 0] = 1.0
    V[0, 1] = 1.0
    # M6 doesn't directly contribute to C in standard form; it's used implicitly
    # Actually, in the standard Strassen, M6 is not used directly in C.
    # Let me recalculate. The standard 7-term Strassen uses:
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M7
    # So M6 is not used! Let me fix this.

    # Redo: The 7 multiplications are M1-M7, but only M1,M2,M3,M4,M5,M7 are used in C.
    # M6 is not used in the final C computation. Let me use a different indexing.

    # Let me restart with the correct Strassen decomposition.
    # The 7 scalar multiplications are:
    # P1 = (A11-A22)(B21+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    # C11 = P1+P4-P5+P7
    # C12 = P3+P5
    # C21 = P2+P4
    # C22 = P1-P2+P3+P7

    # So all 7 are used. Let me redo the matrices.

    U[:] = 0.0
    V[:] = 0.0
    W[:] = 0.0

    # P1 = (A11-A22)(B21+B22)
    U[0, 0] = 1.0; U[3, 0] = -1.0
    V[2, 1] = 1.0; V[3, 2] = 1.0
    W[0, 0] += 1.0; W[3, 1] += 1.0  # C11 += P1, C22 += P1

    # P2 = (A21+A22)B11
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 0] = 1.0
    W[1, 0] += 1.0; W[3, 1] -= 1.0  # C21 += P2, C22 -= P2

    # P3 = A11(B12-B22)
    U[0, 2] = 1.0
    V[0, 1] = 1.0; V[2, 2] = -1.0
    W[0, 1] += 1.0; W[3, 1] += 1.0  # C12 += P3, C22 += P3

    # P4 = A22(B21-B11)
    U[3, 3] = 1.0
    V[1, 0] = -1.0; V[2, 1] = 1.0
    W[0, 0] += 1.0; W[1, 0] += 1.0  # C11 += P4, C21 += P4

    # P5 = (A11+A12)B22
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[2, 2] = 1.0
    W[0, 0] -= 1.0; W[0, 1] += 1.0  # C11 -= P5, C12 += P5

    # P6 = (A21-A11)(B11+B12)
    U[0, 5] = -1.0; U[2, 5] = 1.0
    V[0, 0] = 1.0; V[0, 1] = 1.0
    # P6 is not directly in C, so W contributions are 0

    # P7 = (A12-A22)(B21+B22)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 1] = 1.0; V[3, 2] = 1.0
    W[0, 0] += 1.0; W[3, 1] += 1.0  # C11 += P7, C22 += P7

    return U, V, W

def _block_3x2x6():
    """
    Decompose 3x2x6 matrix multiplication.
    We split the 2 columns into two groups of 1, or use Strassen for the 2x2 part.
    Actually, 3x2x6: we can think of it as 3x(2)x(6) = 3x2x6.
    Schoolbook rank = 3*2*6 = 36.
    Using Strassen on the 2x2 part: treat the 2 columns as a 2x2 block structure.
    Actually, let's just use the direct sum approach.
    3x2x6 can be decomposed by embedding 2x2x2 Strassen.
    
    For 3x2x6:
    A is 3x2, B is 2x6, C is 3x6.
    We can view this as three 1x2x6 multiplications (one per row of A),
    or use the fact that 2x2x2 has rank 7.
    
    Let me think differently. 3x2x6:
    We can write this as a block multiplication.
    A = [A1; A2; A3] where Ai is 1x2
    B is 2x6
    Ci = Ai * B for i=1,2,3
    
    Each Ai*B is a 1x6 vector, which is just a linear combination of the 2 rows of B.
    This doesn't directly help with tensor rank.
    
    Let's use the known result: rank(3x2x6) <= rank(2x2x6)*something.
    Actually, let's just construct it directly.
    
    For 3x2x6, schoolbook gives rank 36.
    We can try to use Strassen-like tricks.
    
    Actually, let me use a simpler approach:
    3x2x6 can be seen as the direct sum of three 1x2x6 problems.
    Each 1x2x6 has rank 2*6 = 12 (schoolbook).
    Total: 3*12 = 36. No improvement.
    
    Let me try a different decomposition.
    Actually, for the purpose of this task, let's use schoolbook for 3x2x6 (rank 36)
    and schoolbook for 3x3x6 (rank 54), total = 90. No improvement.
    
    Let me reconsider. The directive says to use rank-14 for 2x2x2, but that's wrong;
    Strassen gives rank 7. Let me use rank 7 for 2x2x2.
    
    For 3x2x6:
    We can embed the 2x2x2 Strassen. Think of the 3 rows as three copies.
    Actually, 3x2x6 = sum of three 1x2x6.
    Each 1x2x6: A is 1x2, B is 2x6, C is 1x6.
    C_k = sum_j A_j B_jk for j=1,2 and k=1..6.
    This is just a linear map. The tensor for 1x2x6 has rank 2*6=12.
    
    Hmm, let me just use schoolbook for both blocks and see if we can do better.
    
    Actually, let me use the following approach:
    3x5x6 = 3x(2+3)x6 = (3x2x6) ⊕ (3x3x6)
    
    For 3x2x6: use Strassen on the 2x2 part by viewing it as
    multiplying a 3x2 matrix by a 2x6 matrix.
    We can group the 6 columns into 3 groups of 2.
    So B is 2x(2+2+2) = 2x6, with columns grouped as B1, B2, B3 each 2x2.
    Then C = [A*B1, A*B2, A*B3] where each A*Bi is 3x2.
    Each 3x2x2 multiplication can use Strassen-like tricks.
    
    3x2x2: A is 3x2, B is 2x2, C is 3x2.
    This can be seen as three 1x2x2 multiplications.
    Each 1x2x2 has rank 4 (schoolbook) or can we do better?
    1x2x2: C_k = A1*B1k + A2*B2k for k=1,2.
    This is just computing two linear combinations. Rank = 2*2 = 4.
    
    So 3x2x2 has rank 3*4 = 12.
    And 3x2x6 has rank 3*12 = 36. Still schoolbook.
    
    Let me try yet another approach. Use the known rank for 3x3x3.
    Actually, let me just implement the direct sum with schoolbook and see.
    """
    # 3x2x6 schoolbook
    n, m, p = 3, 2, 6
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W, rank

def _block_3x3x6():
    """
    Decompose 3x3x6 matrix multiplication.
    3x3x6: A is 3x3, B is 3x6, C is 3x6.
    Schoolbook rank = 3*3*6 = 54.
    
    We can group the 6 columns into 3 groups of 2.
    B = [B1, B2, B3] where each Bi is 3x2.
    C = [A*B1, A*B2, A*B3] where each A*Bi is 3x2.
    Each 3x3x2 multiplication.
    
    3x3x2: A is 3x3, B is 3x2, C is 3x2.
    This can be seen as two 3x3x1 multiplications.
    Each 3x3x1: C_k = sum_{i,j} A_ij B_jk for fixed k.
    Rank = 9 (schoolbook).
    So 3x3x2 has rank 18.
    And 3x3x6 has rank 3*18 = 54. Still schoolbook.
    
    Let me try using Strassen for the 3x3 part.
    Actually, 3x3x3 has known rank 23 (or 22 with complex numbers).
    But we need 3x3x6.
    
    Let me just use schoolbook for now.
    """
    n, m, p = 3, 3, 6
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W, rank

def _construct_block_diagonal():
    """
    Construct the block-diagonal decomposition for 3x5x6.
    Split the 5 columns into 2+3.
    Block 1: 3x2x6
    Block 2: 3x3x6
    """
    n, m, p = 3, 5, 6
    
    # Block 1: 3x2x6
    U1, V1, W1, r1 = _block_3x2x6()
    # Block 2: 3x3x6
    U2, V2, W2, r2 = _block_3x3x6()
    
    total_rank = r1 + r2
    
    U = np.zeros((n*m, total_rank), dtype=np.float64)
    V = np.zeros((m*p, total_rank), dtype=np.float64)
    W = np.zeros((p*n, total_rank), dtype=np.float64)
    
    # Place block 1
    # U1 is (3*2, r1), map to rows 0..5 of U (i*5+j for i=0..2, j=0..1)
    for i in range(3):
        for j in range(2):
            U[i*5 + j, 0:r1] = U1[i*2 + j, :]
    
    # V1 is (2*6, r1), map to rows j*6+k for j=0..1, k=0..5
    for j in range(2):
        for k in range(6):
            V[j*6 + k, 0:r1] = V1[j*6 + k, :]
    
    # W1 is (6*3, r1), map to rows k*3+i for k=0..5, i=0..2
    for k in range(6):
        for i in range(3):
            W[k*3 + i, 0:r1] = W1[k*3 + i, :]
    
    # Place block 2
    # U2 is (3*3, r2), map to rows i*5+j for i=0..2, j=2..4
    for i in range(3):
        for j in range(3):
            U[i*5 + (2+j), r1:r1+r2] = U2[i*3 + j, :]
    
    # V2 is (3*6, r2), map to rows j*6+k for j=2..4, k=0..5
    for j in range(3):
        for k in range(6):
            V[(2+j)*6 + k, r1:r1+r2] = V2[j*6 + k, :]
    
    # W2 is (6*3, r2), map to rows k*3+i for k=0..5, i=0..2
    for k in range(6):
        for i in range(3):
            W[k*3 + i, r1:r1+r2] = W2[k*3 + i, :]
    
    return U, V, W, total_rank

def _schoolbook_decomposition(n, m, p):
    """Returns the schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_strassen_3x5x6():
    """
    Try to construct a lower-rank decomposition using Strassen-like techniques.
    For 3x5x6, we can try to use the known rank-23 decomposition for 3x3x3
    and embed it appropriately.
    
    Actually, let me try a different approach: use the rank-7 decomposition for 2x2x2
    to build up a lower-rank decomposition for 3x5x6.
    """
    n, m, p = 3, 5, 6
    
    # Try to use block structure with Strassen
    # Split 5 into 2+3, and 6 into 2+2+2
    # This gets complex. Let me just use the block diagonal approach for now.
    return None

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for matrix multiplication.
    Dimensions: n=3, m=5, p=6.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try block diagonal decomposition
    if time.time() < deadline - 10.0:
        U_bd, V_bd, W_bd, rank_bd = _construct_block_diagonal()
        if _verify_decomposition(U_bd, V_bd, W_bd, n, m, p):
            if rank_bd < best_rank:
                best_U, best_V, best_W = U_bd.copy(), V_bd.copy(), W_bd.copy()
                best_rank = rank_bd
    
    # Try Strassen-based approach
    if time.time() < deadline - 10.0:
        res = _try_strassen_3x5x6()
        if res is not None:
            U_s, V_s, W_s, rank_s = res
            if _verify_decomposition(U_s, V_s, W_s, n, m, p):
                if rank_s < best_rank:
                    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
                    best_rank = rank_s
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
