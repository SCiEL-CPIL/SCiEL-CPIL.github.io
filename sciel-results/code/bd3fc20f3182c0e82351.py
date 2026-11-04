"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Compute decomposition
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    for i in range(n*m):
        for j in range(m*p):
            for k in range(p*n):
                val = 0
                for c in range(rank):
                    val += U_arr[i, c] * V_arr[j, c] * W_arr[k, c]
                if T[i, j, k] != 0 and val != 1:
                    return False
                if T[i, j, k] == 0 and val != 0:
                    return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _strassen_2x2():
    """
    Return the Strassen 2x2x2 decomposition as (U, V, W) arrays of shape (4,7).
    
    For 2x2 matrices A, B, C = A*B:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    U maps A[i,j] -> sum_c U[i*2+j, c] * M_c
    V maps B[i,j] -> sum_c V[i*2+j, c] * M_c  
    W maps C[i,j] -> sum_c W[i*2+j, c] * M_c
    
    Actually, we need: C[i,j] = sum_{a,b} A[i,a]*B[a,j]
    In tensor form: T[i*2+a, a*2+j, i*2+j] = 1 for the multiplication tensor
    
    For Strassen:
    M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    M2 = (A[1,0]+A[1,1])*B[0,0]
    M3 = A[0,0]*(B[1,0]-B[1,1])
    M4 = A[1,1]*(B[0,1]-B[0,0])
    M5 = (A[0,0]+A[0,1])*B[1,1]
    M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    U[row, col]: coefficient of A[row//2, row%2] in M_col
    V[row, col]: coefficient of B[row//2, row%2] in M_col
    W[row, col]: coefficient of M_col in C[row//2, row%2]
    """
    # U: 4 rows (A indices 00,01,10,11), 7 cols (M1..M7)
    U = np.zeros((4, 7), dtype=int)
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    # M2 = (A10+A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    # M3 = A00(B10-B11)
    U[0, 2] = 1  # A00
    # M4 = A11(B01-B00)
    U[3, 3] = 1  # A11
    # M5 = (A00+A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1  # -A00
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1  # -A11
    
    # V: 4 rows (B indices 00,01,10,11), 7 cols
    V = np.zeros((4, 7), dtype=int)
    # M1 = (A00+A11)(B00+B11)
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    # M2 = (A10+A11)B00
    V[0, 1] = 1  # B00
    # M3 = A00(B10-B11)
    V[2, 2] = 1  # B10
    V[3, 2] = -1  # -B11
    # M4 = A11(B01-B00)
    V[1, 3] = 1  # B01
    V[0, 3] = -1  # -B00
    # M5 = (A00+A01)B11
    V[3, 4] = 1  # B11
    # M6 = (A10-A00)(B00+B01)
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    # M7 = (A01-A11)(B10+B11)
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    
    # W: 4 rows (C indices 00,01,10,11), 7 cols
    W = np.zeros((4, 7), dtype=int)
    # C00 = M1 + M4 - M5 + M7
    W[0, 0] = 1
    W[0, 3] = 1
    W[0, 4] = -1
    W[0, 6] = 1
    # C01 = M3 + M5
    W[1, 2] = 1
    W[1, 4] = 1
    # C10 = M2 + M4
    W[2, 1] = 1
    W[2, 3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3, 0] = 1
    W[3, 1] = -1
    W[3, 2] = 1
    W[3, 5] = 1
    
    return U, V, W

def _construct_strassen_3x5_7():
    """
    Construct Strassen-based decomposition for 3x5 * 5x7 -> 3x7.
    
    Pad A (3x5) to 4x6, B (5x7) to 6x8.
    Use 2x2 block decomposition with Strassen's 7 multiplications.
    
    A is 4x6: split into 2x2 blocks of size 2x3
    A = [[A11, A12], [A21, A22]] where each block is 2x3
    
    B is 6x8: split into 2x2 blocks of size 3x4
    B = [[B11, B12], [B21, B22]] where each block is 3x4
    
    C = A*B is 4x8: split into 2x2 blocks of size 2x4
    C = [[C11, C12], [C21, C22]] where each block is 2x4
    
    Each block multiplication (2x3)*(3x4) -> (2x4) needs further decomposition.
    
    For (2x3)*(3x4) -> (2x4):
    Pad to (2x4)*(4x4) -> (2x4) by zero-padding.
    Use Strassen on 2x2 blocks:
    A' is 2x4: [[A'11, A'12], [A'21, A'22]] where each is 1x2
    B' is 4x4: [[B'11, B'12], [B'21, B'22]] where each is 2x2
    
    This gets complex. Let me use a simpler approach.
    
    Actually, let's use the tensor decomposition approach directly.
    
    For 3x5 * 5x7:
    We can view this as computing C[i,k] = sum_j A[i,j]*B[j,k] for i=0..2, j=0..4, k=0..6
    
    Use Strassen-style decomposition by padding to 4x8 and using 2x2 blocks.
    
    Let me think differently. We'll use the following approach:
    
    Pad A to 4x8 (add zero columns), pad B to 8x8 (add zero rows/cols).
    Then use Strassen's algorithm recursively on 4x8 * 8x8 -> 4x8.
    
    Level 1: 4x8 * 8x8
    Split A into 2x2 blocks of 2x4: A = [[A11, A12], [A21, A22]]
    Split B into 2x2 blocks of 4x4: B = [[B11, B12], [B21, B22]]
    C = A*B has 2x2 blocks of 2x4: C = [[C11, C12], [C21, C22]]
    
    Strassen:
    M1 = (A11+A22)(B11+B22)  -> 2x4 * 4x4 -> 2x4
    M2 = (A21+A22)B11         -> 2x4 * 4x4 -> 2x4
    M3 = A11(B21-B22)         -> 2x4 * 4x4 -> 2x4
    M4 = A22(B12-B11)         -> 2x4 * 4x4 -> 2x4
    M5 = (A11+A12)B22         -> 2x4 * 4x4 -> 2x4
    M6 = (A21-A11)(B11+B12)   -> 2x4 * 4x4 -> 2x4
    M7 = (A12-A22)(B21+B22)   -> 2x4 * 4x4 -> 2x4
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Now each Mi is a 2x4 * 4x4 -> 2x4 multiplication.
    
    For 2x4 * 4x4 -> 2x4:
    Split A into 2x2 blocks of 1x2: A = [[A11, A12], [A21, A22]]
    Split B into 2x2 blocks of 2x2: B = [[B11, B12], [B21, B22]]
    C = A*B has 2x2 blocks of 1x2: C = [[C11, C12], [C21, C22]]
    
    Strassen:
    N1 = (A11+A22)(B11+B22)  -> 1x2 * 2x2 -> 1x2
    N2 = (A21+A22)B11         -> 1x2 * 2x2 -> 1x2
    N3 = A11(B21-B22)         -> 1x2 * 2x2 -> 1x2
    N4 = A22(B12-B11)         -> 1x2 * 2x2 -> 1x2
    N5 = (A11+A12)B22         -> 1x2 * 2x2 -> 1x2
    N6 = (A21-A11)(B11+B12)   -> 1x2 * 2x2 -> 1x2
    N7 = (A12-A22)(B21+B22)   -> 1x2 * 2x2 -> 1x2
    
    C11 = N1 + N4 - N5 + N7
    C12 = N3 + N5
    C21 = N2 + N4
    C22 = N1 - N2 + N3 + N6
    
    For 1x2 * 2x2 -> 1x2:
    This is a simple multiplication: C[j] = sum_k A[k]*B[k,j]
    Schoolbook rank = 1*2*2 = 4
    
    So total rank = 7 * 7 * 4 = 196? That's worse than schoolbook!
    
    Let me reconsider. The issue is that Strassen helps when the dimensions are powers of 2 and we can fully exploit the recursive structure.
    
    For 3x5*5x7, let me try a different approach.
    
    Actually, let me just implement the schoolbook and try to find a better decomposition through search.
    """
    # For now, return schoolbook as fallback
    return _construct_schoolbook()

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Return schoolbook decomposition
    result = _construct_schoolbook()
    
    # Verify it's correct
    if _verify(result["U"], result["V"], result["W"], 3, 5, 7):
        return result
    
    # Should not reach here, but return anyway
    return result
# EVOLVE-BLOCK-END
