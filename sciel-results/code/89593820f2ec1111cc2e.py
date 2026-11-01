"""
Program to find a rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication.
Uses the known optimal rank 14 decomposition based on two Strassen 2x2x2 blocks.
The verification is vectorized for speed.
"""
import numpy as np
import time

DIMENSIONS = [2, 2, 4]
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # 150s limit, 5s margin

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=2, p=4):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W index convention: W[k*n+i] where k=output column, i=output row.
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for valid (i,j,k)
    and 0 for invalid triples.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Construct the full tensor T of size (n*m, m*p, p*n)
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # We can compute this efficiently using einsum
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Compute T[a, b, c]
    # T = np.einsum('ar,br,cr->abc', U, V, W)
    # This creates a tensor of size (4, 8, 8) = 256 elements. Very fast.
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check valid entries
    # Valid entries are those where:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # This implies:
    # j = a % m, i = a // m
    # j = b // p, k = b % p
    # k = c // n, i = c % n
    
    # So we need to check T[a, b, c] == 1 if:
    # a // m == c % n (i matches)
    # a % m == b // p (j matches)
    
    for a in range(n * m):
        i = a // m
        j = a % m
        
        for b in range(m * p):
            j_b = b // p
            k = b % p
            
            # If j matches, then c is determined by k and i
            if j == j_b:
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return False
    
    # Check invalid entries
    # We need to ensure T[a, b, c] == 0 for all other (a, b, c)
    # Instead of iterating all 256, we can check the ones that are not valid.
    # Or simply check that the sum of squares of "invalid" parts is 0.
    # But it's safer to just check the specific invalid spots or rely on the structure.
    # Given the small size, we can just check all and skip valid ones.
    
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j_b = b // p
            k = b % p
            for c in range(p * n):
                i_c = c % n
                k_c = c // n
                
                # Valid if i == i_c and j == j_b
                if i == i_c and j == j_b:
                    continue
                    
                if abs(T[a, b, c]) > 1e-9:
                    return False
    
    return True

def _build_rank14():
    """
    Construct the rank-14 decomposition using two Strassen blocks.
    W index convention: W[k*n+i] where k=output column, i=output row.
    C[0,0]->W[0], C[1,0]->W[1], C[0,1]->W[2], C[1,1]->W[3],
    C[0,2]->W[4], C[1,2]->W[5], C[0,3]->W[6], C[1,3]->W[7]
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # W (C) indices: C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3],
    #                C[0,2]=W[4], C[1,2]=W[5], C[0,3]=W[6], C[1,3]=W[7]
    
    # Left Block (Columns 0, 1) - Strassen 2x2x2
    # m1 = (A00 + A11) * (B00 + B11)
    r = 0
    U[0][r] = 1; U[3][r] = 1
    V[0][r] = 1; V[5][r] = 1
    W[0][r] = 1; W[3][r] = 1
    
    # m2 = (A10 + A11) * B00
    r = 1
    U[2][r] = 1; U[3][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1
    
    # m3 = A00 * (B01 - B11)
    r = 2
    U[0][r] = 1
    V[1][r] = 1; V[5][r] = -1
    W[2][r] = 1; W[3][r] = 1
    
    # m4 = A11 * (B10 - B00)
    r = 3
    U[3][r] = 1
    V[4][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[1][r] = 1
    
    # m5 = (A00 + A01) * B11
    r = 4
    U[0][r] = 1; U[1][r] = 1
    V[5][r] = 1
    W[0][r] = -1; W[2][r] = 1
    
    # m6 = (A10 - A00) * (B00 + B01)
    r = 5
    U[2][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    
    # m7 = (A01 - A11) * (B10 + B11)
    r = 6
    U[1][r] = 1; U[3][r] = -1
    V[4][r] = 1; V[5][r] = 1
    W[0][r] = 1
    
    # Right Block (Columns 2, 3) - Strassen 2x2x2
    # m1 = (A00 + A11) * (B02 + B13)
    r = 7
    U[0][r] = 1; U[3][r] = 1
    V[2][r] = 1; V[7][r] = 1
    W[4][r] = 1; W[7][r] = 1
    
    # m2 = (A10 + A11) * B02
    r = 8
    U[2][r] = 1; U[3][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1
    
    # m3 = A00 * (B03 - B13)
    r = 9
    U[0][r] = 1
    V[3][r] = 1; V[7][r] = -1
    W[6][r] = 1; W[7][r] = 1
    
    # m4 = A11 * (B12 - B02)
    r = 10
    U[3][r] = 1
    V[6][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[5][r] = 1
    
    # m5 = (A00 + A01) * B13
    r = 11
    U[0][r] = 1; U[1][r] = 1
    V[7][r] = 1
    W[4][r] = -1; W[6][r] = 1
    
    # m6 = (A10 - A00) * (B02 + B03)
    r = 12
    U[2][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1
    
    # m7 = (A01 - A11) * (B12 + B13)
    r = 13
    U[1][r] = 1; U[3][r] = -1
    V[6][r] = 1; V[7][r] = 1
    W[4][r] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline check
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _build_rank14()
    
    # Build the known rank-14 solution
    result_14 = _build_rank14()
    
    # Verify the rank-14 solution
    if _verify(result_14["U"], result_14["V"], result_14["W"], n, m, p):
        return result_14
    
    # Fallback to schoolbook rank 16 (should not happen)
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
# EVOLVE-BLOCK-END
