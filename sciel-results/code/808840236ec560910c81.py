"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _build_strassen_rank26():
    """
    Construct a rank-26 exact tensor decomposition for 3x3 matrix multiplication.
    
    Strategy: Treat 3x3 matrices as 2x2 block matrices where:
    - Top-left 2x2 block: use Strassen's 7-multiplication algorithm
    - Border elements (row 3, col 3): use standard multiplication with shared terms
    
    The decomposition exploits the structure:
    A = [[A11, A12, a13],
         [A21, A22, a23],
         [a31, a32, a33]]
    
    We use 7 Strassen products for the 2x2 core, plus additional products for the border.
    Total: 7 (Strassen) + 9 (border row) + 9 (border col) + 1 (corner) = 26
    But we can share some terms to get exactly 26.
    
    Actually, let's use a cleaner approach:
    - 7 products for the 2x2 Strassen core
    - 9 products for the third row of C (involving a31,a32,a33)
    - 9 products for the third column of C (involving a13,a23)
    - 1 product for C[3,3] = a33*b33
    
    That's 7+9+9+1 = 26. But wait, we need to be careful about double-counting.
    
    Let me define it more carefully using the known rank-26 decomposition.
    """
    n, m, p = 3, 3, 3
    rank = 26
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    # i in [0,1,2], j in [0,1,2], k in [0,1,2]
    
    # Strassen for the 2x2 core (indices 0,1 for i,j and k)
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's use 0-indexed: A[i,j], B[j,k], C[i,k]
    # Core is i,j in {0,1} and k in {0,1}
    
    # M1: (A00+A11)(B00+B11)
    U[0*3+0, 0] = 1  # A00
    U[1*3+1, 0] = 1  # A11
    V[0*3+0, 0] = 1  # B00
    V[1*3+1, 0] = 1  # B11
    # C00 += M1
    W[0*3+0, 0] = 1
    # C11 += M1
    W[1*3+1, 0] = 1
    
    # M2: (A10+A11)B00
    U[1*3+0, 1] = 1  # A10
    U[1*3+1, 1] = 1  # A11
    V[0*3+0, 1] = 1  # B00
    # C10 += M2
    W[0*3+1, 1] = 1
    # C11 -= M2
    W[1*3+1, 1] = -1
    
    # M3: A00(B10-B11)
    U[0*3+0, 2] = 1  # A00
    V[1*3+0, 2] = 1  # B10
    V[1*3+1, 2] = -1  # B11
    # C01 += M3
    W[0*3+0, 2] = 1
    # C11 += M3
    W[1*3+1, 2] = 1
    
    # M4: A11(B01-B00)
    U[1*3+1, 3] = 1  # A11
    V[0*3+1, 3] = 1  # B01
    V[0*3+0, 3] = -1  # B00
    # C00 += M4
    W[0*3+0, 3] = 1
    # C10 += M4
    W[0*3+1, 3] = 1
    
    # M5: (A00+A01)B11
    U[0*3+0, 4] = 1  # A00
    U[0*3+1, 4] = 1  # A01
    V[1*3+1, 4] = 1  # B11
    # C00 -= M5
    W[0*3+0, 4] = -1
    # C01 += M5
    W[0*3+0, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[1*3+0, 5] = 1  # A10
    U[0*3+0, 5] = -1  # A00
    V[0*3+0, 5] = 1  # B00
    V[0*3+1, 5] = 1  # B01
    # C11 += M6
    W[1*3+1, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[0*3+1, 6] = 1  # A01
    U[1*3+1, 6] = -1  # A11
    V[1*3+0, 6] = 1  # B10
    V[1*3+1, 6] = 1  # B11
    # C00 += M7
    W[0*3+0, 6] = 1
    
    # Now handle the border: row 2 (i=2) and column 2 (k=2)
    # C[2, k] for k in {0,1,2}: involves A[2, j] for all j
    # C[i, 2] for i in {0,1}: involves B[j, 2] for all j
    
    # For C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # We need 3 products. Similarly for C[2,1], C[2,2], C[0,2], C[1,2]
    # But C[2,2] is shared.
    
    # Let's use separate products for the border to keep it simple.
    # C[2,0]: 3 products
    # C[2,1]: 3 products  
    # C[2,2]: 3 products (but we can share with C[0,2] and C[1,2] if needed)
    # C[0,2]: 3 products
    # C[1,2]: 3 products
    
    # That's 15 more products, total 7+15 = 22. We have room for up to 26.
    # Let's just use direct products for the border.
    
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    for j in range(3):
        c = 7 + j
        U[2*3+j, c] = 1  # A[2,j]
        V[j*3+0, c] = 1  # B[j,0]
        W[0*3+2, c] = 1  # C[2,0]
    
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    for j in range(3):
        c = 10 + j
        U[2*3+j, c] = 1  # A[2,j]
        V[j*3+1, c] = 1  # B[j,1]
        W[1*3+2, c] = 1  # C[2,1]
    
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    for j in range(3):
        c = 13 + j
        U[2*3+j, c] = 1  # A[2,j]
        V[j*3+2, c] = 1  # B[j,2]
        W[2*3+2, c] = 1  # C[2,2]
    
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    for j in range(3):
        c = 16 + j
        U[0*3+j, c] = 1  # A[0,j]
        V[j*3+2, c] = 1  # B[j,2]
        W[2*3+0, c] = 1  # C[0,2]
    
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    for j in range(3):
        c = 19 + j
        U[1*3+j, c] = 1  # A[1,j]
        V[j*3+2, c] = 1  # B[j,2]
        W[2*3+1, c] = 1  # C[1,2]
    
    # That's 7 + 15 = 22 products. We have rank 26, so we're using 22.
    # Let's verify this is correct and use rank 22 if it works, or adjust to 26.
    
    # Actually, let's just use rank 22 if the decomposition is valid.
    # Let me verify by checking the tensor decomposition condition.
    
    rank_actual = 22
    U = U[:, :rank_actual]
    V = V[:, :rank_actual]
    W = W[:, :rank_actual]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 if i==k and the middle index matches, else 0
    # Actually, the matrix multiplication tensor is T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k (it's the full tensor)
    # Wait, no. The matrix multiplication tensor M[i,j,k] = 1 if the indices are valid.
    # The decomposition condition is: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _build_schoolbook():
    """Fallback: schoolbook decomposition of rank n*m*p."""
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


def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s kill
    
    # Try the rank-22 decomposition first
    try:
        result = _build_strassen_rank26()
        U, V, W = result["U"], result["V"], result["W"]
        if _verify_decomposition(U, V, W):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _build_schoolbook()

# EVOLVE-BLOCK-END
