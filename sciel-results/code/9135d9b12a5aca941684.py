"""Matrix multiplication 2x4x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Returns a rank-36 tensor decomposition for 2x4x5 matrix multiplication.
    
    The decomposition uses:
    - Split P=5 into P1=3 and P2=2
    - For P1=3: Split M=4 into two 2s, each using Strassen(7) + Schoolbook 2x1(4) = 11
    - For P2=2: Split M=4 into two 2s, each using Strassen(7)
    - Total rank: 2*11 + 2*7 = 22 + 14 = 36
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 36
    
    U = np.zeros((n * m, rank))  # (8, 36)
    V = np.zeros((m * p, rank))  # (20, 36)
    W = np.zeros((p * n, rank))  # (10, 36)
    
    col = [0]
    
    def add_strassen(a_row, a_col, b_row, b_col, c_row, c_col):
        """Add a 2x2x2 Strassen decomposition at the given offsets."""
        # U indices: A is n x m, so A[i,j] is at U[i*m + j]
        u00 = a_row * m + a_col
        u01 = a_row * m + a_col + 1
        u10 = (a_row + 1) * m + a_col
        u11 = (a_row + 1) * m + a_col + 1
        
        # V indices: B is m x p, so B[i,j] is at V[i*p + j]
        v00 = b_row * p + b_col
        v01 = b_row * p + b_col + 1
        v10 = (b_row + 1) * p + b_col
        v11 = (b_row + 1) * p + b_col + 1
        
        # W indices: C is p x n, so C[i,j] is at W[i*n + j]
        w00 = c_col * n + c_row
        w01 = (c_col + 1) * n + c_row
        w10 = c_col * n + c_row + 1
        w11 = (c_col + 1) * n + c_row + 1
        
        # Strassen's 7 products for 2x2x2:
        # M1 = (A00 + A11)(B00 + B11)
        U[u00, col[0]] += 1; U[u11, col[0]] += 1
        V[v00, col[0]] += 1; V[v11, col[0]] += 1
        W[w00, col[0]] += 1; W[w11, col[0]] += 1
        col[0] += 1
        
        # M2 = (A10 + A11)B00
        U[u10, col[0]] += 1; U[u11, col[0]] += 1
        V[v00, col[0]] += 1
        W[w10, col[0]] += 1; W[w11, col[0]] -= 1
        col[0] += 1
        
        # M3 = A00(B11 - B01)
        U[u00, col[0]] += 1
        V[v01, col[0]] -= 1; V[v11, col[0]] += 1
        W[w01, col[0]] += 1; W[w11, col[0]] += 1
        col[0] += 1
        
        # M4 = A11(B00 - B10)
        U[u11, col[0]] += 1
        V[v00, col[0]] += 1; V[v10, col[0]] -= 1
        W[w00, col[0]] += 1; W[w10, col[0]] += 1
        col[0] += 1
        
        # M5 = (A00 + A01)B11
        U[u00, col[0]] += 1; U[u01, col[0]] += 1
        V[v11, col[0]] += 1
        W[w00, col[0]] -= 1; W[w01, col[0]] += 1
        col[0] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[u10, col[0]] += 1; U[u00, col[0]] -= 1
        V[v00, col[0]] += 1; V[v01, col[0]] += 1
        W[w11, col[0]] += 1
        col[0] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[u01, col[0]] += 1; U[u11, col[0]] -= 1
        V[v10, col[0]] += 1; V[v11, col[0]] += 1
        W[w00, col[0]] += 1
        col[0] += 1
    
    def add_schoolbook_2x1(a_row, a_col, b_row, b_col, c_row, c_col):
        """Add a 2x2x1 schoolbook decomposition at the given offsets."""
        u00 = a_row * m + a_col
        u01 = a_row * m + a_col + 1
        u10 = (a_row + 1) * m + a_col
        u11 = (a_row + 1) * m + a_col + 1
        
        v0 = b_row * p + b_col
        v1 = (b_row + 1) * p + b_col
        
        w0 = c_col * n + c_row
        w1 = c_col * n + c_row + 1
        
        U[u00, col[0]] = 1; V[v0, col[0]] = 1; W[w0, col[0]] = 1; col[0] += 1
        U[u01, col[0]] = 1; V[v1, col[0]] = 1; W[w0, col[0]] = 1; col[0] += 1
        U[u10, col[0]] = 1; V[v0, col[0]] = 1; W[w1, col[0]] = 1; col[0] += 1
        U[u11, col[0]] = 1; V[v1, col[0]] = 1; W[w1, col[0]] = 1; col[0] += 1
    
    def add_2x2x3(a_row, a_col, b_row, b_col, c_row, c_col):
        """
        Add a 2x2x3 decomposition.
        A is 2x2, B is 2x3, C is 2x3.
        Split B into B1 (2x2, cols 0-1) and B2 (2x1, col 2).
        C1 = A*B1 using Strassen (7 products)
        C2 = A*B2 using Schoolbook (4 products)
        Total: 11 products
        """
        # Strassen for the first 2 columns
        add_strassen(a_row, a_col, b_row, b_col, c_row, c_col)
        # Schoolbook for the 3rd column
        add_schoolbook_2x1(a_row, a_col, b_row, b_col + 2, c_row, c_col + 2)
    
    # Part 1: C[:, 0:3] = A @ B[:, 0:3]
    # Split M=4 into M1=2 (rows 0-1) and M2=2 (rows 2-3)
    # A1 (2x2) @ B11 (2x3) -> C1 (2x3) using 2x2x3
    add_2x2x3(0, 0, 0, 0, 0, 0)
    # A2 (2x2) @ B12 (2x3) -> C2 (2x3) using 2x2x3
    add_2x2x3(0, 2, 2, 0, 0, 0)
    
    # Part 2: C[:, 3:5] = A @ B[:, 3:5]
    # Split M=4 into M1=2 (rows 0-1) and M2=2 (rows 2-3)
    # A1 (2x2) @ B21 (2x2) -> C1 (2x2) using Strassen
    add_strassen(0, 0, 0, 3, 0, 3)
    # A2 (2x2) @ B22 (2x2) -> C2 (2x2) using Strassen
    add_strassen(0, 2, 2, 3, 0, 3)
    
    # Verify col reached rank
    assert col[0] == rank, f"Expected rank {rank}, got {col[0]}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
