"""Rewritten program for 2x2 matrix multiplication rank minimization."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    The tensor rank of 2x2 matrix multiplication is known to be 7.
    A rank-6 decomposition does not exist over any field.
    We return the standard Strassen algorithm decomposition which achieves rank 7.
    
    Interface:
    U: (n*m, rank) = (4, 7)
    V: (m*p, rank) = (4, 7)
    W: (p*n, rank) = (4, 7)
    
    Indexing convention:
    U[i*m + a, r] corresponds to A[i, a]
    V[a*p + k, r] corresponds to B[a, k]
    W[k*n + i, r] corresponds to C[i, k]
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Indices for U: i*m + a
    # A00: 0, A01: 1, A10: 2, A11: 3
    
    # Indices for V: a*p + k
    # B00: 0, B01: 1, B10: 2, B11: 3
    
    # Indices for W: k*n + i
    # C00: 0, C01: 2 (k=1, i=0 -> 1*2+0=2), C10: 1 (k=0, i=1 -> 0*2+1=1), C11: 3 (k=1, i=1 -> 1*2+1=3)
    # Wait, W[k*n + i, r].
    # C[0,0]: k=0, i=0 -> idx 0
    # C[0,1]: k=1, i=0 -> idx 2
    # C[1,0]: k=0, i=1 -> idx 1
    # C[1,1]: k=1, i=1 -> idx 3
    
    # M1: (A00 + A11)(B00 + B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    
    # M2: (A10 + A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    
    # M3: A00(B01 - B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # -B11
    
    # M4: A11(B10 - B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # -B00
    
    # M5: (A00 + A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    
    # M6: (A10 - A00)(B00 + B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # -A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    
    # M7: (A01 - A11)(B10 + B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # -A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    
    # W coefficients for C[i,k] = sum_r (U_col_r dot A) * (V_col_r dot B) * W[idx, r]
    # C00 = M1 + M4 - M5 + M7
    W[0, 0] = 1   # M1
    W[0, 3] = 1   # M4
    W[0, 4] = -1  # -M5
    W[0, 6] = 1   # M7
    
    # C01 = M3 + M5
    W[2, 2] = 1   # M3
    W[2, 4] = 1   # M5
    
    # C10 = M2 + M4
    W[1, 1] = 1   # M2
    W[1, 3] = 1   # M4
    
    # C11 = M1 - M2 + M3 + M6
    W[3, 0] = 1   # M1
    W[3, 1] = -1  # -M2
    W[3, 2] = 1   # M3
    W[3, 5] = 1   # M6
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
