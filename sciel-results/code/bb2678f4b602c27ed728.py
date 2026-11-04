"""Matrix multiplication 4x4x8 tensor decomposition.

We use a Strassen-Kronecker approach:
- 2x2x2 rank-7 Strassen decomposition
- 4x4x4 via Kronecker product: rank 49
- 4x4x8 by stacking two 4x4x4 decompositions: rank 98

The output factor W uses the trace convention W[k*n+i, r].
"""
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2x2():
    """Return (U, V, W) each 4x7 for 2x2x2 matrix multiplication."""
    # M1 = A00*(B01+B11), M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11), M4 = A11*(B00-B10)
    # M5 = (A00+A01)*B11, M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # U rows indexed by (i,j) in row-major: 00,01,10,11
    # U[(i,j), m] = coefficient of A_{i,j} in M_m
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, -1],   # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, 1],    # A11
    ], dtype=float)
    
    # V rows indexed by (j,k) in row-major: 00,01,10,11
    # V[(j,k), m] = coefficient of B_{j,k} in M_m
    V = np.array([
        [0, 1, 0, 1, 0, 1, 0],    # B00
        [1, 0, -1, 0, 0, 1, 0],   # B01
        [0, 0, 1, -1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1],   # B11
    ], dtype=float)
    
    # W rows indexed by (k,i) in row-major: 00,01,10,11  (k*n+i, n=2)
    # W[(k,i), m] = coefficient of M_m in C_{k,i}
    # C00 = M1 - M3 + M4 + M5
    # C01 = M6
    # C10 = M2
    # C11 = M1 + M3 - M5 + M7
    W = np.array([
        [1, 0, -1, 1, 1, 0, 0],   # C00 (k=0,i=0)
        [0, 1, 0, 0, 0, 0, 0],    # C01 (k=0,i=1)
        [0, 0, 0, 0, 0, 1, 0],    # C10 (k=1,i=0)
        [1, 0, 1, 0, -1, 0, 1],   # C11 (k=1,i=1)
    ], dtype=float)
    
    return U, V, W

def _kron(U1, U2):
    """Kronecker product of two factor matrices for tensor product."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    n1 = U1.shape[0]
    n2 = U2.shape[0]
    U = np.zeros((n1 * n2, r1 * r2))
    for i1 in range(n1):
        for i2 in range(n2):
            for c1 in range(r1):
                for c2 in range(r2):
                    U[i1 * n2 + i2, c1 * r2 + c2] = U1[i1, c1] * U2[i2, c2]
    return U

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Get 2x2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2x2()
    
    # 4x4x4 via Kronecker product (two levels of 2x2)
    U4 = _kron(U2, U2)  # shape (16, 49)
    V4 = _kron(V2, V2)  # shape (16, 49)
    W4 = _kron(W2, W2)  # shape (16, 49)
    
    # Verify 4x4x4 decomposition correctness
    # U4[i*4+j, c], V4[j*4+k, c], W4[k*4+i, c]
    # Check: sum_c U4[i*4+j,c] * V4[j*4+k,c] * W4[k*4+i,c] = delta(i,j,k) pattern
    
    # For 4x4x8: split the 8 columns into two 4-column blocks
    # C[:, 0:4] = A @ B[:, 0:4]
    # C[:, 4:8] = A @ B[:, 4:8]
    
    # U stays the same (same A matrix)
    # V is stacked: first 49 columns for B[:,0:4], next 49 for B[:,4:8]
    # W is stacked: first 49 columns for C[:,0:4], next 49 for C[:,4:8]
    
    rank = 98
    U = np.zeros((n * m, rank))  # 16 x 98
    V = np.zeros((m * p, rank))  # 32 x 98
    W = np.zeros((p * n, rank))  # 32 x 98
    
    # First block: columns 0-48
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # Second block: columns 49-97
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4
    W[16:32, 49:98] = W4
    
    # Convert to lists of lists with integer/half-integer entries
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure entries are exactly representable (they should be integers or halves)
    # Convert near-integers to integers
    def clean(val):
        if abs(val - round(val)) < 1e-10:
            return int(round(val))
        else:
            return val
    
    U_clean = [[clean(x) for x in row] for row in U_list]
    V_clean = [[clean(x) for x in row] for row in V_list]
    W_clean = [[clean(x) for x in row] for row in W_list]
    
    return {"U": U_clean, "V": V_clean, "W": W_clean}
