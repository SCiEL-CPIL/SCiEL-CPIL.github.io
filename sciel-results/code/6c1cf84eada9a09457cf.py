"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]
_DEADLINE = time.time() + 135  # safety margin before 150s kill

def _get_bbc_2x2x2():
    """
    Returns the Bini-Brenti-Cursi rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns list of (U, V, W) where U, V, W are 2x2 numpy arrays.
    """
    # Standard BBC components
    # U, V, W are 2x2 matrices
    components = []
    
    # r=0
    components.append((
        np.array([[1, 0], [0, 1]], dtype=float),
        np.array([[1, 0], [0, 1]], dtype=float),
        np.array([[1, 0], [0, 1]], dtype=float)
    ))
    # r=1
    components.append((
        np.array([[1, 0], [0, 0]], dtype=float),
        np.array([[1, 0], [0, 0]], dtype=float),
        np.array([[1, 0], [0, 0]], dtype=float)
    ))
    # r=2
    components.append((
        np.array([[0, 1], [0, 0]], dtype=float),
        np.array([[1, 0], [0, 0]], dtype=float),
        np.array([[0, 1], [0, 0]], dtype=float)
    ))
    # r=3
    components.append((
        np.array([[0, 0], [1, 0]], dtype=float),
        np.array([[0, 1], [0, 0]], dtype=float),
        np.array([[0, 0], [1, 0]], dtype=float)
    ))
    # r=4
    components.append((
        np.array([[0, 0], [0, 1]], dtype=float),
        np.array([[0, 1], [0, 1]], dtype=float),
        np.array([[0, 0], [0, 1]], dtype=float)
    ))
    # r=5
    components.append((
        np.array([[1, 1], [0, 0]], dtype=float),
        np.array([[0, 0], [1, 0]], dtype=float),
        np.array([[1, 0], [0, 0]], dtype=float)
    ))
    # r=6
    components.append((
        np.array([[0, 1], [0, 1]], dtype=float),
        np.array([[0, 0], [0, 1]], dtype=float),
        np.array([[0, 1], [0, 0]], dtype=float)
    ))
    
    return components

def _build_4x4x4_bbc():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication using Kronecker product of BBC 2x2x2.
    Returns U, V, W as numpy arrays of shape (16, 49), (16, 49), (16, 49).
    Note: For 4x4x4, n=4, m=4, p=4.
    U shape: (n*m, rank) = (16, 49)
    V shape: (m*p, rank) = (16, 49)
    W shape: (p*n, rank) = (16, 49)
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    bbc = _get_bbc_2x2x2()
    
    col = 0
    for r1 in range(7):
        for r2 in range(7):
            U1, V1, W1 = bbc[r1]
            U2, V2, W2 = bbc[r2]
            
            # Kronecker product of the 2x2 matrices
            U_kron = np.kron(U1, U2)
            V_kron = np.kron(V1, V2)
            W_kron = np.kron(W1, W2)
            
            # Flatten and place into the rank column
            # The row indices for U in 4x4x4 are (i, j) mapped to i*4+j.
            # np.kron(U1, U2) has shape 4x4.
            # Row index in kron corresponds to (i1, i2) -> i1*2+i2? 
            # np.kron(A, B) where A is 2x2, B is 2x2.
            # Result is 4x4.
            # Index (i, j) in result corresponds to A[i//2, j//2] * B[i%2, j%2].
            # This matches the standard row-major flattening if we consider i = i1*2+i2, j=j1*2+j2.
            # So U_kron[i, j] is the value for row i, col j.
            # We need to place this into U[i*4+j, col].
            
            U[:, col] = U_kron.flatten()
            V[:, col] = V_kron.flatten()
            W[:, col] = W_kron.flatten()
            
            col += 1
            
    return U, V, W

def _build_4x4x1_schoolbook():
    """
    Build rank-16 decomposition for 4x4x1 matrix multiplication (schoolbook).
    n=4, m=4, p=1.
    U shape: (16, 16)
    V shape: (4, 16)
    W shape: (4, 16)
    """
    n, m, p = 4, 4, 1
    rank = n * m
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p, col] = 1
            W[0 * n + i, col] = 1 # k=0
            col += 1
            
    return U, V, W

def construct():
    """
    Construct a tensor decomposition for 4x4x9 matrix multiplication.
    Uses BBC rank-49 for two 4x4x4 blocks and schoolbook rank-16 for one 4x4x1 block.
    Total rank = 49 + 49 + 16 = 114.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # 1. Build 4x4x4 block 1 (k=0..3)
    U1, V1, W1 = _build_4x4x4_bbc()
    
    # 2. Build 4x4x4 block 2 (k=4..7)
    # The structure is identical, just shifted in the V and W indices.
    # V for 4x4x4 has shape (16, 49). Rows are j*p+k.
    # For block 1, k is 0..3. V1 rows correspond to j*4+k where k=0..3.
    # For block 2, k is 4..7.
    # We can reuse the same U, but V and W need to be permuted/shifted.
    # Actually, it's easier to just build a second 4x4x4 decomposition and place it correctly.
    U2, V2, W2 = _build_4x4x4_bbc()
    
    # 3. Build 4x4x1 block (k=8)
    U3, V3, W3 = _build_4x4x1_schoolbook()
    
    # Now assemble the full matrices.
    # U has shape (16, 114).
    # U1, U2, U3 all have 16 rows.
    # U1 cols 0..48, U2 cols 49..97, U3 cols 98..113.
    
    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank)) # m*p = 4*9 = 36
    W = np.zeros((36, total_rank)) # p*n = 9*4 = 36
    
    # Place Block 1
    U[:, 0:49] = U1
    # V1 has shape (16, 49). Rows j*4+k for k=0..3.
    # In full V, rows are j*9+k.
    # For a given j (0..3) and k (0..3), V1 row is j*4+k.
    # In V, row is j*9+k.
    for j in range(4):
        for k in range(4):
            V[j * 9 + k, 0:49] = V1[j * 4 + k, :]
    
    # W1 has shape (16, 49). Rows k*4+i for k=0..3.
    # In full W, rows are k*4+i? No, W index is k*n+i = k*4+i.
    # Wait, W shape is (p*n, rank) = (36, rank).
    # Index is k*n + i.
    # For block 1, k=0..3.
    # W1 row index is k*4+i.
    # In W, row index is k*4+i.
    # So W1 maps directly to W rows 0..15.
    W[0:16, 0:49] = W1
    
    # Place Block 2
    U[:, 49:98] = U2
    # V2 rows are j*4+k for k=0..3.
    # In full V, these correspond to k=4..7.
    # V row index is j*9 + (4+k).
    for j in range(4):
        for k in range(4):
            V[j * 9 + 4 + k, 49:98] = V2[j * 4 + k, :]
    
    # W2 rows are k*4+i for k=0..3.
    # In full W, these correspond to k=4..7.
    # W row index is (4+k)*4 + i = 16 + k*4 + i.
    W[16:32, 49:98] = W2
    
    # Place Block 3 (4x4x1)
    U[:, 98:114] = U3
    # V3 shape (4, 16). Rows j*1+0 = j.
    # In full V, k=8. Row index j*9+8.
    for j in range(4):
        V[j * 9 + 8, 98:114] = V3[j, :]
    
    # W3 shape (4, 16). Rows 0*4+i = i.
    # In full W, k=8. Row index 8*4+i = 32+i.
    W[32:36, 98:114] = W3
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
