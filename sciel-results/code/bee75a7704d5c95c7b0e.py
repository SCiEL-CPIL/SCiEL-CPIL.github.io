import numpy as np

DIMENSIONS = [2, 3, 5]

def _add_strassen_block(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    """
    Adds a 2x2x2 Strassen block to the decomposition.
    Indices:
    U: i*m + j
    V: j*p + k
    W: k*n + i
    """
    # M1: (A[0,0]+A[1,1])(B[0,0]+B[1,1]) -> C[0,0], C[1,1]
    U[i_offset*m + j_offset, col] += 1
    U[(i_offset+1)*m + (j_offset+1), col] += 1
    V[j_offset*p + k_offset, col] += 1
    V[(j_offset+1)*p + (k_offset+1), col] += 1
    W[k_offset*n + i_offset, col] += 1
    W[(k_offset+1)*n + (i_offset+1), col] += 1
    col += 1

    # M2: (A[1,0]+A[1,1])B[0,0] -> C[1,0], -C[1,1]
    U[(i_offset+1)*m + j_offset, col] += 1
    U[(i_offset+1)*m + (j_offset+1), col] += 1
    V[j_offset*p + k_offset, col] += 1
    W[k_offset*n + (i_offset+1), col] += 1
    W[(k_offset+1)*n + (i_offset+1), col] -= 1
    col += 1

    # M3: A[0,0](B[0,1]-B[1,1]) -> C[0,1], C[1,1]
    U[i_offset*m + j_offset, col] += 1
    V[j_offset*p + (k_offset+1), col] += 1
    V[(j_offset+1)*p + (k_offset+1), col] -= 1
    W[(k_offset+1)*n + i_offset, col] += 1
    W[(k_offset+1)*n + (i_offset+1), col] += 1
    col += 1

    # M4: A[1,1](B[1,0]-B[0,0]) -> C[0,0], C[1,0]
    U[(i_offset+1)*m + (j_offset+1), col] += 1
    V[(j_offset+1)*p + k_offset, col] += 1
    V[j_offset*p + k_offset, col] -= 1
    W[k_offset*n + i_offset, col] += 1
    W[k_offset*n + (i_offset+1), col] += 1
    col += 1

    # M5: (A[0,0]+A[0,1])B[1,1] -> -C[0,0], C[0,1]
    U[i_offset*m + j_offset, col] += 1
    U[i_offset*m + (j_offset+1), col] += 1
    V[(j_offset+1)*p + (k_offset+1), col] += 1
    W[k_offset*n + i_offset, col] -= 1
    W[(k_offset+1)*n + i_offset, col] += 1
    col += 1

    # M6: (A[1,0]-A[0,0])(B[0,0]+B[0,1]) -> C[1,1]
    U[(i_offset+1)*m + j_offset, col] += 1
    U[i_offset*m + j_offset, col] -= 1
    V[j_offset*p + k_offset, col] += 1
    V[j_offset*p + (k_offset+1), col] += 1
    W[(k_offset+1)*n + (i_offset+1), col] += 1
    col += 1

    # M7: (A[0,1]-A[1,1])(B[1,0]+B[1,1]) -> C[0,0]
    U[i_offset*m + (j_offset+1), col] += 1
    U[(i_offset+1)*m + (j_offset+1), col] -= 1
    V[(j_offset+1)*p + k_offset, col] += 1
    V[(j_offset+1)*p + (k_offset+1), col] += 1
    W[k_offset*n + i_offset, col] += 1
    col += 1

    return col

def _add_schoolbook_block_2xK(U, V, W, col, j_index, n, m, p):
    """
    Adds a 2x1xK schoolbook block for a specific j.
    C_{ik} += A_{ij} B_{jk}
    """
    for i in range(n):
        for k in range(p):
            U[i*m + j_index, col] += 1
            V[j_index*p + k, col] += 1
            W[k*n + i, col] += 1
            col += 1
    return col

def _add_schoolbook_block_2x2x1(U, V, W, col, j_offset, k_index, n, m, p):
    """
    Adds a 2x2x1 schoolbook block.
    C_{ik} += sum_{j in {j_offset, j_offset+1}} A_{ij} B_{j, k_index}
    """
    for i in range(n):
        for j in range(2):
            j_global = j_offset + j
            U[i*m + j_global, col] += 1
            V[j_global*p + k_index, col] += 1
            W[k_index*n + i, col] += 1
            col += 1
    return col

def construct():
    n, m, p = DIMENSIONS
    
    # Total rank 28
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Block 1: j=2, all k (2x1x5) -> rank 10
    col = _add_schoolbook_block_2xK(U, V, W, col, 2, n, m, p)
    
    # Block 2: j=0,1, k=0,1 (2x2x2) -> rank 7
    col = _add_strassen_block(U, V, W, col, 0, 0, 0, n, m, p)
    
    # Block 3: j=0,1, k=2 (2x2x1) -> rank 4
    col = _add_schoolbook_block_2x2x1(U, V, W, col, 0, 2, n, m, p)
    
    # Block 4: j=0,1, k=3,4 (2x2x2) -> rank 7
    col = _add_strassen_block(U, V, W, col, 0, 0, 3, n, m, p)
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
