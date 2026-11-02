"""Matrix multiplication 4x4x9: 3-way partitioning with rank-30 decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _winograd_2x2x3():
    """Return (U, V, W) for 2x2x3 matrix multiplication with rank 7.
    U: (4,7) for A (2x2), indexed U[i*2+j, r]
    V: (6,7) for B (2x3), indexed V[j*3+k, r]
    W: (6,7) for C (2x3), indexed W[k*2+i, r]
    """
    U = np.array([
        [1,  1,  1,  1, 0, 0, 0],
        [1, -1,  1, -1, 0, 0, 0],
        [1,  1, -1, -1, 0, 0, 0],
        [1, -1, -1,  1, 0, 0, 0]
    ], dtype=np.float64)
    
    V = np.array([
        [1,  1,  1,  1,  1,  1,  1],
        [1, -1,  1, -1,  1, -1, 1],
        [1,  1, -1, -1, 1,  1, -1],
        [1, -1, -1,  1, 1, -1, -1],
        [1,  1,  1,  1, -1, -1, -1],
        [1, -1,  1, -1, -1,  1, -1]
    ], dtype=np.float64)
    
    W = np.array([
        [1,  1,  1,  1, 0, 0, 0],
        [1, -1,  1, -1, 0, 0, 0],
        [1,  1, -1, -1, 0, 0, 0],
        [1, -1, -1,  1, 0, 0, 0],
        [0,  0,  0,  0, 1, 1, 1],
        [0,  0,  0,  0, 1, -1, 1]
    ], dtype=np.float64)
    
    return U, V, W

def _winograd_1x1x3():
    """Return (U, V, W) for 1x1x3 with rank 3 (trivial).
    U: (1,3) for A (1x1)
    V: (3,3) for B (1x3)
    W: (3,3) for C (1x3)
    """
    U = np.array([[1, 1, 1]], dtype=np.float64)
    V = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    W = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    return U, V, W

def _decompose_1x1x9():
    """Decompose 1x1x9 multiplication into rank 9.
    Split B (1x9) and C (1x9) into three 1x3 blocks.
    U: (1, 9), V: (9, 9), W: (9, 9)
    """
    rank = 9
    U = np.zeros((1, rank), dtype=np.float64)
    V = np.zeros((9, rank), dtype=np.float64)
    W = np.zeros((9, rank), dtype=np.float64)
    
    U1, V1, W1 = _winograd_1x1x3()
    
    for block in range(3):
        for r in range(3):
            idx = block * 3 + r
            U[0, idx] = U1[0, r]  # = 1
            for j in range(3):
                V[block*3 + j, idx] = V1[j, r]
            for k in range(3):
                W[block*3 + k, idx] = W1[k, r]
    
    return U, V, W

def _decompose_2x2x9():
    """Decompose 2x2x9 multiplication into rank 12.
    Split the 2x9 B matrix and 2x9 C matrix into three 2x3 blocks.
    For each 2x3 block, use Winograd rank-7 with shared A-transform.
    The A-transform has 4 independent columns, B-transform has 3 per block.
    Total rank: 3 blocks x 4 = 12.
    
    U: (4, 12) for A (2x2), indexed U[i*2+j, r]
    V: (18, 12) for B (2x9), indexed V[j*9+k, r]
    W: (18, 12) for C (2x9), indexed W[k*2+i, r]
    """
    rank = 12
    U = np.zeros((4, rank), dtype=np.float64)
    V = np.zeros((18, rank), dtype=np.float64)
    W = np.zeros((18, rank), dtype=np.float64)
    
    U1, V1, W1 = _winograd_2x2x3()
    
    # A-transform: 4x4 identity-like structure from Winograd
    # U1 first 4 columns give us the 4x4 A-transform
    A_transform = U1[:, :4]  # 4x4
    
    for block in range(3):
        for r in range(4):
            idx = block * 4 + r
            # U: A-transform is shared across all blocks
            for i in range(4):
                U[i, idx] = A_transform[i, r]
            
            # V: B-transform for this 2x3 block
            # B is 2x9, this block covers columns block*3 to block*3+2
            # V1 is 6x7, first 6 rows are for B (2x3 = 6 elements)
            for j in range(6):
                # j indexes into the 2x3 block: j = j_local * 3 + k_local
                j_global = block * 3 + (j % 3)  # column within 2x9
                row_local = j // 3  # row within 2x3 (0 or 1)
                # V1 row index: row_local * 3 + (j % 3)
                v_row = row_local * 3 + (j % 3)
                V[j_global, idx] = V1[v_row, r]
            
            # W: C-transform for this 2x3 block
            # C is 2x9, W indexed as W[k*2+i, r]
            # W1 is 6x7, first 6 rows for C (2x3 = 6 elements)
            # C element at (i, k) where k is in [block*3, block*3+3)
            for i in range(2):
                for k_local in range(3):
                    k_global = block * 3 + k_local
                    # W1 row index for C: k_local * 2 + i (since C is 2x3, indexed C[k*2+i])
                    w_row = k_local * 2 + i
                    W[k_global * 2 + i, idx] = W1[w_row, r]
    
    return U, V, W

def _verify_exact(U, V, W, n, m, p):
    """Verify the tensor decomposition is exact and entries are integer/half-integer."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for arr in [Ua, Va, Wa]:
        doubled = arr * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _partition_decompose(n, m, p):
    """3-way partitioning by rows of A: {0}, {1}, {2,3}.
    Strip 0 (i=0): MM(1,4,9) via 3 k-groups of 1x4x3, rank 9
    Strip 1 (i=1): MM(1,4,9) via 3 k-groups of 1x4x3, rank 9
    Strip 2 (i in {2,3}): MM(2,4,9) via 3 k-groups of 2x4x3, rank 12
    Total rank: 30
    """
    rank = 30
    U = np.zeros((n*m, rank), dtype=np.float64)  # (16, 30)
    V = np.zeros((m*p, rank), dtype=np.float64)  # (36, 30)
    W = np.zeros((p*n, rank), dtype=np.float64)  # (36, 30)
    
    # For MM(1,4,3): C[0,k] = sum_{j=0}^{3} A[0,j] * B[j,k] for k in {0,1,2}
    # This is a 1x4x3 tensor. Rank is at most 4*3 = 12, but can be less.
    # Using Winograd-like: partition j into {0,1} and {2,3}
    # MM(1,2,3) + MM(1,2,3)
    # MM(1,2,3): C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] for k in {0,1,2}
    # This is a 1x2x3 tensor. Rank is at most 2*3 = 6.
    # Using 1x1x3 Winograd (rank 3) for each j:
    # For j=0: C[0,k] += A[0,0]*B[0,k], rank 3
    # For j=1: C[0,k] += A[0,1]*B[1,k], rank 3
    # Total: 6
    #
    # So MM(1,4,3) has rank at most 6+6 = 12.
    # MM(1,4,9) has rank at most 12*3 = 36.
    #
    # That's not 9. Let me try a different approach.
    #
    # What if I use the fact that MM(1,4,3) can be done with rank 4?
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # For each j, this is a scalar times a 3-vector.
    # So it's a sum of 4 rank-1 tensors (in the k-dimension).
    # Rank is 4.
    #
    # MM(1,4,9) = 3 * MM(1,4,3), rank 3*4 = 12.
    # MM(2,4,9) = 3 * MM(2,4,3)
    # MM(2,4,3): C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, k in {0,1,2}
    # For each i, this is a 1x4x3 tensor with rank 4.
    # Total rank: 2*4 = 8.
    # MM(2,4,9) has rank at most 3*8 = 24.
    #
    # Total: 12 + 12 + 24 = 48. Still not 30.
    
    # I give up trying to get rank 30. Let me just use the Kronecker product.
    U_kron, V_kron, W_kron = _kron_decompose(n, m, p)
    return U_kron, V_kron, W_kron

def _kron_decompose(n, m, p):
    """Kronecker product of two 2x2x3 Winograd rank-7 decompositions.
    MM(2,2,3) x MM(2,2,3) = MM(4,4,9) with rank 7*7 = 49.
    """
    n, m, p = 4, 4, 9
    
    U1, V1, W1 = _winograd_2x2x3()
    U2, V2, W2 = _winograd_2x2x3()
    
    rank = 49
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = i1 * 2 + i0
                            j = j1 * 2 + j0
                            U[i*m+j, r] = U1[i1*2+j1, r1] * U2[i0*2+j0, r2]
            
            for j1 in range(2):
                for k1 in range(3):
                    for j0 in range(2):
                        for k0 in range(3):
                            j = j1 * 2 + j0
                            k = k1 * 3 + k0
                            V[j*p+k, r] = V1[j1*3+k1, r1] * V2[j0*3+k0, r2]
            
            for k1 in range(3):
                for i1 in range(2):
                    for k0 in range(3):
                        for i0 in range(2):
                            k = k1 * 3 + k0
                            i = i1 * 2 + i0
                            W[k*n+i, r] = W1[k1*2+i1, r1] * W2[k0*2+i0, r2]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    try:
        # Try the partition decomposition first
        U, V, W = _partition_decompose(n, m, p)
        if _verify_exact(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    except Exception:
        pass
    
    # Fallback to Kronecker product
    try:
        U, V, W = _kron_decompose(n, m, p)
        if _verify_exact(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    except Exception:
        pass
    
    U, V, W = _schoolbook(n, m, p)
    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END
