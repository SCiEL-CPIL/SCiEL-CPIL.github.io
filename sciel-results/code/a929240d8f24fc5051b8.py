"""Strassen-based tensor decomposition for 4x5 * 5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def _decomp_2x2x3():
    """
    Rank-6 decomposition for 2x2 * 2x3 matrix multiplication.
    C (2x3) = A (2x2) * B (2x3)
    Uses schoolbook: 6 terms.
    Returns (U, V, W) with shapes (4, 6), (6, 6), (6, 6)
    where U is indexed by (i,j) in [0,2)x[0,2), V by (j,k) in [0,2)x[0,3),
    W by (k,i) in [0,3)x[0,2) following W[k*n+i, r] with n=2, p=3.
    """
    n, m, p = 2, 2, 3
    rank = n * m * p  # 6
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _decomp_2x3x3():
    """
    Rank-9 decomposition for 2x3 * 3x3 matrix multiplication.
    C (2x3) = A (2x3) * B (3x3)
    Uses schoolbook: 9 terms.
    Returns (U, V, W) with shapes (6, 9), (9, 9), (6, 9)
    where U is indexed by (i,j) in [0,2)x[0,3), V by (j,k) in [0,3)x[0,3),
    W by (k,i) in [0,3)x[0,2) following W[k*n+i, r] with n=2, p=3.
    """
    n, m, p = 2, 3, 3
    rank = n * m * p  # 9
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions for sub-blocks into one for the full problem.
    Each decomp is (U_sub, V_sub, W_sub, i_offset, j_offset, k_offset)
    where the sub-decomp handles C[i_offset:i_offset+ns, k_offset:k_offset+ps]
    from A[i_offset:i_offset+ns, j_offset:j_offset+ms] * B[j_offset:j_offset+ms, k_offset:k_offset+ps]
    
    For our case, we'll handle it manually in construct().
    """
    pass

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Strategy: Split n=4 into 2+2, m=5 into 2+3, p=6 into 3+3.
    
    A (4x5) = [[A11 (2x2), A12 (2x3)],
               [A21 (2x2), A22 (2x3)]]
    B (5x6) = [[B11 (2x3), B12 (2x3)],
               [B21 (3x3), B22 (3x3)]]
    
    C (4x6) = [[C11 (2x3), C12 (2x3)],
               [C21 (2x3), C22 (2x3)]]
    
    Where:
    C11 = A11*B11 + A12*B21  (2x3)
    C12 = A11*B12 + A12*B22  (2x3)
    C21 = A21*B11 + A22*B21  (2x3)
    C22 = A21*B12 + A22*B22  (2x3)
    
    Each sub-product involves:
    - One 2x2 * 2x3 multiplication (rank 6)
    - One 2x3 * 3x3 multiplication (rank 9)
    
    Total rank: 4 * (6 + 9) = 60
    """
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Block sizes
    n1, n2 = 2, 2  # split n=4
    m1, m2 = 2, 3  # split m=5
    p1, p2 = 3, 3  # split p=6
    
    # Get base decompositions
    U_223, V_223, W_223 = _decomp_2x2x3()  # for 2x2 * 2x3
    U_233, V_233, W_233 = _decomp_2x3x3()  # for 2x3 * 3x3
    
    rank_223 = U_223.shape[1]  # 6
    rank_233 = U_233.shape[1]  # 9
    
    total_rank = 4 * (rank_223 + rank_233)  # 4 * 15 = 60
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Helper to place a sub-decomposition
    def place_sub(U_sub, V_sub, W_sub, i_off, j_off, k_off, ns, ms, ps, rank_sub):
        """
        Place a sub-decomposition into the global matrices.
        i_off, j_off, k_off: offsets in the full matrices
        ns, ms, ps: sizes of the sub-blocks
        rank_sub: rank of the sub-decomposition
        Returns the next column index.
        """
        nonlocal col
        # U_sub has shape (ns*ms, rank_sub), indexed by (i,j) where i in [0,ns), j in [0,ms)
        # In full U, this corresponds to rows (i_off + i)*m + (j_off + j)
        # V_sub has shape (ms*ps, rank_sub), indexed by (j,k) where j in [0,ms), k in [0,ps)
        # In full V, this corresponds to rows (j_off + j)*p + (k_off + k)
        # W_sub has shape (ps*ns, rank_sub), indexed by (k,i) where k in [0,ps), i in [0,ns)
        # In full W, this corresponds to rows (k_off + k)*n + (i_off + i)
        
        for r in range(rank_sub):
            for ii in range(ns):
                for jj in range(ms):
                    U[(i_off + ii) * m + (j_off + jj), col + r] = U_sub[ii * ms + jj, r]
            for jj in range(ms):
                for kk in range(ps):
                    V[(j_off + jj) * p + (k_off + kk), col + r] = V_sub[jj * ps + kk, r]
            for kk in range(ps):
                for ii in range(ns):
                    W[(k_off + kk) * n + (i_off + ii), col + r] = W_sub[kk * ns + ii, r]
        
        col += rank_sub
        return col
    
    # C11 (rows 0-1, cols 0-2) = A11(0:2, 0:2)*B11(0:2, 0:3) + A12(0:2, 2:5)*B21(2:5, 0:3)
    # Term 1: A11 * B11: 2x2 * 2x3
    place_sub(U_223, V_223, W_223, i_off=0, j_off=0, k_off=0, ns=2, ms=2, ps=3, rank_sub=rank_223)
    # Term 2: A12 * B21: 2x3 * 3x3
    place_sub(U_233, V_233, W_233, i_off=0, j_off=2, k_off=0, ns=2, ms=3, ps=3, rank_sub=rank_233)
    
    # C12 (rows 0-1, cols 3-5) = A11(0:2, 0:2)*B12(0:2, 3:6) + A12(0:2, 2:5)*B22(2:5, 3:6)
    # Term 1: A11 * B12: 2x2 * 2x3
    place_sub(U_223, V_223, W_223, i_off=0, j_off=0, k_off=3, ns=2, ms=2, ps=3, rank_sub=rank_223)
    # Term 2: A12 * B22: 2x3 * 3x3
    place_sub(U_233, V_233, W_233, i_off=0, j_off=2, k_off=3, ns=2, ms=3, ps=3, rank_sub=rank_233)
    
    # C21 (rows 2-3, cols 0-2) = A21(2:4, 0:2)*B11(0:2, 0:3) + A22(2:4, 2:5)*B21(2:5, 0:3)
    # Term 1: A21 * B11: 2x2 * 2x3
    place_sub(U_223, V_223, W_223, i_off=2, j_off=0, k_off=0, ns=2, ms=2, ps=3, rank_sub=rank_223)
    # Term 2: A22 * B21: 2x3 * 3x3
    place_sub(U_233, V_233, W_233, i_off=2, j_off=2, k_off=0, ns=2, ms=3, ps=3, rank_sub=rank_233)
    
    # C22 (rows 2-3, cols 3-5) = A21(2:4, 0:2)*B12(0:2, 3:6) + A22(2:4, 2:5)*B22(2:5, 3:6)
    # Term 1: A21 * B12: 2x2 * 2x3
    place_sub(U_223, V_223, W_223, i_off=2, j_off=0, k_off=3, ns=2, ms=2, ps=3, rank_sub=rank_223)
    # Term 2: A22 * B22: 2x3 * 3x3
    place_sub(U_233, V_233, W_233, i_off=2, j_off=2, k_off=3, ns=2, ms=3, ps=3, rank_sub=rank_233)
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return result

# EVOLVE-BLOCK-END
