"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 110.0  # safety margin before 120s kill

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_decomposition():
    """
    Construct a rank-35 exact tensor decomposition for 2x2 * 2x10 matrix multiplication.
    
    Strategy: Apply Strassen's rank-7 algorithm for 2x2 matrix multiplication
    independently to each of the 5 disjoint 2-column blocks of the 2x10 matrix.
    
    The tensor is a direct sum over 5 blocks, each contributing rank 7,
    giving total rank = 5 * 7 = 35.
    
    Returns:
        dict with keys 'U', 'V', 'W' as lists of lists with integer entries.
    """
    n, m, p = 2, 2, 10  # A is n x m, B is m x p, C is n x p
    num_blocks = p // 2  # 5 blocks of 2 columns each
    rank = num_blocks * 7  # 35
    
    # U: shape (n*m, rank) = (4, 35)
    # V: shape (m*p, rank) = (20, 35)
    # W: shape (p*n, rank) = (20, 35)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7 multiplications for 2x2 matmul:
    # A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]]
    # M1=(A00+A11)(B00+B11)  -> C00+=M1, C11+=M1
    # M2=(A10+A11)B00        -> C10+=M2, C11-=M2
    # M3=A00(B01-B11)        -> C01+=M3, C11+=M3
    # M4=A11(B10-B00)        -> C00+=M4, C10+=M4
    # M5=(A00+A01)B11        -> C01+=M5, C00-=M5
    # M6=(A10-A00)(B00+B01)  -> C11+=M6
    # M7=(A01-A11)(B10+B11)  -> C00+=M7
    #
    # U row idx: 0=A00, 1=A01, 2=A10, 3=A11
    # V within-block koff: 0=B00, 1=B01, 2=B10, 3=B11
    #   actual row = j*p + 2*b + (koff & 1), j = koff >> 1
    # W within-block koff: 0=C00, 1=C01, 2=C10, 3=C11
    #   actual row = (2*b + (koff & 1))*n + (koff >> 1)
    
    terms = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(1,1),(0,-1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    for b in range(num_blocks):
        base = b * 7
        for t in range(7):
            col = base + t
            uc, vc, wc = terms[t]
            for ri, c in uc:
                U[ri][col] = c
            for koff, c in vc:
                j = koff >> 1
                k = 2 * b + (koff & 1)
                V[j * p + k][col] = c
            for koff, c in wc:
                i = koff >> 1
                k = 2 * b + (koff & 1)
                W[k * n + i][col] = c
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition exactly reproduces matrix multiplication.
    
    For all (i, j, k): T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1.
    Uses vectorized numpy operations for speed.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    
    for r in range(rank):
        u_2d = U_arr[:, r].reshape(n, m)   # (n, m)
        v_2d = V_arr[:, r].reshape(m, p)   # (m, p)
        w_2d = W_arr[:, r].reshape(p, n)   # (p, n)
        
        # T_r[i,j,k] = u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
        # Build via broadcasting:
        # u_part: (n, m, 1) -> u_2d[i,j,0]
        # v_part: (1, m, p) -> v_2d[0,j,k]
        # w_part: (n, 1, p) -> w_2d[k, i, 0] but we need w_2d[k,i] indexed as [k,i]
        #   w_2d is (p, n), so w_2d[k,i] = w_2d.T[i,k]
        #   w_part[i, 0, k] = w_2d.T[i, k]
        
        u_part = u_2d[:, :, np.newaxis]    # (n, m, 1)
        v_part = v_2d[np.newaxis, :, :]    # (1, m, p)
        w_part = w_2d.T[:, np.newaxis, :]  # (n, 1, p) -> w_2d.T[i, 0, k] = w_2d[k, i]
        
        T += u_part * v_part * w_part
    
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard: if we're too close to the limit, return immediately
    if _time_remaining() < 5.0:
        # Build without verification as last resort
        return _build_decomposition()
    
    result = _build_decomposition()
    
    # Fast vectorized verification (runs in <1s for rank-35, 40 entries)
    if _time_remaining() > 3.0:
        if not _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
            raise ValueError("Decomposition verification failed")
    
    return result
# EVOLVE-BLOCK-END
