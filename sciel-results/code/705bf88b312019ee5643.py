"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 120.0

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
    
    For all (i, j, k): T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    should equal C[i,k] where C = A * B, i.e., C[i,k] = sum_j A[i,j]*B[j,k].
    
    The tensor T[i,j,k] should be 1 if the coefficient of A[i,j]*B[j,k] in C
    is 1, which is always the case for standard matrix multiplication.
    So T[i,j,k] should equal 1 for all i,j,k.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute tensor: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # U row index: i*n + ... wait, let me re-check the indexing.
    # U has shape (n*m, rank). Row index for A[i,j] is i*m + j = i*2 + j.
    # V has shape (m*p, rank). Row index for B[j,k] is j*p + k.
    # W has shape (p*n, rank). Row index for C[k,i] is k*n + i.
    
    # We need to verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # for all i in [0,n), j in [0,m), k in [0,p).
    
    rank = U_arr.shape[1]
    
    # Compute outer product sum efficiently
    # For each r, we have U[:,r] (n*m,), V[:,r] (m*p,), W[:,r] (p*n,)
    # T = sum_r outer(outer(U_r, V_r), W_r) reshaped appropriately
    
    # Let's do it in chunks to avoid huge intermediate arrays
    T = np.zeros((n, m, p), dtype=float)
    
    for r in range(rank):
        u_r = U_arr[:, r]  # (n*m,)
        v_r = V_arr[:, r]  # (m*p,)
        w_r = W_arr[:, r]  # (p*n,)
        
        # Reshape u_r to (n, m), v_r to (m, p), w_r to (p, n)
        u_2d = u_r.reshape(n, m)
        v_2d = v_r.reshape(m, p)
        w_2d = w_r.reshape(p, n)
        
        # For each (i,j,k): u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
        # = sum over i,j,k of u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
        
        # Compute product for this r
        # temp[i,j,k] = u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
        
        # u_2d[i,j] * v_2d[j,k] gives (n,p) matrix: (u_2d @ v_2d)[i,k]
        # Then multiply by w_2d[k,i] which is (p,n)
        
        uv = u_2d @ v_2d  # (n, p)
        # Now T_r[i,k] = sum_j uv[i,j] * ... no wait
        # We need sum_j u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
        # = sum_j (u_2d[i,j] * v_2d[j,k]) * w_2d[k,i]
        # The sum over j of u_2d[i,j]*v_2d[j,k] is (u_2d @ v_2d)[i,k]
        # So T_r[i,k] = (u_2d @ v_2d)[i,k] * w_2d[k,i]... 
        # No! w_2d[k,i] depends on k and i, so:
        # T_r[i,k] = sum_j u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
        # This is not a simple matrix product because w_2d[k,i] is not separable.
        
        # Let me just do it element-wise for verification (small dimensions)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
    
    # T[i,j,k] should be 1 for all i,j,k
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=1e-10)

def construct():
    result = _build_decomposition()
    
    # Optional verification (disabled for speed in production)
    # n, m, p = DIMENSIONS
    # if not _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
    #     raise ValueError("Decomposition verification failed")
    
    return result
# EVOLVE-BLOCK-END
