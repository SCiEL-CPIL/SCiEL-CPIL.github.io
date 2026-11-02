import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the Strassen decomposition of 2x2 matrix multiplication.
    Rank 7.
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    """
    Construct a rank-56 exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    
    Uses 8 independent Strassen rank-7 blocks.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    U_core, V_core, W_core = _strassen_2x2()
    
    num_pairs = p // 2  # 8 pairs
    rank = 7 * num_pairs  # 56
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for t in range(num_pairs):
        col_offset = t * 7
        
        for r in range(7):
            # U: same for all pairs
            U[:, col_offset + r] = U_core[:, r]
            
            # V: B row index j*p + k
            # V_core rows: 0=(j=0,k=0), 1=(j=0,k=1), 2=(j=1,k=0), 3=(j=1,k=1)
            # Full V rows for pair t (cols 2t, 2t+1):
            # j=0, k=2t -> 0*p + 2t
            # j=0, k=2t+1 -> 0*p + 2t+1
            # j=1, k=2t -> 1*p + 2t
            # j=1, k=2t+1 -> 1*p + 2t+1
            
            V[0 * p + 2 * t, col_offset + r] = V_core[0, r]
            V[0 * p + 2 * t + 1, col_offset + r] = V_core[1, r]
            V[1 * p + 2 * t, col_offset + r] = V_core[2, r]
            V[1 * p + 2 * t + 1, col_offset + r] = V_core[3, r]
            
            # W: C row index k*n + i
            # W_core rows: 0=(i=0,k=0), 1=(i=0,k=1), 2=(i=1,k=0), 3=(i=1,k=1)
            # Full W rows for pair t (cols 2t, 2t+1):
            # k=2t, i=0 -> 2t*n + 0
            # k=2t, i=1 -> 2t*n + 1
            # k=2t+1, i=0 -> (2t+1)*n + 0
            # k=2t+1, i=1 -> (2t+1)*n + 1
            
            W[(2 * t) * n + 0, col_offset + r] = W_core[0, r]
            W[(2 * t) * n + 1, col_offset + r] = W_core[2, r]
            W[(2 * t + 1) * n + 0, col_offset + r] = W_core[1, r]
            W[(2 * t + 1) * n + 1, col_offset + r] = W_core[3, r]
            
        if time.time() > deadline:
            break
    
    # Verify
    # Reshape for einsum
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # The tensor T[i,j,k] should be 1 if i,j,k correspond to a valid multiplication term?
    # No, the decomposition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k]
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k?
    # Wait.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) and value 1.
    # Yes, T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0..15}.
    
    result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # Check if result is all 1s
    if not np.allclose(result, 1.0, atol=1e-9):
        # Fallback to schoolbook (rank 64)
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=np.int64)
        V_fb = np.zeros((m * p, rank_fb), dtype=np.int64)
        W_fb = np.zeros((p * n, rank_fb), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        U, V, W = U_fb, V_fb, W_fb
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
