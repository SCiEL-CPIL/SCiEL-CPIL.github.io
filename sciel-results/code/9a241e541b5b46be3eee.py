"""Matrix multiplication 2x2 x 2x16 via Strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the Strassen decomposition of 2x2 matrix multiplication.
    The 7 rank-1 terms have coefficients in {-1, 0, 1}.
    
    For C = A * B where A is 2x2 and B is 2x2:
    A = [[a, b], [c, d]]
    B = [[e, f], [g, h]]
    C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    Strassen's 7 products:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
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


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification of the tensor decomposition."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(result, 1.0, atol=tol)


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    
    Uses Strassen's rank-7 decomposition for the 2x2 core, applied
    independently to each of the 8 column-pairs of the 2x16 matrix.
    Total rank = 7 * 8 = 56.
    
    The 2x16 matrix B has columns indexed 0..15. We group them into
    8 pairs: (col 2t, col 2t+1) for t=0..7.
    
    For each pair t, we treat B[:, 2t:2t+2] as a 2x2 matrix and apply
    Strassen. The output C is 2x16, where C[:, 2t:2t+2] = A * B[:, 2t:2t+2].
    
    U shape: (n*m, rank) = (4, 56)
    V shape: (m*p, rank) = (32, 56)
    W shape: (p*n, rank) = (32, 56)
    
    Indexing conventions:
    - U row index: i*m + j where (i,j) indexes A (i=0,1; j=0,1)
    - V row index: j*p + k where (j,k) indexes B (j=0,1; k=0..15)
    - W row index: k*n + i where (k,i) indexes C output (k=0..15; i=0,1)
      Per the trace convention W[k*n+i, r]
    """
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
            # U: same for all pairs, just shifted in column
            U[:, col_offset + r] = U_core[:, r]
            
            # V: B row index is j*p + k. For pair t, columns are 2t and 2t+1.
            # V_core rows: 0=(j=0,local_k=0), 1=(j=0,local_k=1), 2=(j=1,local_k=0), 3=(j=1,local_k=1)
            # Full V row: j*p + (2t + local_k)
            V[0 * p + 2 * t, col_offset + r] = V_core[0, r]
            V[0 * p + 2 * t + 1, col_offset + r] = V_core[1, r]
            V[1 * p + 2 * t, col_offset + r] = V_core[2, r]
            V[1 * p + 2 * t + 1, col_offset + r] = V_core[3, r]
            
            # W: output C row index is k*n + i. For pair t, output columns are 2t and 2t+1.
            # W_core rows: 0=C(i=0,local_k=0), 1=C(i=0,local_k=1), 2=C(i=1,local_k=0), 3=C(i=1,local_k=1)
            # Full W row: (2t + local_k)*n + i
            W[(2 * t) * n + 0, col_offset + r] = W_core[0, r]
            W[(2 * t + 1) * n + 0, col_offset + r] = W_core[1, r]
            W[(2 * t) * n + 1, col_offset + r] = W_core[2, r]
            W[(2 * t + 1) * n + 1, col_offset + r] = W_core[3, r]
    
    # Verify the decomposition
    if not _verify_decomposition_fast(U, V, W, n, m, p):
        # Fallback: schoolbook decomposition (rank = n*m*p = 64)
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=np.int64)
        V_fb = np.zeros((m * p, rank_fb), dtype=np.int64)
        W_fb = np.zeros((p * n, rank_fb), dtype=np.int64)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, column] = 1
                    V_fb[j * p + k, column] = 1
                    W_fb[k * n + i, column] = 1
                    column += 1
        U, V, W = U_fb, V_fb, W_fb
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
