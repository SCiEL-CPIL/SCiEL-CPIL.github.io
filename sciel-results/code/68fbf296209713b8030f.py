import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _verify(U, V, W, n, m, p, atol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U_3d = U.T.reshape(rank, n, m)
    V_3d = V.T.reshape(rank, m, p)
    W_3d = W.T.reshape(rank, p, n)
    
    T_decomp = np.einsum('rij,rjk,rki->ijk', U_3d, V_3d, W_3d)
    T_target = np.ones((n, m, p))
    
    if not np.allclose(T_decomp, T_target, atol=atol):
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
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

def _strassen_2x2(n, m, p):
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7) with the standard W convention W[k*n+i, r].
    
    For C = A @ B where A is 2x2, B is 2x2, C is 2x2:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # A indices: A[i,j] -> i*2+j (since m=2)
    # B indices: B[j,k] -> j*2+k (since p=2)
    # C indices: C[i,k] -> i*2+k, but W convention is W[k*n+i] where n=2
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    # C00 += M1, C11 += M1
    # C00 = C[0,0], W index = 0*2+0 = 0
    # C11 = C[1,1], W index = 1*2+1 = 3
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    # C10 += M2, C11 -= M2
    # C10 = C[1,0], W index = 0*2+1 = 1
    # C11 = C[1,1], W index = 1*2+1 = 3
    W[1, 1] = 1
    W[3, 1] = -1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1  # B11
    # C01 += M3, C11 += M3
    # C01 = C[0,1], W index = 1*2+0 = 2
    # C11 = C[1,1], W index = 1*2+1 = 3
    W[2, 2] = 1
    W[3, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1  # B00
    # C00 += M4, C10 += M4
    # C00 = C[0,0], W index = 0*2+0 = 0
    # C10 = C[1,0], W index = 0*2+1 = 1
    W[0, 3] = 1
    W[1, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    # C00 -= M5, C01 += M5
    # C00 = C[0,0], W index = 0*2+0 = 0
    # C01 = C[0,1], W index = 1*2+0 = 2
    W[0, 4] = -1
    W[2, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1  # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    # C11 += M6
    # C11 = C[1,1], W index = 1*2+1 = 3
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1  # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    # C00 += M7
    # C00 = C[0,0], W index = 0*2+0 = 0
    W[0, 6] = 1
    
    return U, V, W

def _block_strassen(n, m, p):
    """
    Build a block Strassen decomposition for 3x3 * 3x8.
    
    Strategy: Use Strassen on 2x2 blocks where possible, schoolbook for the rest.
    
    We can partition the 3x3 matrix A into blocks:
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    
    And B (3x8) into blocks:
    B = [[B11, B12, ..., B18],
         [B21, B22, ..., B28],
         [B31, B32, ..., B38]]
    
    C = A @ B (3x8)
    
    One approach: Use Strassen for the 2x2 subproblem involving rows 0,1 and columns 0,1 of A
    and corresponding rows of B, then schoolbook for the remaining parts.
    
    Actually, let me think of a cleaner block approach.
    
    Partition A into:
    - Top-left 2x2 block (rows 0-1, cols 0-1)
    - Top-right 2x1 block (rows 0-1, col 2)
    - Bottom-left 1x2 block (row 2, cols 0-1)
    - Bottom-right 1x1 block (row 2, col 2)
    
    Partition B into:
    - Top 2x8 block (rows 0-1)
    - Bottom 1x8 block (row 2)
    
    C = A @ B
    C[0:2, :] = A[0:2, 0:2] @ B[0:2, :] + A[0:2, 2:3] @ B[2:3, :]
    C[2:3, :] = A[2:3, 0:2] @ B[0:2, :] + A[2:3, 2:3] @ B[2:3, :]
    
    The 2x2 @ 2x8 part can use Strassen-like optimization.
    2x2 @ 2x8: We can use Strassen on the 2x2 block of A and treat the 2x8 B as two 2x4 blocks,
    or directly apply the 7-multiplication scheme.
    
    For 2x2 @ 2x8, we can use the same 7 products as Strassen but applied to each pair of columns.
    Actually, Strassen for 2x2 @ 2x8 requires 7 products where each product is a 1x8 @ 1x8? No.
    
    Let me think differently. For 2x2 @ 2x8:
    A = [[a, b], [c, d]]
    B = [b11...b18; b21...b28]
    C = [[a*b11 + b*b21, ..., a*b18 + b*b28],
         [c*b11 + d*b21, ..., c*b18 + d*b28]]
    
    Using Strassen:
    M1 = (a+d)*(b1+b2)  where b1, b2 are the two rows of B
    M2 = (c+d)*b1
    M3 = a*(b2-b1)
    M4 = d*(b1-b2)  -- wait, this isn't quite right for rectangular.
    
    Actually, for 2x2 @ 2x8, we can use the standard Strassen 7 products:
    M1 = (A00+A11) * (B00+B10)  -- but B is 2x8, so B00 is row 0, B10 is row 1
    Actually, let's just use the 7 Strassen products where each "product" is a dot product of a 1x2 vector with a 2x8 matrix, giving a 1x8 result.
    
    M1 = (A[0,0]+A[1,1]) * (B[0,:]+B[1,:])  -- 1x8
    M2 = (A[1,0]+A[1,1]) * B[0,:]  -- 1x8
    M3 = A[0,0] * (B[1,:]-B[0,:])  -- 1x8
    M4 = A[1,1] * (B[0,:]-B[1,:])  -- 1x8  -- wait, this gives negative of M3's second factor
    M5 = (A[0,0]+A[0,1]) * B[1,:]  -- 1x8
    M6 = (A[1,0]-A[0,0]) * (B[0,:]+B[1,:])  -- 1x8
    M7 = (A[0,1]-A[1,1]) * (B[0,:]+B[1,:])  -- 1x8
    
    C[0,:] = M1 + M4 - M5 + M7
    C[1,:] = M2 + M4 + M3  -- wait, let me re-derive
    
    Standard Strassen for 2x2:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    For 2x2 @ 2x8, the "C" is 2x8, so:
    C[0,:] = M1 + M4 - M5 + M7  (each term is 1x8)
    C[1,:] = M2 + M4  -- no, this doesn't work because the standard Strassen gives 2x2 output.
    
    Let me just directly compute the 7 products for 2x2 @ 2x8:
    The key insight is that we need 7 scalar multiplications per output column pair, but since B has 8 columns, we can process them in pairs or all at once.
    
    Actually, the simplest approach: use Strassen on 2x2 blocks of A and B where B is partitioned into 2x4 blocks.
    
    Let me just use a different strategy: 
    - Use Strassen (rank 7) for the 2x2 @ 2x2 subproblem
    - Use schoolbook for the remaining parts
    
    For 3x3 @ 3x8:
    Partition A into 2x2 (top-left), 2x1 (top-right), 1x2 (bottom-left), 1x1 (bottom-right)
    Partition B into 2x8 (top), 1x8 (bottom)
    
    C[0:2, :] = A[0:2, 0:2] @ B[0:2, :] + A[0:2, 2:3] @ B[2:3, :]
    C[2:3, :] = A[2:3, 0:2] @ B[0:2, :] + A[2:3, 2:3] @ B[2:3, :]
    
    For A[0:2, 0:2] @ B[0:2, :]: This is 2x2 @ 2x8.
    We can use Strassen-style: 7 products, each involving a 1x2 vector and a 2x8 matrix.
    
    Let me implement 2x2 @ 2x8 with 7 products:
    A = [[a, b], [c, d]]
    B = [row0; row1] where row0, row1 are 1x8
    
    M1 = (a+d) * (row0+row1)
    M2 = (c+d) * row0
    M3 = a * (row1-row0)
    M4 = d * (row0-row1)
    M5 = (a+b) * row1
    M6 = (c-a) * (row0+row1)
    M7 = (b-d) * (row0+row1)
    
    C[0,:] = M1 + M4 - M5 + M7
    C[1,:] = M2 + M4 + M3
    
    Let me verify:
    C[0,:] should be a*row0 + b*row1
    M1 + M4 - M5 + M7 = (a+d)(row0+row1) + d(row0-row1) - (a+b)row1 + (b-d)(row0+row1)
    = a*row0 + a*row1 + d*row0 + d*row1 + d*row0 - d*row1 - a*row1 - b*row1 + b*row0 + b*row1 - d*row0 - d*row1
    = a*row0 + (a+a-a-a)*row1 + d*row0 + (d+d-d-d)*row1 + b*row0 + (-b+b)*row1 - d*row1 + d*row1
    Wait, let me be more careful:
    = a*row0 + a*row1 + d*row0 + d*row1 + d*row0 - d*row1 - a*row1 - b*row1 + b*row0 + b*row1 - d*row0 - d*row1
    Coefficients of row0: a + d + d + b - d = a + d + b
    Coefficients of row1: a + d - d - a - b + b - d = -d
    That gives (a+d+b)*row0 - d*row1, which is not a*row0 + b*row1.
    
    I'm making errors. Let me just use the standard Strassen formulas properly.
    
    Standard Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    For 2x2 @ 2x8, B is 2x8, so B00 and B01 are not 1x1 but rather... this doesn't directly apply.
    
    I think the cleanest approach is to just use schoolbook for now and try a random search for improvement.
    """
    return _schoolbook(n, m, p)

def _random_search(n, m, p, target_rank, max_iter=100, deadline=None):
    """
    Try to find a decomposition of rank target_rank using random initialization and local search.
    """
    if deadline is None:
        deadline = time.time() + 10
    
    rng = np.random.RandomState(42)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Sparse random initialization
        nnz = 3
        for r in range(target_rank):
            u_idx = rng.choice(n * m, size=min(nnz, n * m), replace=False)
            v_idx = rng.choice(m * p, size=min(nnz, m * p), replace=False)
            w_idx = rng.choice(p * n, size=min(nnz, p * n), replace=False)
            
            U[u_idx, r] = rng.uniform(-1, 1, size=len(u_idx))
            V[v_idx, r] = rng.uniform(-1, 1, size=len(v_idx))
            W[w_idx, r] = rng.uniform(-1, 1, size=len(w_idx))
        
        if _verify(U, V, W, n, m, p, atol=1e-6):
            return U, V, W, target_rank
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = U_school.shape[1]
    
    # Try to improve using random search
    # The theoretical lower bound is at least max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But known results suggest higher. Let's try ranks from 50 down to 24.
    
    # First, try a few specific known decompositions if we can construct them.
    # For now, just do random search.
    
    search_deadline = deadline - 20  # Leave 20s for final verification and return
    
    for target_rank in range(60, 23, -1):
        if time.time() > search_deadline:
            break
        
        result = _random_search(n, m, p, target_rank, max_iter=5, deadline=search_deadline)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = rank
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
