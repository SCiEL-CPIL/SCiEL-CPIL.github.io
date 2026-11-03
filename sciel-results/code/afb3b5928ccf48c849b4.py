"""Matrix multiplication 2x4x6 rank minimization via structured Strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, atol=1e-9):
    """Verify that the decomposition exactly reproduces T."""
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    T_check = np.zeros_like(T)
    for r in range(U.shape[1]):
        T_check += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_check, T, atol=atol)

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    Each term is a dict with keys 'u' (list of (idx, coeff)), 'v' (list of (idx, coeff)),
    and 'w' (list of (idx, coeff)).
    Indexing: A is 2x2, B is 2x2, C is 2x2.
    A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    """
    return [
        # m0: (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        # m1: (A10+A11)*B00 -> C10 += 1, C11 += -1
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        # m2: A00*(B01-B11) -> C01 += 1, C11 += 1
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        # m3: A11*(B10-B00) -> C00 += 1, C10 += 1
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        # m4: (A00+A01)*B11 -> C00 += -1, C01 += 1
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        # m5: (A10-A00)*(B00+B01) -> C11 += 1
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        # m6: (A01-A11)*(B10+B11) -> C00 += 1
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]},
    ]

def _construct_strassen_2x2x4(n, m, p):
    """
    Construct a rank-42 decomposition using Strassen on 2x2x2 blocks.
    
    Partition the 2x4 matrix A into two 2x2 blocks (columns 0-1 and 2-3).
    Partition the 4x6 matrix B into three 4x2 blocks (cols 0-1, 2-3, 4-5).
    For each pair (A_block, B_block), we compute a 2x2 * 4x2 product.
    """
    strassen_terms = _strassen_2x2x2_terms()
    
    rank = 6 * 7  # 42
    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12
    
    U = np.zeros((U_dim, rank), dtype=np.float64)
    V = np.zeros((V_dim, rank), dtype=np.float64)
    W = np.zeros((W_dim, rank), dtype=np.float64)
    
    col = 0
    for a_block in range(2):  # 0 or 1
        for b_block in range(3):  # 0, 1, or 2
            for term in strassen_terms:
                for u_idx, u_coeff in term['u']:
                    local_i = u_idx // 2
                    local_j = u_idx % 2
                    global_i = local_i
                    global_j = a_block * 2 + local_j
                    U[global_i * m + global_j, col] = u_coeff
                
                for v_idx, v_coeff in term['v']:
                    local_j = v_idx // 2
                    local_k = v_idx % 2
                    row_base = a_block * 2
                    global_row = row_base + local_j
                    global_col = b_block * 2 + local_k
                    V[global_row * p + global_col, col] = v_coeff
                
                for w_idx, w_coeff in term['w']:
                    local_i = w_idx // 2
                    local_k = w_idx % 2
                    global_i = local_i
                    global_k = b_block * 2 + local_k
                    W[global_k * n + global_i, col] = w_coeff
                
                col += 1
    
    return U, V, W, rank

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    U = np.zeros((U_dim, rank), dtype=np.float64)
    V = np.zeros((V_dim, rank), dtype=np.float64)
    W = np.zeros((W_dim, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def _construct_shared_strassen(n, m, p):
    """
    Construct a lower-rank decomposition by sharing Strassen computations.
    
    Key insight: C = A*B where A is 2x4, B is 4x6.
    Write A = [A1 | A2] (two 2x2 blocks) and B as a 4x6 matrix.
    
    C = A1 * B[0:2,:] + A2 * B[2:4,:]
    
    For the 2x2 * 2x6 product (A1 * B_top where B_top is 2x6), we can
    use a shared Strassen approach. The 2x2 * 2x6 = 2x6 product can be
    decomposed more efficiently than 3 separate 2x2*2x2 products.
    
    Specifically, for computing M (2x2) * N (2x6) = P (2x6):
    We can use 7 Strassen-like products that each multiply 2x2 by 2x2,
    but we arrange them so that the B-side linear combinations are shared
    across column blocks.
    
    Alternative: Use the fact that 2x2*2x6 can be done with rank 7*3=21
    naively, but potentially less with shared structure.
    
    Let me try: compute A1*B_top and A2*B_bot separately, each as 2x2*2x6.
    For 2x2*2x6, partition N into three 2x2 blocks: N = [N0, N1, N2].
    Then M*N = M*N0 + M*N1 + M*N2, each 2x2*2x2.
    
    Using Strassen for each: 7*3 = 21 per 2x2*2x6 product.
    Total: 2 * 21 = 42. Same as before.
    
    To do better, we need to share across the two 2x2*2x6 products.
    
    Let me try a different partitioning. Instead of splitting A into two 2x2
    column blocks, let's think of the whole thing differently.
    
    C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1; j=0..3; k=0..5.
    
    Let me try: define 7 "Strassen-like" linear combinations of A that are
    shared across all k, and 7 for B that depend on k-block.
    
    Actually, here's a better idea. Use the 4x4x4 rank-17 decomposition
    embedded in 4x6x4. 
    
    Embed 2x4*4x6 into 4x4*4x6 by setting A' = [A; A] (4x4) and 
    C' = [C; C] (4x6). Then C' = A'*B.
    
    The 4x4*4x6 = 4x6 product can be split into 4x4*4x2 + 4x4*4x2 + 4x4*4x2.
    Each 4x4*4x2 can potentially be done with rank ~17 (if we use the 
    4x4x4 rank-17 and extend).
    
    Actually, let me try a concrete approach: use 3 independent 4x4*4x2 
    products, each done with a rank-17-like decomposition adapted from 4x4x4.
    
    The known rank-17 decomposition for 4x4x4 uses 17 multiplications.
    For 4x4*4x2, we need 4x4*4x2 = 4x2, which is half the output.
    We can use the rank-17 4x4x4 decomposition and take only the relevant
    output components, but that still uses 17 multiplications per 4x4*4x2.
    
    Total: 3 * 17 = 51. That's worse than 42.
    
    Let me try yet another approach. The 2x4*4x6 tensor has a special 
    structure because n=2 is small. 
    
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    For each k, we're computing a 2x4 * 4x1 = 2x1 product.
    The rank of 2x4*4x1 is 4 (schoolbook) or possibly less.
    Actually, 2x4*4x1: C[0] = sum_j A[0,j]*B[j], C[1] = sum_j A[1,j]*B[j].
    This is just two dot products of length 4. Rank = 4 (each term A[i,j]*B[j] 
    for fixed i,j). Can we do better? No, because we need 8 independent 
    products (2 rows * 4 cols of A times 4 rows of B), but they share the B[j]
    factor. The tensor is sum_j (A[0,j]*e0 + A[1,j]*e1) ⊗ B[j] ⊗ (e0*e0 + e1*e1)
    Hmm, this is getting complex.
    
    Let me just try to find a rank-35 or lower decomposition through 
    systematic search of the structure.
    
    For now, return the 42-rank decomposition as baseline.
    """
    return _construct_strassen_2x2x4(n, m, p)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    Uses Strassen on 2x2x2 sub-blocks as the primary approach.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110

    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12

    # Build the target tensor for verification
    T = _build_tensor(n, m, p)

    # Try Strassen rank-42 decomposition
    U, V, W, rank = _construct_strassen_2x2x4(n, m, p)
    
    # Verify the Strassen decomposition
    if not _verify(U, V, W, T):
        # Fallback to schoolbook
        U, V, W, rank = _construct_schoolbook(n, m, p)

    # Final verification
    if not _verify(U, V, W, T):
        # Fallback to schoolbook
        U, V, W, rank = _construct_schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
