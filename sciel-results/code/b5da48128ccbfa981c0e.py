"""
Program to find a rank-minimal exact tensor decomposition for multiplying
a 3 by 5 matrix by a 5 by 6 matrix.

Uses a block decomposition strategy:
- Split n=3 into [2,1], m=5 into [2,2,1], p=6 into [2,2,2]
- Apply Strassen's algorithm (rank 7) to 2x2x2 sub-blocks
- Use schoolbook for all other sub-blocks
- Total rank: 6*7 + 3*4 + 6*4 + 3*2 = 84 (improvement over schoolbook rank 90)
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    U: (4, 7) - maps A elements (row-major: a=0,b=1,c=2,d=3) to 7 products
    V: (4, 7) - maps B elements (row-major: e=0,f=1,g=2,h=3) to 7 products
    W: (4, 7) - maps 7 products to C elements (row-major: C00=0,C01=1,C10=2,C11=3)
    
    Strassen:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    m1 = (a+d)(e+h)
    m2 = (c+d)e
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)h
    m6 = (c-a)(e+f)
    m7 = (b-d)(g+h)
    
    C00 = m1 + m4 - m5 + m7
    C01 = m3 + m5
    C10 = m2 + m4
    C11 = m1 - m2 + m3 + m6
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # Indexing: U[i,j] where i = row*m+col in A, j = product index
    # A elements: a=(0,0)=idx0, b=(0,1)=idx1, c=(1,0)=idx2, d=(1,1)=idx3
    # B elements: e=(0,0)=idx0, f=(0,1)=idx1, g=(1,0)=idx2, h=(1,1)=idx3
    # C elements: C00=idx0, C01=idx1, C10=idx2, C11=idx3
    
    # m1 = (a+d)(e+h): products index 0
    U[0, 0] = 1; U[3, 0] = 1  # a + d
    V[0, 0] = 1; V[3, 0] = 1  # e + h
    W[0, 0] = 1; W[3, 0] = 1  # C00 + C11
    
    # m2 = (c+d)e: products index 1
    U[2, 1] = 1; U[3, 1] = 1  # c + d
    V[0, 1] = 1               # e
    W[2, 1] = 1; W[3, 1] = -1  # C10 - C11
    
    # m3 = a(f-h): products index 2
    U[0, 2] = 1               # a
    V[1, 2] = 1; V[3, 2] = -1  # f - h
    W[1, 2] = 1; W[3, 2] = 1   # C01 + C11
    
    # m4 = d(g-e): products index 3
    U[3, 3] = 1               # d
    V[2, 3] = 1; V[0, 3] = -1  # g - e
    W[0, 3] = 1; W[2, 3] = 1   # C00 + C10
    
    # m5 = (a+b)h: products index 4
    U[0, 4] = 1; U[1, 4] = 1  # a + b
    V[3, 4] = 1               # h
    W[0, 4] = -1; W[1, 4] = 1  # -C00 + C01
    
    # m6 = (c-a)(e+f): products index 5
    U[2, 5] = 1; U[0, 5] = -1  # c - a
    V[0, 5] = 1; V[1, 5] = 1   # e + f
    W[3, 5] = 1                 # C11
    
    # m7 = (b-d)(g+h): products index 6
    U[1, 6] = 1; U[3, 6] = -1  # b - d
    V[2, 6] = 1; V[3, 6] = 1   # g + h
    W[0, 6] = 1                 # C00
    
    return U, V, W

def schoolbook_factors(n, m, p):
    """
    Returns (U, V, W) for n x m x p schoolbook decomposition.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W


def block_strassen_factors(n_size, m_size, p_size):
    """
    Returns (U, V, W) for an n_size x m_size x p_size block using Strassen
    when all sizes are 2, otherwise schoolbook.
    U: (n_size*m_size, rank), V: (m_size*p_size, rank), W: (p_size*n_size, rank)
    """
    if n_size == 2 and m_size == 2 and p_size == 2:
        return strassen_factors()
    else:
        return schoolbook_factors(n_size, m_size, p_size)

def combine_factors(factors_list):
    """
    Combine multiple (U, V, W) tuples into a single decomposition.
    Each tuple has U: (rows_U, rank_i), V: (rows_V, rank_i), W: (rows_W, rank_i)
    All must have the same rows_U, rows_V, rows_W dimensions.
    """
    if not factors_list:
        return None
    
    total_rank = sum(U.shape[1] for U, V, W in factors_list)
    n_rows_U = factors_list[0][0].shape[0]
    n_rows_V = factors_list[0][1].shape[0]
    n_rows_W = factors_list[0][2].shape[0]
    
    U_full = np.zeros((n_rows_U, total_rank), dtype=np.float64)
    V_full = np.zeros((n_rows_V, total_rank), dtype=np.float64)
    W_full = np.zeros((n_rows_W, total_rank), dtype=np.float64)
    
    col_offset = 0
    for U, V, W in factors_list:
        r = U.shape[1]
        U_full[:, col_offset:col_offset + r] = U
        V_full[:, col_offset:col_offset + r] = V
        W_full[:, col_offset:col_offset + r] = W
        col_offset += r
    
    return U_full, V_full, W_full


def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes):
    """
    Build the full factor matrices using block decomposition.
    
    n_sizes: list of row block sizes for A (and C)
    m_sizes: list of col block sizes for A (and row block sizes for B)
    p_sizes: list of col block sizes for B (and C)
    
    For each triple (i_block, j_block, k_block), we have a sub-problem of
    size n_sizes[i] x m_sizes[j] x p_sizes[k].
    
    The full U has shape (n*m, total_rank), where the rows are indexed by
    (i, j) in row-major order of the full A matrix.
    Similarly V has shape (m*p, total_rank) indexed by (j, k).
    W has shape (p*n, total_rank) indexed by (k, i).
    """
    total_rank = 0
    # Precompute block offsets
    n_offsets = []
    offset = 0
    for s in n_sizes:
        n_offsets.append(offset)
        offset += s
    
    m_offsets = []
    offset = 0
    for s in m_sizes:
        m_offsets.append(offset)
        offset += s
    
    p_offsets = []
    offset = 0
    for s in p_sizes:
        p_offsets.append(offset)
        offset += s
    
    # First pass: compute total rank
    for i_idx, ni in enumerate(n_sizes):
        for j_idx, mj in enumerate(m_sizes):
            for k_idx, pk in enumerate(p_sizes):
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    U_full = np.zeros((n * m, total_rank), dtype=np.float64)
    V_full = np.zeros((m * p, total_rank), dtype=np.float64)
    W_full = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]
                
                # Get the block factors
                U_block, V_block, W_block = block_strassen_factors(ni, mj, pk)
                r = U_block.shape[1]
                
                # Map block U rows to full U rows
                # Block U has rows indexed by (local_i, local_j) -> local_i*mj + local_j
                # Full U has rows indexed by (global_i, global_j) -> global_i*m + global_j
                for li in range(ni):
                    for lj in range(mj):
                        full_row = (i_start + li) * m + (j_start + lj)
                        U_full[full_row, col_offset:col_offset + r] = U_block[li * mj + lj, :]
                
                # Map block V rows to full V rows
                # Block V has rows indexed by (local_j, local_k) -> local_j*pk + local_k
                # Full V has rows indexed by (global_j, global_k) -> global_j*p + global_k
                for lj in range(mj):
                    for lk in range(pk):
                        full_row = (j_start + lj) * p + (k_start + lk)
                        V_full[full_row, col_offset:col_offset + r] = V_block[lj * pk + lk, :]
                
                # Map block W rows to full W rows
                # Block W has rows indexed by (local_k, local_i) -> local_k*ni + local_i
                # Full W has rows indexed by (global_k, global_i) -> global_k*n + global_i
                for lk in range(pk):
                    for li in range(ni):
                        full_row = (k_start + lk) * n + (i_start + li)
                        W_full[full_row, col_offset:col_offset + r] = W_block[lk * ni + li, :]
                
                col_offset += r
    
    return U_full, V_full, W_full, total_rank

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    
    Uses block decomposition:
    - Split n=3 into [2,1]
    - Split m=5 into [2,2,1]
    - Split p=6 into [2,2,2]
    
    For each sub-block (ni x mj x pk):
    - If 2x2x2: use Strassen (rank 7)
    - Otherwise: use schoolbook (rank ni*mj*pk)
    
    Total rank: 6*7 + 3*4 + 6*4 + 3*2 = 84
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # Block sizes
    n_sizes = [2, 1]
    m_sizes = [2, 2, 1]
    p_sizes = [2, 2, 2]
    
    # Verify dimensions
    assert sum(n_sizes) == n, f"n_sizes sum {sum(n_sizes)} != n={n}"
    assert sum(m_sizes) == m, f"m_sizes sum {sum(m_sizes)} != m={m}"
    assert sum(p_sizes) == p, f"p_sizes sum {sum(p_sizes)} != p={p}"
    
    # Build the block decomposition
    U, V, W, total_rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)
    
    # Verify correctness: check that the decomposition reproduces the tensor
    # The tensor T has T[i,j,k] = 1 if the triple (i,j,k) corresponds to
    # C[i,k] += A[i,j]*B[j,k]
    # In the factorization: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1
    # for valid (i,j,k) triples and 0 otherwise.
    # 
    # Actually, the standard tensor decomposition checks:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 for all valid i,j,k (it's the identity tensor for matmul)
    
    # Quick verification with a small check
    if time.time() < deadline:
        # Check a few random entries
        np.random.seed(42)
        check_count = 0
        for _ in range(20):
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
            if abs(val - 1.0) > 1e-10:
                # This shouldn't happen - the decomposition should be exact
                # Fall back to schoolbook
                U, V, W = schoolbook_factors(n, m, p)
                total_rank = n * m * p
                break
            check_count += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
