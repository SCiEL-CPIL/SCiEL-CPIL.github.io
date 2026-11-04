"""Matrix multiplication 3x5x5 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _strassen_2x2():
    """Return Strassen 2x2 matrix multiplication tensor (rank 7)."""
    n, m, p = 2, 2, 2
    U = np.zeros((n * m, 7))
    V = np.zeros((m * p, 7))
    W = np.zeros((p * n, 7))
    
    # m1 = (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # m3 = a00*(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # m4 = a11*(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # m5 = (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    W[0, 4] = -1
    
    # m6 = (a10-a00)*(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # m7 = (a01-a11)*(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _verify(U, V, W, n, m, p, atol=1e-10):
    """Check if the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    AB_direct = A @ B

    U_reshaped = U.T.reshape(rank, n, m)
    V_reshaped = V.T.reshape(rank, m, p)
    W_reshaped_temp = W.T.reshape(rank, p, n)
    W_reshaped = W_reshaped_temp.transpose(0, 2, 1)

    U_A = np.einsum('rij,ij->r', U_reshaped, A)
    V_B = np.einsum('rjk,jk->r', V_reshaped, B)
    AB_decomp = np.einsum('r,r,rik->ik', U_A, V_B, W_reshaped)

    return np.allclose(AB_direct, AB_decomp, atol=atol)


def _block_multiply(n1, m1, p1, n2, m2, p2, use_strassen=False):
    """
    Multiply block1 (n1×m1) by block2 (m2×p2) where m1==m2.
    Returns U, V, W for this block multiplication.
    If use_strassen and all dims are 2, use Strassen (rank 7).
    Otherwise use schoolbook.
    """
    if use_strassen and n1 == 2 and m1 == 2 and p1 == 2:
        return _strassen_2x2()
    else:
        return _schoolbook(n1, m1, p1)


def _combine_decompositions(decomps):
    """
    Combine multiple independent tensor decompositions.
    Each decomp is (U, V, W) for a sub-problem.
    The sub-problems must cover disjoint (i,k) output positions.
    """
    total_rank = sum(d[0].shape[1] for d in decomps)
    
    # We need to know the full dimensions
    # This is tricky - let's handle it differently
    raise NotImplementedError


def _construct_block_decomposition(n, m, p):
    """
    Construct a block decomposition for n×m × m×p matrix multiplication.
    
    Strategy: Partition the m dimension (shared between A and B) into blocks.
    For 3×5×5: partition m=5 into [2,2,1] or [2,1,2] etc.
    
    A (3×5) = [A1 | A2 | A3] where A1 is 3×2, A2 is 3×2, A3 is 3×1
    B (5×5) = [B1; B2; B3] where B1 is 2×5, B2 is 2×5, B3 is 1×5
    
    AB = A1*B1 + A2*B2 + A3*B3
    
    For each block product, we can use Strassen if the block is 2×2×2,
    or schoolbook otherwise.
    """
    # Try different partitions of m=5
    # Partition options: [2,2,1], [2,1,2], [1,2,2], [3,2], [2,3], [1,1,1,1,1], etc.
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = n * m * p  # schoolbook
    
    # Define partitions of m
    partitions = [
        [2, 2, 1],
        [2, 1, 2],
        [1, 2, 2],
        [3, 2],
        [2, 3],
        [1, 4],
        [4, 1],
        [5],
    ]
    
    for part in partitions:
        if sum(part) != m:
            continue
        
        # Build the full decomposition
        U_cols = []
        V_cols = []
        W_cols = []
        
        # Track which rows of A and B each block uses
        a_row_offset = 0  # This is for the j-index in A
        b_row_offset = 0  # This is for the j-index in B
        
        for block_size in part:
            # Block: A[:, a_row_offset:a_row_offset+block_size] @ B[b_row_offset:b_row_offset+block_size, :]
            # A_block is n × block_size
            # B_block is block_size × p
            
            # Use Strassen only if n==2, block_size==2, p==2
            use_strassen = (n == 2 and block_size == 2 and p == 2)
            
            U_blk, V_blk, W_blk = _block_multiply(n, block_size, p, n, block_size, p, use_strassen)
            
            # Now we need to embed this into the full problem
            # U_blk has shape (n*block_size, rank_blk)
            # We need to map it to the full U of shape (n*m, rank_total)
            
            U_full = np.zeros((n * m, U_blk.shape[1]))
            V_full = np.zeros((m * p, V_blk.shape[1]))
            W_full = np.zeros((p * n, W_blk.shape[1]))
            
            # Map U: U_blk[i*block_size + j, r] -> U_full[i*m + (a_row_offset + j), r]
            for i in range(n):
                for j in range(block_size):
                    for r in range(U_blk.shape[1]):
                        U_full[i * m + (a_row_offset + j), r] = U_blk[i * block_size + j, r]
            
            # Map V: V_blk[j*p + k, r] -> V_full[(b_row_offset + j)*p + k, r]
            for j in range(block_size):
                for k in range(p):
                    for r in range(V_blk.shape[1]):
                        V_full[(b_row_offset + j) * p + k, r] = V_blk[j * p + k, r]
            
            # Map W: W_blk[k*n + i, r] -> W_full[k*n + i, r] (no change needed)
            W_full = W_blk.copy()
            
            U_cols.append(U_full)
            V_cols.append(V_full)
            W_cols.append(W_full)
            
            a_row_offset += block_size
            b_row_offset += block_size
        
        # Combine all blocks
        U_combined = np.hstack(U_cols)
        V_combined = np.hstack(V_cols)
        W_combined = np.hstack(W_cols)
        
        rank = U_combined.shape[1]
        
        if rank < best_rank:
            if _verify(U_combined, V_combined, W_combined, n, m, p):
                best_rank = rank
                best_U = U_combined
                best_V = V_combined
                best_W = W_combined
    
    if best_U is not None:
        return best_U, best_V, best_W
    else:
        return _schoolbook(n, m, p)


def _construct_tensor_product_decomposition(n, m, p):
    """
    Use the tensor product structure: M_{n,m,p} = M_{n,1,1} ⊗ M_{1,m,p}
    
    For n=3, m=5, p=5:
    M_{3,5,5} can be viewed as 3 copies of M_{1,5,5} sharing the B factor.
    
    Actually, the correct decomposition is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    This is NOT simply a tensor product. Let me think again.
    
    The tensor for M_{n,m,p} has entries T[i,j,k] = 1 (in the appropriate basis).
    A rank-1 term is u ⊗ v ⊗ w where:
    - u is in R^{n×m}
    - v is in R^{m×p}  
    - w is in R^{n×p} (with trace convention)
    
    The condition is: sum_r u_r[i,j] * v_r[j',k] * w_r[k',i'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    For the tensor product approach:
    If we write n = n1 * n2, m = m1 * m2, p = p1 * p2, then
    M_{n,m,p} = M_{n1,m1,p1} ⊗ M_{n2,m2,p2}
    and rank(M_{n,m,p}) ≤ rank(M_{n1,m1,p1}) * rank(M_{n2,m2,p2})
    
    For 3×5×5, we can try:
    3 = 3*1, 5 = 5*1, 5 = 5*1: M_{3,1,1} ⊗ M_{1,5,5}
    rank(M_{3,1,1}) = 3 (schoolbook)
    rank(M_{1,5,5}) = ? (need to find)
    
    Or: 3 = 1*3, 5 = 1*5, 5 = 1*5: M_{1,1,1} ⊗ M_{3,5,5} (trivial)
    
    The key is finding a good decomposition of M_{1,5,5}.
    """
    # For now, fall back to block decomposition
    return _construct_block_decomposition(n, m, p)


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition (rank 75)
    U, V, W = _schoolbook(n, m, p)
    
    if not _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try block decomposition
    if time.time() < deadline:
        U_block, V_block, W_block = _construct_block_decomposition(n, m, p)
        if _verify(U_block, V_block, W_block, n, m, p):
            rank_block = U_block.shape[1]
            if rank_block < best_rank:
                best_rank = rank_block
                best_U, best_V, best_W = U_block, V_block, W_block
    
    # Try tensor product decomposition
    if time.time() < deadline - 10:
        U_tp, V_tp, W_tp = _construct_tensor_product_decomposition(n, m, p)
        if _verify(U_tp, V_tp, W_tp, n, m, p):
            rank_tp = U_tp.shape[1]
            if rank_tp < best_rank:
                best_rank = rank_tp
                best_U, best_V, best_W = U_tp, V_tp, W_tp
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
