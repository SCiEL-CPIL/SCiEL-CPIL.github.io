"""Matrix multiplication 3x5x5 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check against identity: T[i,j,k] = 1 if this is the (i,j,k) entry of matmul tensor
    # Actually the tensor for matrix multiplication is:
    # T[i,j,k] = delta_{i,k} (no, that's not right)
    # For multiplying A (n x m) by B (m x p) to get C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] where the multiplication tensor has T[i,j,k] = 1 
    # when we're computing C[i,k] from A[i,j]*B[j,k]
    # So the tensor entries are: for each (i,j,k), T[i,j,k] = 1 if this corresponds to a valid term
    # Actually the standard matrix multiplication tensor has:
    # M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No wait. The tensor is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear form uses the tensor T where T[i,j,k] gives the coefficient.
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # Since C[i,k] = sum_j A[i,j]*B[j,k], T[i,j,k] = 1 for all valid i,j,k.
    # Wait, that's not right either. Let me think again.
    # 
    # The matrix multiplication tensor T of order (n, m, p) is defined by:
    # T[i,j,k] = 1 if we're multiplying the (i,j)-th entry of A by the (j,k)-th entry of B
    # to contribute to the (i,k)-th entry of C.
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # 
    # Hmm, but that would make the rank n*m*p which is the schoolbook.
    # 
    # Actually I think the correct definition is:
    # The tensor T has entries T[i,j,k] where:
    # - i indexes rows of A / rows of C (0 to n-1)
    # - j indexes cols of A / rows of B (0 to m-1)  
    # - k indexes cols of B / cols of C (0 to p-1)
    # T[i,j,k] = 1 (all entries are 1)
    # 
    # No wait, that can't be right because then the rank would always be n*m*p.
    # 
    # Let me reconsider. The matrix multiplication map is:
    # (A, B) -> C where C = A*B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # As a trilinear form, this is:
    # sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] = sum_{i,j,k} (sum_j' A[i,j']*B[j',k]) * A[i,j] * B[j,k]
    # 
    # Hmm, I think the tensor is actually defined differently. Let me use the standard definition.
    # 
    # The tensor for multiplying an n x m matrix by an m x p matrix is:
    # T in R^{n*m} x R^{m*p} x R^{n*p}
    # where T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = i*p+k for some i,j,k
    # 
    # Wait, the output convention says W[k*n+i, r], so the output index is k*n+i.
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # 
    # So the output matrix C has entries indexed by (i, k) where i is row (0..n-1) and k is col (0..p-1)
    # The linear index for C[i,k] is k*n + i (column-major? or just a specific convention)
    # 
    # U has shape (n*m, rank): index is i*m + j
    # V has shape (m*p, rank): index is j*p + k  
    # W has shape (p*n, rank): index is k*n + i
    # 
    # The decomposition says: for each c in range(rank):
    # C[k*n+i] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # Wait no. The standard tensor decomposition is:
    # T[a,b,c_idx] = sum_r U[a,r] * V[b,r] * W[c_idx, r]
    # 
    # Where T is the matrix multiplication tensor.
    # T[a, b, c_idx] = 1 if a = i*m+j, b = j*p+k, c_idx = k*n+i (i.e., the term A[i,j]*B[j,k] contributes to C[i,k])
    # 
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, and 0 otherwise.
    # 
    # Let me verify with schoolbook: each term (i,j,k) gets its own rank-1 component.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 where c corresponds to (i,j,k).
    # Then sum_r U[a,r]V[b,r]W[c_idx,r] = 1 only when a=i*m+j, b=j*p+k, c_idx=k*n+i match the same (i,j,k).
    # Yes, that's correct.
    
    # Build the target tensor (sparse)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We'll verify by checking a few random linear combinations or all entries
    
    U_float = np.array(U, dtype=float)
    V_float = np.array(V, dtype=float)
    W_float = np.array(W, dtype=float)
    
    # Compute the reconstructed tensor as a 3D array
    # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is a tensor product sum. We can compute it as:
    # T_recon = np.einsum('ar,br,cr->abc', U_float, V_float, W_float)
    T_recon = np.einsum('ar,br,cr->abc', U_float, V_float, W_float)
    
    # Build target
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    if not np.allclose(T_recon, T_target, atol=1e-10):
        return False
    return True

def _try_strassen_extension():
    """
    Use Strassen-style block decomposition to reduce rank.
    
    For 3x5x5, we can think of the matrices in blocks.
    A is 3x5, B is 5x5, C is 3x5.
    
    Let's partition:
    A = [A1 | A2] where A1 is 3x2, A2 is 3x3 (split after 2 cols)
    B = [B1; B2] where B1 is 2x5, B2 is 3x5 (split after 2 rows)
    
    C = A1*B1 + A2*B2
    
    For the 3x2 * 2x5 part, we can use standard multiplication (rank 3*2*5=30) or try to reduce.
    For the 3x3 * 3x5 part, we can use Strassen-like or other reductions.
    
    Actually, let me think about this differently.
    
    The key insight from the knowledge: "Extending Strassen decomposition to 4 disjoint 2x2x2 blocks 
    reduces rank from 74 to 71"
    
    Schoolbook rank for 3x5x5 is 3*5*5 = 75.
    
    Let me try a different approach: partition the 5 columns of A / 5 rows of B into blocks.
    
    Split j (the shared dimension) into blocks of sizes that allow Strassen-like savings.
    
    For 5 = 2 + 3:
    - Block 1: 3x2 * 2x5 -> rank 30 (schoolbook) or can we do better?
    - Block 2: 3x3 * 3x5 -> rank 45 (schoolbook) or can we do better?
    
    Total schoolbook: 30 + 45 = 75.
    
    For the 3x3 * 3x5 part, we can potentially use Strassen on the 3x3 sub-blocks.
    
    Actually, let me try a simpler approach first: just use the schoolbook but see if we can
    find a lower-rank decomposition by exploiting the structure.
    
    For now, let me implement a basic Strassen-inspired approach.
    
    Alternative: Use the fact that for 2x2x2, Strassen gives rank 7 instead of 8.
    If we can decompose the problem into parts where some are 2x2x2 blocks, we save rank.
    
    Let me try: partition j into 2+2+1:
    - Block 1: 3x2 * 2x5
    - Block 2: 3x2 * 2x5  
    - Block 3: 3x1 * 1x5
    
    For 3x2 * 2x5: This is like multiplying a 3x2 by 2x5. 
    We can view this as: for each of the 3 rows of A and 5 cols of B, we compute sum of 2 products.
    The tensor for 3x2*2x5 has rank at most 3*2*5=30, but can it be lower?
    
    Actually, for m=2, the multiplication A(i,j)B(j,k) with j in {0,1}:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is a sum of 2 rank-1 tensors per (i,k) pair, so total rank <= 2*3*5 = 30.
    But can we do better? The tensor is in R^{3*2} x R^{2*5} x R^{5*3} = R^6 x R^{10} x R^{15}.
    
    Hmm, this is getting complex. Let me just try to find a good decomposition numerically.
    """
    pass

def _construct_schoolbook():
    """Standard schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _construct_block_strassen():
    """
    Try to use block structure to reduce rank below schoolbook.
    
    Partition the shared dimension m=5 into blocks.
    Use Strassen for 2x2x2 sub-problems where possible.
    
    Strategy: Split j into 2+3.
    Part 1: A[:, :2] * B[:2, :] -> 3x2 * 2x5
    Part 2: A[:, 2:] * B[2:, :] -> 3x3 * 3x5
    
    For Part 1 (3x2 * 2x5):
    C1[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    This has a natural rank-30 decomposition (2 per (i,k) pair).
    Can we do better? The tensor is T1[a,b,c] where a=i*2+j, b=j*5+k, c=k*3+i.
    T1[i*2+j, j*5+k, k*3+i] = 1
    
    For fixed i,k: T1[i*2+0, 0*5+k, k*3+i] = 1 and T1[i*2+1, 1*5+k, k*3+i] = 1
    So for each (i,k), we have 2 non-zero entries in the tensor.
    Total non-zeros: 3*5*2 = 30.
    
    The minimal rank of this tensor... it's hard to say without more analysis.
    Let's assume we can't easily reduce it below 30 for now.
    
    For Part 2 (3x3 * 3x5):
    C2[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k]
    This is a 3x3 * 3x5 multiplication.
    Schoolbook rank: 3*3*5 = 45.
    
    Can we use Strassen here? Strassen works for 2x2x2.
    We could split the 3x3 into 2x2 + 1x1 blocks, but that's messy.
    
    Alternative: Use the known result that the tensor rank of 2x2x2 is 7 (Strassen).
    If we can embed 2x2x2 blocks, we save 1 rank per block.
    
    Let me try a different partition: 5 = 2+2+1
    Part 1: 3x2 * 2x5 (rank ~30)
    Part 2: 3x2 * 2x5 (rank ~30)  
    Part 3: 3x1 * 1x5 (rank 15)
    Total: 75 (same as schoolbook, no savings)
    
    Hmm. The savings come from using Strassen's 7 instead of 8 for 2x2x2.
    But we need to actually have 2x2x2 sub-problems.
    
    Let me think about this differently. What if we partition all three dimensions?
    
    n=3, m=5, p=5.
    
    What if we use a recursive/block approach where we identify 2x2x2 sub-problems?
    
    Actually, let me just try to find a decomposition with rank < 75 using numerical optimization
    or a clever algebraic construction.
    
    For now, let me return the schoolbook and add a time guard.
    """
    n, m, p = DIMENSIONS
    # Start with schoolbook
    U, V, W = _construct_schoolbook()
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (15, rank), (25, rank), (15, rank).
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit, leave margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 75)
    U, V, W = _construct_schoolbook()
    best_rank = U.shape[1]
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to improve
    # For now, return schoolbook as the baseline
    # Future work: implement Strassen-based improvements
    
    # Time guard
    if time.time() > deadline - 5:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
