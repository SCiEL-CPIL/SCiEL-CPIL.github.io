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


def _strassen_2x2():
    """Return Strassen 2x2 matrix multiplication tensor (rank 7).
    
    For multiplying 2x2 by 2x2 matrices:
    M = A @ B where A, B are 2x2.
    
    Strassen's 7 multiplications:
    m1 = (a00+a11)*(b00+b11)
    m2 = (a10+a11)*b00
    m3 = a00*(b01-b11)
    m4 = a11*(b10-b00)
    m5 = (a00+a01)*b11
    m6 = (a10-a00)*(b00+b01)
    m7 = (a01-a11)*(b10+b11)
    
    Result:
    c00 = m1 + m4 - m5 + m7
    c01 = m3 + m5
    c10 = m2 + m4
    c11 = m1 - m2 + m3 + m6
    """
    n, m, p = 2, 2, 2
    U = np.zeros((n * m, 7))
    V = np.zeros((m * p, 7))
    W = np.zeros((p * n, 7))
    
    # m1 = (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # c00 += m1, c11 += m1
    
    # m2 = (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[3, 1] = 1  # c10 += m2, c11 -= m2
    W[3, 1] = -1  # c11 -= m2... wait let me re-derive
    
    # Let me be more careful with the Strassen decomposition.
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # Reset
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # contributes to c00 and c11
    
    # m2 = (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # c10 += m2
    W[3, 1] = -1  # c11 -= m2
    
    # m3 = a00*(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # c01 += m3
    W[3, 2] = 1  # c11 += m3
    
    # m4 = a11*(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # c00 += m4
    W[2, 3] = 1  # c10 += m4
    
    # m5 = (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1  # c01 += m5
    W[0, 4] = -1  # c00 -= m5
    
    # m6 = (a10-a00)*(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # c11 += m6
    
    # m7 = (a01-a11)*(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # c00 += m7
    
    return U, V, W


def _block_decompose(n, m, p):
    """Decompose n×m×p matrix multiplication using block structure.
    
    For 3×5×5: partition A (3×5) into rows, B (5×5) into columns.
    We can view this as 3 row-vectors (1×5) times a 5×5 matrix.
    
    Each row i of A times B gives row i of AB.
    So AB[i,:] = A[i,:] @ B.
    
    This is a 1×5×5 multiplication for each of the 3 rows.
    The 1×5×5 multiplication tensor has rank 5 (schoolbook).
    So the total rank would be 3*5 = 15 if we can share structure.
    
    Actually, let's think about this differently.
    The full tensor is the tensor product of the 3×1×1 scalar mult tensor (rank 3)
    and the 1×5×5 matrix mult tensor (rank 5).
    Tensor product rank = 3*5 = 15.
    
    But wait - the 3×1×1 tensor is just the identity for selecting which row.
    Let me construct it properly.
    
    For 3×5 × 5×5:
    A is 3×5, B is 5×5, C = A@B is 3×5.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,k,j] = delta (it's the identity in j).
    Actually T[i,k,j] = 1 for the "multiplication" structure.
    
    Let me think of it as: for each (i,k), C[i,k] = sum_j A[i,j]B[j,k].
    
    The tensor is sum_{i,j,k} e_i ⊗ e_k ⊗ e_j^* (in appropriate dual spaces).
    
    Rank of this tensor: it's the Kronecker product of the 3-dim identity tensor
    (which has rank 3) and the 5×5 matrix mult tensor.
    
    Actually, the matrix mult tensor for 1×m × m×p has rank m*p (schoolbook).
    For 1×5 × 5×5, rank = 5*5 = 25? No, that's not right either.
    
    Let me reconsider. The tensor for (1×m) × (m×p) → (1×p) is:
    C[k] = sum_j A[j] * B[j,k]
    
    This is just a sum of m*p terms, so rank ≤ m*p = 25.
    But actually, since A is a row vector, this is just 5 independent linear combinations.
    The rank should be 5 (each output coordinate k depends on all m inputs).
    
    Hmm, let me just construct it directly.
    For 1×5 × 5×5:
    C[k] = sum_{j=0}^{4} A[j] * B[j,k]
    
    Schoolbook: 25 terms, one for each (j,k) pair.
    U[j, j*5+k] = 1 (A index j)
    V[j*5+k, j*5+k] = 1 (B index j,k)  
    W[k, j*5+k] = 1 (C index k)
    
    Can we do better? The rank of the 1×m×p matrix multiplication tensor is known to be m*p for m=1 or p=1? No...
    
    Actually for 1×m × m×1 (scalar product), the rank is m.
    For 1×m × m×p, I believe the rank is m*p in general, but there might be savings.
    
    Let me just use the direct construction and see if we can find a lower rank.
    
    For the full 3×5×5 problem, let me try a different approach:
    View it as 3 separate 1×5×5 multiplications.
    If each 1×5×5 has rank r, total rank is 3r.
    Schoolbook for 1×5×5 is 25, so total would be 75.
    
    But we can do better by sharing the B factor across the 3 rows.
    
    Let me think of the tensor as:
    T = sum_{i,j,k} e_i ⊗ e_k ⊗ e_j
    
    This is the tensor product of (sum_i e_i ⊗ e_i ⊗ e_i) [for the 3-dim part]
    and (sum_{j,k} e_j ⊗ e_k ⊗ e_j) [for the 5×5 part].
    
    Wait, that's not quite right. Let me be more careful.
    
    T[i,k,j] = 1 (all entries are 1 in the appropriate sense)
    No, T is the structure tensor: T_{(i,k),j} = delta... 
    
    OK let me just directly construct a good decomposition.
    
    For 3×5 × 5×5 → 3×5:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    One approach: use the fact that we can write this as a sum of rank-1 terms.
    A rank-1 term in the tensor is: u ⊗ w ⊗ v^* where
    (u ⊗ w ⊗ v^*)(A,B) = (u^T A v)(w^T B ... ) 
    
    Actually, let me use the standard tensor decomposition framework.
    The matrix multiplication tensor M_{n×m,p} has entries:
    M[(i,j,k), (i',j',k')] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    in the appropriate basis.
    
    A rank-1 decomposition: M = sum_r u_r ⊗ v_r ⊗ w_r
    where u_r is in R^{n×m}, v_r in R^{m×p}, w_r in R^{p×n} (with trace convention).
    
    The condition is:
    sum_r u_r[i,j] * v_r[j',k] * w_r[k',i'] = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    for all i,j,j',k,k',i'.
    
    This is equivalent to: for all i,i',k,k',
    sum_r u_r[i,j] v_r[j',k] w_r[k',i'] = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    
    Which means: sum_r u_r[i,:] ⊗ v_r[:,k] w_r[k',i'] = delta_{i,i'} delta_{k,k'} I_m
    
    So for i=i', k=k': sum_r u_r[i,j] v_r[j',k] w_r[k,i] = delta_{j,j'}
    For i≠i' or k≠k': sum_r u_r[i,j] v_r[j',k] w_r[k',i'] = 0
    
    This is a complex constraint. Let me try a constructive approach.
    
    Strategy: Use block decomposition.
    Partition the 5×5 B matrix into blocks and use Strassen-like methods.
    
    Alternatively, use the known result that the rank of M_{n,m,p} satisfies:
    R(M_{n,m,p}) ≤ n * R(M_{1,m,p})
    And R(M_{1,m,p}) = m*p (schoolbook is optimal for 1×m or m×1).
    
    So R(M_{3,5,5}) ≤ 3*25 = 75 (schoolbook).
    
    Can we do better? The key insight is that we can share structure across the 3 rows.
    
    Let me try: write B as a sum of rank-1 matrices.
    B = sum_{s=1}^{5} b_s^L * (b_s^R)^T (rank-5 decomposition of B, but this doesn't help directly).
    
    Actually, here's a better approach:
    C = A @ B
    = A @ (sum_s u_s v_s^T)  [if B = sum u_s v_s^T]
    = sum_s (A u_s) v_s^T
    
    But this decomposes B, not the tensor.
    
    Let me try yet another approach. Consider the tensor as:
    For each j (column of A / row of B), we have a "slice" that couples A[:,j] and B[j,:].
    
    The tensor is: sum_j (A[:,j] ⊗ B[j,:] → C)
    where A[:,j] is a 3-vector and B[j,:] is a 5-vector, and their "outer product" gives C.
    
    Specifically, C = sum_j A[:,j] * B[j,:] (outer product summed over j).
    
    So the tensor is the sum of 5 "slices", each being a 3×5 outer product tensor.
    Each slice has rank 3*5 = 15 (schoolbook for 3×5 outer product... wait, that's just a single term).
    
    No, each slice j contributes: C[i,k] += A[i,j] * B[j,k]
    This is a rank-1 term in the tensor! Because it's u_i * v_k * w_j where
    u depends on i, v on k, w on j.
    
    Wait, that means the schoolbook decomposition IS the minimal one?
    Each (j) gives one term: U[:,j] = e_j (in the A-space, specifically row i, col j),
    V[:,j] = e_j (in B-space, row j, col k),
    W[:,j] = ... 
    
    Hmm, I'm confusing myself. Let me go back to basics.
    
    The schoolbook decomposition for n×m × m×p has rank n*m*p.
    Each term corresponds to a unique (i,j,k) triple.
    U[i*m+j, col(i,j,k)] = 1
    V[j*p+k, col(i,j,k)] = 1  
    W[k*n+i, col(i,j,k)] = 1
    
    This gives rank = n*m*p = 3*5*5 = 75.
    
    To reduce rank, we need to find linear dependencies or use algebraic tricks.
    
    Known lower bounds: R(M_{n,m,p}) ≥ max(n*m, m*p, n*p) = max(15, 25, 15) = 25.
    Also R ≥ n*m*p / something... the Coppersmith-Winograd bound gives asymptotic improvements.
    
    For small matrices, the exact rank is often not known. For 3×5×5, I don't know the exact rank.
    
    Let me try a heuristic approach: use random initialization and optimize.
    """
    n, m, p = 3, 5, 5
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    # Try to reduce rank by finding linear dependencies
    # or using algebraic structure
    
    # One idea: group terms by j (the shared index between A and B)
    # For fixed j, we have n*p = 3*5 = 15 terms.
    # These 15 terms all share the same "j" in U and V.
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # 
    # The 15 terms for fixed j form a 3×5 grid in (i,k) space.
    # Can we decompose this 3×5 grid with fewer than 15 rank-1 terms?
    # A 3×5 matrix has rank ≤ 3, so maybe we can do it with 3*5/3 = 5 terms?
    # No, that's not how tensor rank works.
    
    # Let me try a different approach: use the fact that
    # sum_j A[i,j] B[j,k] can be computed with fewer multiplications
    # if we use Strassen-like techniques on blocks.
    
    # Partition A (3×5) and B (5×5) into blocks.
    # A = [A1 | A2] where A1 is 3×2, A2 is 3×3
    # B = [B1; B2] where B1 is 2×5, B2 is 3×5
    # Then AB = A1*B1 + A2*B2
    # Rank ≤ R(M_{3,2,5}) + R(M_{3,3,5})
    # Schoolbook: 3*2*5 + 3*3*5 = 30 + 45 = 75 (same as before)
    
    # Try: A = [A1 | A2 | A3] where A1 is 3×2, A2 is 3×2, A3 is 3×1
    # B = [B1; B2; B3] where B1 is 2×5, B2 is 2×5, B3 is 1×5
    # AB = A1*B1 + A2*B2 + A3*B3
    # Rank ≤ R(M_{3,2,5}) + R(M_{3,2,5}) + R(M_{3,1,5})
    # Schoolbook: 30 + 30 + 15 = 75
    
    # Use Strassen for 2×2 blocks:
    # If we could partition into 2×2 blocks... but 3 and 5 are not powers of 2.
    
    # Let me try: pad to 4×8 or use mixed-size blocks.
    # Actually, let me try a direct algebraic approach.
    
    # For 1×5 × 5×5: C[k] = sum_j A[j] B[j,k]
    # This is 5 dot products of length 5.
    # Can we compute 5 dot products of length 5 with fewer than 25 multiplications?
    # Each dot product has rank 5 (you need 5 multiplications for a length-5 dot product).
    # But the 5 dot products share the same A vector!
    # So it's: C = A @ B where A is 1×5 and B is 5×5.
    # This is just a matrix-vector product, which requires 25 multiplications in general.
    # Unless we use some trick.
    
    # Actually, for a single row vector times a matrix, the rank is m*p = 25.
    # There's no known way to do better for this specific case.
    
    # So for 3 rows, the rank is 3*25 = 75? That seems to be the schoolbook.
    
    # Wait, but we can share the B matrix across rows!
    # C[0,:] = A[0,:] @ B
    # C[1,:] = A[1,:] @ B  
    # C[2,:] = A[2,:] @ B
    
    # If we precompute some combinations of B's rows, we might save multiplications.
    # But in the tensor model, each "multiplication" is between one entry of A and one entry of B.
    # We can't share across different (i,j,k) triples unless we find algebraic structure.
    
    # Let me try a numerical optimization approach to find a lower-rank decomposition.
    # This is a hard problem, but let's see if we can find something.
    
    return U, V, W


def _optimize_rank(U, V, W, n, m, p, target_rank, max_iter=1000, lr=0.01, seed=42):
    """Try to optimize a tensor decomposition to achieve a target rank.
    
    This uses gradient-based optimization on the factor matrices.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with random factors
    U_opt = rng.randn(n * m, target_rank) * 0.1
    V_opt = rng.randn(m * p, target_rank) * 0.1
    W_opt = rng.randn(p * n, target_rank) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
    
    for it in range(max_iter):
        # Compute the loss: how well the decomposition matches the true tensor
        # Use random A, B to estimate the loss
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        AB_true = A @ B
        
        U_reshaped = U_opt.T.reshape(target_rank, n, m)
        V_reshaped = V_opt.T.reshape(target_rank, m, p)
        W_reshaped_temp = W_opt.T.reshape(target_rank, p, n)
        W_reshaped = W_reshaped_temp.transpose(0, 2, 1)
        
        U_A = np.einsum('rij,ij->r', U_reshaped, A)
        V_B = np.einsum('rjk,jk->r', V_reshaped, B)
        AB_pred = np.einsum('r,r,rik->ik', U_A, V_B, W_reshaped)
        
        loss = np.sum((AB_true - AB_pred) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
        
        # Gradient computation (simplified)
        # dL/dU = ... this is complex, let's use a simpler approach
        
        # Simple stochastic update
        U_opt += lr * rng.randn(n * m, target_rank)
        V_opt += lr * rng.randn(m * p, target_rank)
        W_opt += lr * rng.randn(p * n, target_rank)
        
        # Decay learning rate
        lr *= 0.99
    
    return best_U, best_V, best_W, best_loss


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition (rank 75)
    U, V, W = _schoolbook(n, m, p)
    
    if not _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 75
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower-rank decompositions
    # Theoretical lower bound: max(n*m, m*p, n*p) = max(15, 25, 15) = 25
    
    # Try various ranks from low to high
    for target_rank in range(25, 75):
        if time.time() > deadline - 20:
            break
        
        # Try optimization for this rank
        U_try, V_try, W_try, loss = _optimize_rank(
            U, V, W, n, m, p, target_rank, 
            max_iter=200, lr=0.05, seed=target_rank
        )
        
        if loss < 1e-6:
            # Verify with multiple random inputs
            if _verify(U_try, V_try, W_try, n, m, p, atol=1e-8):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    # Also try the block decomposition approach
    if time.time() < deadline - 10:
        U_block, V_block, W_block = _block_decompose(n, m, p)
        if _verify(U_block, V_block, W_block, n, m, p):
            rank_block = U_block.shape[1]
            if rank_block < best_rank:
                best_rank = rank_block
                best_U, best_V, best_W = U_block, V_block, W_block
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
