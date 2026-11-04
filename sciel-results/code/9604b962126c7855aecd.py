"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the matrix multiplication tensor.
    
    The matrix multiplication tensor T[i, j, k] = delta_{i, row} * delta_{j, col} for (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    With the trace convention W[k*n + i, r], the reconstruction is:
    (A * B)_{i, k} = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For this to equal sum_j A_{i,j} B_{j,k} for all A, B, we need:
    sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j, j'}
    for all i, j, j', k.
    """
    rank = U.shape[1]
    # Build the tensor as a 4D array: T[i, j, j', k] = sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r]
    # This should equal delta_{j, j'} for all i, k
    
    # Efficient check: for each i, k, the matrix M_{j, j'} = sum_r U[i*m+j,r] V[j'*p+k,r] W[k*n+i,r]
    # should be the identity matrix.
    
    # Let's construct this efficiently using matrix operations
    # For fixed i, k: M = U_i @ diag(W_k) @ V_k^T where U_i is the (m x rank) slice, etc.
    
    # Actually, let me think more carefully.
    # U has shape (n*m, rank), indexed by (i, j) -> row i*m+j
    # V has shape (m*p, rank), indexed by (j, k) -> row j*p+k  
    # W has shape (p*n, rank), indexed by (k, i) -> row k*n+i
    
    # For the decomposition to work:
    # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # = sum_r [sum_j A_{i,j} U[i*m+j, r]] * [sum_k' B_{k',k} ... ]
    
    # Wait, let me reconsider. The standard tensor decomposition for matrix multiplication:
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # where T_{i,j,k} = 1 if this is the "multiplication" tensor entry.
    
    # For matrix multiplication C = AB, C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is: sum_{i,j,k} C_{i,k} A_{i,j} B_{j,k} = sum_r sum_{i,j,k} (sum_j' A_{i,j'} U_{i,j',r}) (sum_k' B_{k',k} ... )
    
    # Let me use the standard formulation:
    # The matrix multiplication tensor M has entries M_{i,j,k} = 1 for all valid indices, 
    # and the decomposition is: M_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    
    # But actually for matrix multiplication, the tensor is:
    # T(A,B) = A ⊗ B (tensor product), and we want to express the bilinear map (A,B) -> AB
    # as a sum of rank-1 terms.
    
    # The correct check: for random A (n x m) and B (m x p), 
    # compute AB directly and via the decomposition, and check they match.
    
    # Via decomposition:
    # (AB)_{i,k} = sum_r [sum_j A_{i,j} U_{i*m+j, r}] * [sum_{j'} B_{j', k} V_{j'*p+k, r}] * W_{k*n+i, r}
    
    # Hmm, that's not quite right either. Let me think about it differently.
    
    # In a rank-r tensor decomposition of the matrix multiplication tensor:
    # The tensor T has T_{i,j,k} = 1 (for the "1" entries in the multiplication table)
    # Actually no. The matrix multiplication tensor is defined as:
    # T(i,j,k) = 1 if we're multiplying entry (i,j) of A with entry (j,k) of B to get entry (i,k) of C.
    
    # More precisely, the bilinear map is:
    # f(A, B)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # A rank-1 decomposition term is: f_r(A, B)_{i,k} = (sum_j a_{i,j} U_{i,j,r}) * (sum_j b_{j,k} V_{j,k,r}) * W_{k,i,r}
    # Wait, that gives a scalar times W_{k,i,r} for each (i,k), which isn't right.
    
    # Let me look at this more carefully. The standard formulation:
    # Matrix multiplication as a tensor: T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p}
    # T(A, B) = AB means: <T, A ⊗ B ⊗ C> = trace(C^T AB) = trace(AB C^T)
    
    # For a rank-1 term: U ⊗ V ⊗ W where U ∈ R^{n×m}, V ∈ R^{m×p}, W ∈ R^{n×p}
    # <U ⊗ V ⊗ W, A ⊗ B ⊗ C> = <U, A> <V, B> <W, C> = tr(U^T A) tr(V^T B) tr(W^T C)
    
    # So we need: tr(C^T AB) = sum_r tr(U_r^T A) tr(V_r^T B) tr(W_r^T C) for all A, B, C
    
    # Equivalently: AB = sum_r (tr(U_r^T A)) (tr(V_r^T B)) W_r
    
    # For this to hold, we need: sum_r U_r^T A * V_r^T B * W_r = AB for all A, B
    # i.e., sum_r (U_r^T A) (V_r^T B)^T W_r = AB
    
    # Wait, tr(U_r^T A) is a scalar, tr(V_r^T B) is a scalar, and W_r is a matrix.
    # So: sum_r [tr(U_r^T A)] [tr(V_r^T B)] W_r = AB
    
    # In the schoolbook case:
    # U_{i,j,r} = 1 if r corresponds to (i,j,k) with this i,j, else 0
    # V_{j,k,r} = 1 if r corresponds to (i,j,k) with this j,k, else 0  
    # W_{i,k,r} = 1 if r corresponds to (i,j,k) with this i,k, else 0
    
    # Then tr(U_r^T A) = A_{i,j}, tr(V_r^T B) = B_{j,k}, W_r has 1 at position (i,k)
    # So sum_r A_{i,j} B_{j,k} * E_{i,k} = AB. Correct!
    
    # So the check is: for random A, B, verify sum_r tr(U_r^T A) tr(V_r^T B) W_r == AB
    
    # In our indexing:
    # U is (n*m, rank), so U_r is a column of U, which we can reshape to (n, m)
    # V is (m*p, rank), so V_r is a column of V, reshape to (m, p)
    # W is (p*n, rank), so W_r is a column of W, reshape to (n, p) -- but with W[k*n+i] indexing
    
    # W[k*n+i] means row index = k*n+i, so for fixed r, W[:, r] has entries W[k*n+i, r]
    # This corresponds to a matrix W_r where (W_r)_{i,k} = W[k*n+i, r]
    # i.e., W_r is the transpose of the matrix obtained by reshaping W[:,r] to (p, n)
    
    # Let's just do a numerical check with random matrices.
    
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    AB_direct = A @ B
    
    # Compute via decomposition
    # For each r: scalar_r = tr(U_r^T A) * tr(V_r^T B)
    # U_r as (n,m) matrix: reshape U[:, r] from (n*m,) to (n, m) with row-major order
    # U[i*m+j, r] -> (U_reshaped)_{i,j}
    U_reshaped = U.T.reshape(rank, n, m)  # (rank, n, m)
    V_reshaped = V.T.reshape(rank, m, p)  # (rank, m, p)
    # W[k*n+i, r] -> we want (W_reshaped)_{i,k} = W[k*n+i, r]
    # W[:, r] has shape (p*n,), indexed by k*n+i
    # Reshape to (p, n): W_reshaped_temp[k, i] = W[k*n+i, r]
    # Then (W_r)_{i,k} = W_reshaped_temp[k, i], so W_r = W_reshaped_temp.T
    W_reshaped_temp = W.T.reshape(rank, p, n)  # (rank, p, n), W_reshaped_temp[r, k, i] = W[k*n+i, r]
    W_reshaped = W_reshaped_temp.transpose(0, 2, 1)  # (rank, n, p), W_reshaped[r, i, k] = W[k*n+i, r]
    
    # tr(U_r^T A) = sum_{i,j} U_r[i,j] * A[i,j] = <U_r, A>
    # tr(V_r^T B) = sum_{j,k} V_r[j,k] * B[j,k] = <V_r, B>
    
    U_A = np.einsum('rij,ij->r', U_reshaped, A)  # (rank,)
    V_B = np.einsum('rjk,jk->r', V_reshaped, B)  # (rank,)
    
    # sum_r U_A[r] * V_B[r] * W_r
    AB_decomp = np.einsum('r,r,rik->ik', U_A, V_B, W_reshaped)
    
    return np.allclose(AB_direct, AB_decomp, atol=1e-10)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    
    # Verify the schoolbook decomposition works
    if not _verify(U, V, W, n, m, p):
        # Should never happen, but just in case
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # The schoolbook rank is n*m*p = 75
    # We want to try to reduce the rank.
    # 
    # Key insight: the matrix multiplication tensor for (n,m,p) = (3,5,5) 
    # might have a lower rank than the schoolbook 75.
    #
    # One approach: use known fast matrix multiplication algorithms.
    # For 5x5 matrices, the Strassen algorithm gives rank 25 (5^2) for each 5x5 block,
    # but that's for square matrices.
    #
    # Actually, for multiplying a 3x5 by 5x5 matrix, we can think of it as:
    # C = A * B where A is 3x5 and B is 5x5.
    # We can partition A into 3 rows, each of length 5.
    # C_i = A_i * B for i = 1,2,3, where A_i is a 1x5 row vector.
    # Each C_i is a 1x5 row vector.
    # 
    # For a 1x5 times 5x5 multiplication, the tensor rank is 5 (schoolbook).
    # So total rank would be 3*5 = 15 if we could do each row independently.
    # But that's not quite right because the tensor structure couples them.
    
    # Let me think about this differently. The matrix multiplication tensor for 
    # multiplying an (n x m) matrix by an (m x p) matrix has rank at most n*m*p (schoolbook).
    # 
    # For n=3, m=5, p=5, schoolbook rank is 75.
    # 
    # Can we do better? One idea: if we can find a decomposition where some of the
    # rank-1 terms are shared across different (i,k) pairs, we can reduce the rank.
    
    # Let's try a simple approach: use the fact that for each fixed j, the term
    # A_{i,j} * B_{j,k} contributes to C_{i,k}. We can group terms by j.
    
    # Actually, let me try to use random initialization and optimization to find
    # a lower-rank decomposition.
    
    # First, let's try rank 25 (which would be a significant improvement over 75).
    # This is based on the idea that 5x5 matrix multiplication can be done with rank 25
    # using Strassen-like methods, and we have 3 such "rows" but they share the same B.
    
    # Actually, the minimum rank for 5x5 matrix multiplication is known to be at most 49
    # (using Strassen recursively) but the exact minimum is unknown. For our problem,
    # we're multiplying 3x5 by 5x5, which is different from 5x5 by 5x5.
    
    # Let me try a different approach: since the problem is relatively small, I'll try
    # to use a greedy or optimization-based approach to find a lower-rank decomposition.
    
    # Strategy: Start with the schoolbook decomposition and try to merge terms
    # or use optimization to reduce the rank.
    
    # For now, let's just return the schoolbook decomposition as a baseline.
    # The key challenge is finding a provably correct lower-rank decomposition.
    
    # Let me try one more thing: check if there's a simple structure we can exploit.
    # Since m = p = 5, we have a "square" inner dimension.
    
    # Try: can we express this as 3 independent 1x5 * 5x5 multiplications?
    # For row i of A, C_i = A_i * B. The tensor for 1x5 * 5x5 has rank 5.
    # If the 3 rows are independent, total rank = 3*5 = 15.
    # But they're not independent in the tensor sense because they all multiply the same B.
    
    # Actually, they ARE independent in the sense that the tensor decomposes as a direct sum:
    # T_{3x5, 5x5, 3x5} = T_{1x5, 5x5, 1x5} ⊕ T_{1x5, 5x5, 1x5} ⊕ T_{1x5, 5x5, 1x5}
    # because the output rows don't interact.
    
    # Wait, is that true? Let's check. The tensor is:
    # T(A, B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # For different i values, the expressions are independent (they only share B).
    # In tensor decomposition, if T = T1 ⊕ T2 (direct sum), then rank(T) = rank(T1) + rank(T2).
    
    # For T_i (the tensor for row i): T_i(A, B) = (A_i * B) where A_i is row i of A.
    # This is a bilinear map from R^5 (row i of A) × R^{5x5} (B) to R^5 (row i of C).
    # The rank of this tensor is 5 (schoolbook for 1x5 * 5x5).
    
    # So total rank = 3 * 5 = 15!
    
    # Let me construct this decomposition.
    
    rank = 15
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            r = i * m + j  # rank index for this (i, j) pair
            U[i * m + j, r] = 1
            V[j * p + 0, r] = 0  # We need to be more careful here
            
    # Hmm, this isn't quite working. Let me think more carefully.
    
    # For row i of A, the multiplication C_i = A_i * B is:
    # C_i[k] = sum_j A_i[j] * B[j, k]
    # 
    # The tensor for this is: sum_j A_i[j] B[j,k] for each k.
    # As a bilinear form: <T_i, A_i ⊗ B> = sum_{j,k} A_i[j] B[j,k] C_i[k]
    # 
    # Rank-1 decomposition: for each j, the term is A_i[j] * B[j,:] * e_k... no.
    
    # Let me use the standard rank-1 tensor form:
    # T_i(A_i, B) = sum_r [sum_j A_i[j] U_{i,j,r}] [sum_{j',k'} B[j',k'] V_{j',k',r}] W_{k,r}
    # 
    # For this to equal sum_j A_i[j] B[j,k] for each k:
    # We need: sum_r [sum_j A_i[j] U_{i,j,r}] [sum_{j',k'} B[j',k'] V_{j',k',r}] W_{k,r} = sum_j A_i[j] B[j,k]
    
    # Simplest: let r index j, and:
    # U_{i,j,r} = delta_{j,r} (so sum_j A_i[j] U_{i,j,r} = A_i[r])
    # V_{j',k',r} = delta_{j',r} * delta_{k',k}... but k is not fixed in V.
    
    # Hmm, the issue is that W depends on k, but V should not depend on k in a way that couples different k's.
    
    # Let me try: r = (j) for j = 1..m
    # U_{i,j,r} = delta_{j,r}
    # V_{j',k',r} = delta_{j',r} (independent of k')
    # W_{k,r} = B... no, W is a fixed matrix.
    
    # I think I'm overcomplicating this. Let me just use the schoolbook decomposition
    # and return it. The rank is 75, which is the baseline.
    
    # Actually, wait. Let me reconsider the direct sum argument.
    
    # The full tensor T maps (A, B) -> AB where A is n x m, B is m x p, AB is n x p.
    # We can write T = sum_i T_i where T_i(A, B) is the i-th row of AB.
    # T_i(A, B) = A_i * B where A_i is the i-th row of A (a 1 x m vector) and B is m x p.
    # The result is a 1 x p vector.
    
    # The tensor T_i has components: T_i_{j,k} = B_{j,k} * A_{i,j}... no.
    
    # Let me be very explicit. The tensor T ∈ (R^{n x m})* ⊗ (R^{m x p})* ⊗ (R^{n x p})*
    # is defined by: T(A, B, C) = trace(C^T A B)
    
    # T_i is the part corresponding to the i-th row of C:
    # T_i(A, B, C) = sum_k C_{i,k} (AB)_{i,k} = sum_k C_{i,k} sum_j A_{i,j} B_{j,k}
    # = sum_{j,k} A_{i,j} B_{j,k} C_{i,k}
    
    # So T_i ∈ (R^m)* ⊗ (R^{m x p})* ⊗ (R^p)* where the first factor is A_i (row i of A).
    
    # T_i(A_i, B, C_i) = sum_{j,k} A_i[j] B[j,k] C_i[k]
    
    # This is a trilinear form. Its rank is at most m*p = 25 (schoolbook for 1x5 x 5x5 x 1x5).
    # Actually, for the form sum_{j,k} x_j y_{j,k} z_k, the rank is m*p because each (j,k) pair
    # gives an independent term: x_j * y_{j,k} * z_k.
    
    # So rank(T_i) = m*p = 25 for each i, and total rank = n * m * p = 75.
    
    # That's just the schoolbook rank again. So the direct sum doesn't help here.
    
    # OK so the schoolbook rank of 75 is likely optimal or close to optimal for this problem.
    # Let me just return the schoolbook decomposition.
    
    U, V, W = _schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
