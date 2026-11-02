"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _build_tensor():
    """Build the 3x4x5 matrix multiplication tensor T[i,j,k] = delta[i,j,k]
    where T[i, j, k] = 1 if the (i,k)-th entry of the product depends on A[i,j]*B[j,k].
    Actually the multiplication tensor is T[i, j, k] such that
    (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor has entries T[i, j, k] = 1 for all i,j,k (it's the identity in the middle).
    But for decomposition we think of it as a 3-index tensor T[i, j, k] where
    U[i,j] * V[j,k] * W[i,k] ... 
    
    Actually the standard formulation: the multiplication tensor T is in A^* ⊗ B^* ⊗ C^*
    where A is n×m, B is m×p, C is n×p.
    T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C where the contraction is (A ⊗ B) contracted with T.
    
    More precisely: T[i, j, k] = 1 if the (i,k) output depends on A[i,j] and B[j,k].
    Since (AB)[i,k] = sum_j A[i,j] B[j,k], every triple (i,j,k) contributes.
    So T is the tensor with T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    
    Wait, that's not right either. The tensor for matrix multiplication is:
    T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i^* ⊗ e_j^* ⊗ e_{(i,k)}^*
    where e_i^* are basis of A^*, e_j^* of B^*, e_{(i,k)}^* of C^*.
    
    So T[i, j, (i,k)] = 1 and 0 otherwise. This is a 3-tensor of shape (n*m, m*p, n*p).
    """
    n, m, p = DIMENSIONS
    # T is shape (n*m, m*p, n*p)
    T = np.zeros((n * m, m * p, n * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i  (official trace convention W[k*n+i, r])
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T

def _verify_decomposition(U, V, W, T):
    """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] == T"""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Reconstruct tensor
    T_recon = np.zeros(T.shape)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_recon, T, atol=1e-10)

def _try_decomposition(T, max_rank, time_limit=120):
    """Try to find a decomposition of T with rank <= max_rank using greedy + polish."""
    n, m, p = DIMENSIONS
    dim_u, dim_v, dim_w = T.shape
    start_time = time.time()
    
    # Start with schoolbook
    rank_school = n * m * p
    U = np.zeros((dim_u, rank_school))
    V = np.zeros((dim_v, rank_school))
    W = np.zeros((dim_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Try to reduce rank by finding linear dependencies
    # The tensor T can be viewed as a matrix of shape (dim_u, dim_v*dim_w)
    # Rank of T as a matrix gives a lower bound on tensor rank
    T_mat = T.reshape(dim_u, dim_v * dim_w)
    matrix_rank = np.linalg.matrix_rank(T_mat, tol=1e-10)
    
    # Also try other matricizations
    T_mat2 = T.reshape(dim_v, dim_u * dim_w)
    matrix_rank2 = np.linalg.matrix_rank(T_mat2, tol=1e-10)
    
    T_mat3 = T.reshape(dim_w, dim_u * dim_v)
    matrix_rank3 = np.linalg.matrix_rank(T_mat3, tol=1e-10)
    
    min_possible = max(matrix_rank, matrix_rank2, matrix_rank3)
    
    # For 3x4x5: 
    # T shape is (12, 20, 15)
    # Matricization (12, 300): rank?
    # Matricization (20, 180): rank?
    # Matricization (15, 240): rank?
    
    # Let's compute these
    print(f"Matrix ranks: {matrix_rank}, {matrix_rank2}, {matrix_rank3}")
    print(f"Min possible rank: {min_possible}")
    
    # Try greedy approach: start with full rank, try to merge terms
    # A term is (u, v, w) vector triple
    # Two terms (u1,v1,w1) and (u2,v2,w2) can be merged if there exist scalars
    # such that the sum can be expressed with fewer terms
    
    # Alternative: use the fact that for matrix multiplication, we can use
    # "fast matrix multiplication" style decompositions
    
    # For 3x4x5, let's try a structured approach:
    # Group by the middle index j (0 to 3)
    # For each j, we have a 3x5 outer product structure
    # T_j[i, k] contributes to U[i*m+j], V[j*p+k], W[k*n+i]
    
    # The key insight: for each j, the sub-tensor is rank 1 in the (i,k) space
    # but different j's share the same i and k indices
    
    # Let's try: can we find a decomposition where we use the same U and V
    # patterns across different j values?
    
    # Try rank 20: one term per (i,k) pair? No, that's 15.
    # Try rank 15: one term per (i,k) pair
    # For each (i,k), we need sum_j A[i,j] B[j,k]
    # This is a dot product over j, which requires m terms in general
    # Unless we can use a clever basis
    
    # For m=4, the minimum rank to compute a dot product of length 4 is 4
    # (over reals), so for each (i,k) pair we need at least 4 terms
    # Total: 15 * 4 = 60, which is schoolbook
    
    # But wait - we can share structure across different (i,k) pairs!
    # The U vectors for different i's are independent (different rows)
    # The V vectors for different k's are independent (different columns)
    # The W vectors for different (i,k) are independent
    
    # Actually, the tensor rank is at least the maximum of the three matricization ranks.
    # Let me compute those more carefully.
    
    # For the 3x4x5 multiplication tensor:
    # T[i, j, k] is nonzero only when... wait, I need to re-examine.
    
    # The multiplication tensor for C = AB where A is 3x4, B is 4x5:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # The tensor T has T[i, j, k] = 1 for all i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3,4}
    # Wait no - T is a 3-tensor with indices (input_A, input_B, output_C)
    # T[a, b, c] = 1 iff the output c depends on input A at position a and input B at position b
    
    # For matrix mult: output (i,k) depends on A[i,j] and B[j,k] for each j
    # So T[i*m+j, j*p+k, k*n+i] = 1
    
    # This is what I built. Let me verify the matricization ranks.
    
    # The tensor T of shape (12, 20, 15)
    # Matricize as (12, 300): each row corresponds to an A index
    # Row i*m+j: has 1s at positions j*p+k for k=0..4, and the W index is k*n+i
    # So row i*m+j has 5 ones (one for each k)
    
    # Two rows i1*m+j1 and i2*m+j2 are linearly independent if (i1,j1) != (i2,j2)
    # because the support patterns differ.
    # Actually row i*m+j has 1s at W positions k*n+i for k=0..4
    # So rows with different i have disjoint W supports (different i in the W index)
    # Rows with same i but different j have the same W support pattern but different V supports
    
    # Let me just compute numerically.
    
    # For now, let's try a search approach with time limit
    
    # Try to find decomposition of rank r for r from min_possible to schoolbook
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    # Try a few candidate ranks
    for target_rank in range(min_possible, rank_school + 1):
        if time.time() - start_time > time_limit - 10:
            break
        if target_rank >= best_rank:
            break
        
        # Try to find a decomposition of this rank
        # Use a simple approach: random initialization + optimization
        # But for exact decomposition, we need a different approach
        
        # For small tensors, we can try to solve linear systems
        # A rank-r decomposition means T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
        # This is equivalent to: for each (a,c) pair, T[a,:,c] = sum_r u_r[a] * v_r * w_r[c]
        
        # Fix U and W, solve for V, and vice versa
        # This is alternating least squares
        
        # Initialize randomly
        np.random.seed(42)
        U_curr = np.random.randn(dim_u, target_rank)
        V_curr = np.random.randn(dim_v, target_rank)
        W_curr = np.random.randn(dim_w, target_rank)
        
        # Normalize
        for r in range(target_rank):
            norm = np.linalg.norm(W_curr[:, r])
            if norm > 1e-10:
                W_curr[:, r] /= norm
        
        # Alternating minimization
        converged = False
        for iteration in range(100):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Solve for U given V, W
            # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
            # For each a: T[a, :, :] = sum_r U[a, r] * (V[:, r] ⊗ W[:, r])
            # Reshape: T[a].reshape(dim_v*dim_w) = U[a, :] @ (V ⊙ W).T
            # where (V ⊙ W) is the Khatri-Rao product
            
            # Build the design matrix
            # For each r, the vector is V[:, r] ⊗ W[:, r] (Kronecker product)
            # Actually we need V[:, r] * W[:, r] element-wise, then flatten
            # No: T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
            # So for fixed a: T[a, b, c] over (b,c) = sum_r U[a,r] * (V[b,r] W[c,r])
            # The matrix M_r[b, c] = V[b, r] * W[c, r]
            # T[a].reshape(dim_v, dim_w) = sum_r U[a,r] * M_r
            # So T[a].reshape(dim_v*dim_w) = U[a, :] @ (stack of vec(M_r))
            
            # Build design matrix D of shape (dim_v*dim_w, target_rank)
            D = np.zeros((dim_v * dim_w, target_rank))
            for r in range(target_rank):
                D[:, r] = (V_curr[:, r] * W_curr[:, r]).ravel()
            
            # Solve U[a, :] = T[a].ravel() @ pinv(D)
            D_pinv = np.linalg.pinv(D, rcond=1e-10)
            U_new = np.zeros((dim_u, target_rank))
            for a in range(dim_u):
                U_new[a, :] = T[a].ravel() @ D_pinv
            
            # Solve for V given U, W
            D2 = np.zeros((dim_u * dim_w, target_rank))
            for r in range(target_rank):
                D2[:, r] = (U_new[:, r] * W_curr[:, r]).ravel()
            D2_pinv = np.linalg.pinv(D2, rcond=1e-10)
            V_new = np.zeros((dim_v, target_rank))
            for b in range(dim_v):
                V_new[b, :] = T[:, b, :].reshape(dim_u, dim_w).ravel() @ D2_pinv
            
            # Solve for W given U, V
            D3 = np.zeros((dim_u * dim_v, target_rank))
            for r in range(target_rank):
                D3[:, r] = (U_new[:, r] * V_new[:, r]).ravel()
            D3_pinv = np.linalg.pinv(D3, rcond=1e-10)
            W_new = np.zeros((dim_w, target_rank))
            for c in range(dim_w):
                W_new[c, :] = T[:, :, c].reshape(dim_u, dim_v).ravel() @ D3_pinv
            
            # Check convergence
            U_curr, V_curr, W_curr = U_new, V_new, W_new
            
            # Normalize W
            for r in range(target_rank):
                norm = np.linalg.norm(W_curr[:, r])
                if norm > 1e-10:
                    W_curr[:, r] /= norm
                    U_curr[:, r] *= norm
                    V_curr[:, r] *= norm
            
            # Check residual
            T_recon = np.zeros(T.shape)
            for r in range(target_rank):
                T_recon += np.einsum('a,b,c->abc', U_curr[:, r], V_curr[:, r], W_curr[:, r])
            residual = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
            
            if residual < 1e-10:
                converged = True
                break
        
        if converged:
            # Verify
            if _verify_decomposition(U_curr, V_curr, W_curr, T):
                best_U, best_V, best_W = U_curr, V_curr, W_curr
                best_rank = target_rank
                print(f"Found decomposition of rank {target_rank}")
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return U_list, V_list, W_list, best_rank

def construct():
    start_time = time.time()
    T = _build_tensor()
    
    # Quick check: try a few structured decompositions first
    
    # Schoolbook
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_sb = [[0.0] * rank_school for _ in range(n * m)]
    V_sb = [[0.0] * rank_school for _ in range(m * p)]
    W_sb = [[0.0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1.0
                V_sb[j * p + k][col] = 1.0
                W_sb[k * n + i][col] = 1.0
                col += 1
    
    # Try to find better
    U_best, V_best, W_best, rank_best = U_sb, V_sb, W_sb, rank_school
    
    try:
        U_found, V_found, W_found, rank_found = _try_decomposition(T, rank_school, time_limit=110)
        if rank_found < rank_best:
            U_best, V_best, W_best, rank_best = U_found, V_found, W_found, rank_found
    except Exception as e:
        print(f"Error in decomposition search: {e}")
    
    # Final verification
    U_arr = np.array(U_best)
    V_arr = np.array(V_best)
    W_arr = np.array(W_best)
    if not _verify_decomposition(U_arr, V_arr, W_arr, T):
        # Fall back to schoolbook
        U_best, V_best, W_best = U_sb, V_sb, W_sb
    
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
