import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Strassen parameters for 2x2x2
    U_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    
    V_strassen = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [-1, 0, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    W_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1],# P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(n * m):
                U[i][col] = U_strassen[r][i]
                
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l]][col] = V_strassen[r][l]
                
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l]][col] = W_strassen[r][l]
                
            col += 1
    
    # Verify the base decomposition
    def verify(U_mat, V_mat, W_mat):
        """Verify that sum_r U[i,r]*V[j,r]*W[k,r] = delta(j,k) for all i,j,k"""
        n_, m_, p_ = n, m, p
        # T(i, j, k) = sum_r U[i,r] * V[j,r] * W[k,r]
        # Should equal delta_{j,k} where i indexes A row, j indexes A col/B row, k indexes B col
        # A is n x m, B is m x p, C = A*B is n x p
        # Tensor: T(i, j, k) = delta_{j, k'} where k' is B column index
        # Actually the multiplication tensor: C[i][k] = sum_j A[i][j] * B[j][k]
        # So T(i, j, k) = 1 if j == k (in the B matrix sense), 0 otherwise
        # Wait - let me re-read. The tensor for matrix multiplication (n,m,p):
        # T is a 3-tensor with indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
        # T(i,j,k) = 1 if j == k else 0 ... no that's not right either.
        # 
        # Standard: C[i][k] = sum_j A[i][j] * B[j][k]
        # The tensor T has T[i,j,k] = delta_{j,k} ... no.
        # T[i,j,k] should be such that sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
        # And T[i,j,k] = 1 if this term contributes to C[i][k] via A[i][j]*B[j][k]
        # So T[i,j,k] = 1 for all valid i,j,k (it's just the identity in the j-k contraction)
        # Actually no: T[i,j,k] = delta_{j,k} is wrong.
        # 
        # Let me think again. For matrix mult (n,m,p):
        # C[i][k] = sum_{j=0}^{m-1} A[i][j] * B[j][k]
        # The rank-r decomposition: C[i][k] = sum_{r=0}^{r-1} (sum_j U[i,r]*A[i][j]) * (sum_k V[j,r]*B[j][k]) * W[k,r]
        # No, that's not right either.
        #
        # Standard tensor decomposition:
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # where T[i,j,k] is the multiplication tensor.
        # For matrix multiplication (n x m) * (m x p):
        # T[i,j,k] = 1 if j is the shared index... 
        # Actually T[i,j,k] = delta_{j,k} doesn't make sense since j in [0,m) and k in [0,p).
        #
        # The correct tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # NO. The tensor is: T[i,j,k] = 1 if this triple (i,j,k) represents a valid multiplication term.
        # Every term A[i][j]*B[j][k] contributes to C[i][k], so T[i,j,k] = 1 for all i,j,k.
        # That can't be right because then the rank would be 1.
        #
        # I think the correct interpretation is:
        # The multiplication tensor T has T[i,j,k] = 1 if and only if the indices form a valid
        # (row of A, col of A / row of B, col of B) triple. Since every j in [0,m) is a valid
        # shared index, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # But that gives rank 1 which is clearly wrong.
        #
        # Let me reconsider. The standard definition:
        # T in R^{n x m x p}, T[i,j,k] = delta_{j,k} ... no, j and k have different ranges.
        #
        # Actually I think the issue is that the tensor is defined as:
        # T[i,j,k] = 1 if j == k (when m == p), or more generally the tensor encodes
        # the bilinear map. Let me just verify numerically with the Strassen construction.
        
        # Build the tensor from U, V, W
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For matrix mult (n,m,p), T[i,j,k] should be 1 for all i,j,k
        # because C[i][k] = sum_j A[i][j] B[j][k]
        # and each term is separated as (A[i][j]) * (B[j][k]) with coefficient 1.
        # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        # Wait, that would mean the tensor is all-ones, which has rank 1. That's clearly wrong.
        # 
        # I think the correct tensor is: T[i,j,k] = delta_{j,k} is NOT right.
        # The correct one: for each output element C[i][k], we sum over j:
        # C[i][k] = sum_j A[i][j] * B[j][k]
        # The tensor T[i,j,k] represents the coefficient of A[i][j] * B[j][k] in C[i][k].
        # That coefficient is 1 for all valid i,j,k. So T is all ones? No...
        #
        # OH WAIT. I think I've been confusing myself. Let me look at this differently.
        # The tensor is T in R^{(n*m) x (m*p) x (p*n)} or similar indexing.
        # 
        # Actually, looking at the shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # So the indices are:
        # i in [0, n*m): combined (row, col) of A
        # j in [0, m*p): combined (row, col) of B  
        # k in [0, p*n): combined (col, row) of C (with convention k*n+i)
        #
        # T[i,j,k] = 1 if the multiplication term A[row_i][col_i] * B[row_j][col_j] contributes to C[row_k][col_k]
        # i.e., if col_i == row_j AND row_i == row_k AND col_j == col_k
        # 
        # With i = a_row * m + a_col, j = b_row * p + b_col, k = c_col * n + c_row
        # T[i,j,k] = 1 if a_col == b_row AND a_row == c_row AND b_col == c_col
        # Otherwise 0.
        
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        
        # Compute T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # This is an outer product sum
        # T = sum_r np.outer(np.outer(U_arr[:,r], V_arr[:,r]), W_arr[:,r])
        # More efficiently:
        T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        
        # Expected tensor
        T_expected = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n*m):
            a_row = i // m
            a_col = i % m
            for j in range(m*p):
                b_row = j // p
                b_col = j % p
                if a_col != b_row:
                    continue
                for k in range(p*n):
                    c_col = k // n
                    c_row = k % n
                    if a_row == c_row and b_col == c_col:
                        T_expected[i,j,k] = 1.0
        
        if not np.allclose(T, T_expected, atol=1e-10):
            return False
        return True
    
    if not verify(U, V, W):
        raise ValueError("Base Strassen decomposition failed verification")
    
    # Now attempt rank reduction by finding linear dependencies
    # The key insight: the A-matrix (U factors) are identical across all 4 column blocks.
    # This means we can potentially combine terms across blocks.
    # 
    # Strategy: For each Strassen term r (0..6), we have 4 copies (one per block t).
    # The U vectors are identical across blocks. The V and W vectors differ.
    # 
    # If we can find coefficients alpha_{t,r} such that sum_t alpha_{t,r} * (V_t_r, W_t_r) 
    # gives a lower-rank representation, we can reduce the rank.
    #
    # More concretely: the tensor can be written as:
    # T = sum_{t=0}^{3} sum_{r=0}^{6} U_r * V_{t,r} * W_{t,r}
    # where U_r is the same for all t.
    #
    # We want to find a decomposition with fewer than 28 terms.
    # 
    # Approach: Try to find a basis for the V-W products.
    # The V vectors live in R^{m*p} = R^{16}, W vectors in R^{p*n} = R^{16}.
    # Each term is a rank-1 tensor U_r * V_{t,r} * W_{t,r}.
    # 
    # Since U_r is fixed per r, the question is whether the 4 vectors (V_{t,r}, W_{t,r})
    # for t=0..3 can be combined.
    #
    # Let's try a different approach: use the structure to find a minimal decomposition.
    # 
    # For the 2x2x2 case, Strassen gives rank 7 (optimal).
    # For 2x2x8, we can think of it as 4 independent 2x2x2 multiplications sharing the A matrix.
    # The rank is at most 4*7 = 28.
    #
    # Can we do better? The A matrix has 4 elements. If we could "reuse" Strassen products
    # across column blocks, we might reduce rank.
    #
    # Let's try: express the problem as finding a rank-r decomposition where
    # sum_{s=0}^{r-1} U_s[i] * V_s[j] * W_s[k] = T[i,j,k]
    #
    # We'll use a greedy/iterative approach: start with the 28-term decomposition and try
    # to merge terms by finding linear dependencies.
    #
    # Actually, let's try a more direct approach: use the fact that the tensor has a
    # Kronecker product structure. T_{2x2x8} = T_{2x2x2} ⊗ I_4 in some sense.
    # 
    # More precisely: T[i,j,k] = 1 iff a_col == b_row, a_row == c_row, b_col == c_col
    # Let a_col = b_row = j' (shared index in [0,2)), a_row = c_row = i' (in [0,2)),
    # b_col = c_col = k' (in [0,8)).
    # 
    # So T = sum_{j'=0}^{1} sum_{i'=0}^{1} sum_{k'=0}^{7} e_{i',j'} * e_{j',k'} * e_{k',i'}
    # where e_{i',j'} is a basis vector in the appropriate space.
    #
    # This is just the identity tensor, which has rank n*m*p = 32 in general.
    # Strassen reduces the 2x2x2 part to rank 7, giving 7*4 = 28.
    #
    # Can we do better than 28? The lower bound for 2x2x8 is at least 2*2*8/2 = 16
    # (from the rank of the flattening). Let's check the flattening ranks.
    #
    # For now, let's try to find a rank < 28 decomposition by exploiting the structure.
    # 
    # Idea: Use the Strassen decomposition but with shared "intermediate" computations.
    # In Strassen, we compute 7 products of 2x2 matrices. For 2x2x8, we have 4 column
    # pairs of B. We could potentially share some of the intermediate sums.
    #
    # Let's try a constructive approach:
    # B is 2x8 = [B_0 B_1 B_2 B_3] where each B_t is 2x2.
    # C = A * B = [A*B_0 A*B_1 A*B_2 A*B_3]
    # 
    # Using Strassen for each A*B_t independently gives rank 28.
    # But if we compute A*(B_0+B_1), A*(B_0-B_1), etc., we might share work.
    #
    # Actually, let's try: define combined B matrices and use Strassen on them.
    # B_0 + B_1, B_0 - B_1, B_2 + B_3, B_2 - B_3, etc.
    # Then A*(B_0+B_1) and A*(B_0-B_1) can be used to recover A*B_0 and A*B_1.
    # But this doesn't reduce the number of Strassen multiplications.
    #
    # Let me try a different strategy: use a rank-reduction search.
    # Start with the 28-term decomposition and try to eliminate terms by finding
    # linear combinations.
    
    def try_reduce_rank(U_init, V_init, W_init, max_rank, time_limit=120):
        """Try to find a lower-rank decomposition starting from the given one."""
        start_t = time.time()
        n_, m_, p_ = n, m, p
        
        # Build the target tensor
        U_arr = np.array(U_init, dtype=float)
        V_arr = np.array(V_init, dtype=float)
        W_arr = np.array(W_init, dtype=float)
        T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        
        # For rank reduction, we try to find a decomposition with r < 28 terms.
        # We'll use a simple approach: try to find a rank-r decomposition by
        # solving for the factors.
        #
        # This is NP-hard in general, but for small ranks we can try.
        # 
        # Alternative: use the structure. The U factors repeat across blocks.
        # Let's try to find a decomposition where the U factors are from a smaller set.
        
        # Try ranks from max_rank down to 16
        for r in range(min(max_rank, 27), 15, -1):
            if time.time() - start_t > time_limit:
                break
            
            # Try to find a rank-r decomposition
            # We'll use a randomized approach: initialize and iterate
            best_valid = None
            
            for attempt in range(50):
                if time.time() - start_t > time_limit:
                    break
                
                # Initialize with random factors
                np.random.seed(42 + attempt)
                U_try = np.random.randn(n*m, r)
                V_try = np.random.randn(m*p, r)
                W_try = np.random.randn(p*n, r)
                
                # Iteratively refine using ALS (Alternating Least Squares)
                for iteration in range(200):
                    if time.time() - start_t > time_limit:
                        break
                    
                    # Update U: for each i, U[i,:] should satisfy
                    # sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
                    # Fix V and W, solve for U[i,:]
                    # T[i,j,k] = sum_r U[i,r] * (V[j,r] * W[k,r])
                    # Let M[i,j,k] = T[i,j,k], and for fixed i:
                    # M[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                    # This is: vec(M[i,:,:]) = sum_r U[i,r] * (V[:,r] * W[:,r])
                    # So U[i,:] is the least-squares solution to:
                    # (V[:,r] * W[:,r])_j,k * U[i,r] = M[i,j,k]
                    
                    # Build the design matrix for each i
                    # D[j,k,r] = V[j,r] * W[k,r]
                    D = np.einsum('jr,kr->jkr', V_try, W_try)  # (m*p, p*n, r)
                    D_flat = D.reshape(m*p * p*n, r)
                    
                    for i in range(n*m):
                        target = T[i, :, :].flatten()
                        # Solve D_flat @ U[i,:] = target
                        U_try[i, :], residuals, rank_val, sv = np.linalg.lstsq(D_flat, target, rcond=None)
                    
                    # Similarly update V and W
                    # For V[j,:]: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                    # Fix U and W: T[i,j,k] = sum_r V[j,r] * (U[i,r] * W[k,r])
                    D2 = np.einsum('ir,kr->ikr', U_try, W_try)  # (n*m, p*n, r)
                    D2_flat = D2.reshape(n*m * p*n, r)
                    for j in range(m*p):
                        target = T[:, j, :].reshape(n*m * p*n)
                        V_try[j, :], residuals, rank_val, sv = np.linalg.lstsq(D2_flat, target, rcond=None)
                    
                    # For W[k,:]: T[i,j,k] = sum_r W[k,r] * (U[i,r] * V[j,r])
                    D3 = np.einsum('ir,jr->ijr', U_try, V_try)  # (n*m, m*p, r)
                    D3_flat = D3.reshape(n*m * m*p, r)
                    for k in range(p*n):
                        target = T[:, :, k].reshape(n*m * m*p)
                        W_try[k, :], residuals, rank_val, sv = np.linalg.lstsq(D3_flat, target, rcond=None)
                    
                    # Check convergence
                    T_recon = np.einsum('ir,jr,kr->ijk', U_try, V_try, W_try)
                    err = np.linalg.norm(T_recon - T)
                    if err < 1e-10:
                        # Found a valid decomposition
                        # Round to nearest integer/half-integer
                        U_rounded = np.round(U_try * 2) / 2
                        V_rounded = np.round(V_try * 2) / 2
                        W_rounded = np.round(W_try * 2) / 2
                        
                        # Verify rounded version
                        T_rounded = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                        if np.allclose(T_rounded, T, atol=1e-10):
                            # Convert to lists
                            U_list = U_rounded.tolist()
                            V_list = V_rounded.tolist()
                            W_list = W_rounded.tolist()
                            best_valid = (U_list, V_list, W_list, r)
                            break
                        
                        # If rounding doesn't work, try with integers
                        U_rounded = np.round(U_try)
                        V_rounded = np.round(V_try)
                        W_rounded = np.round(W_try)
                        T_rounded = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                        if np.allclose(T_rounded, T, atol=1e-10):
                            U_list = U_rounded.tolist()
                            V_list = V_rounded.tolist()
                            W_list = W_rounded.tolist()
                            best_valid = (U_list, V_list, W_list, r)
                            break
                        break
                    if err < 1e-6:
                        # Close, try rounding
                        U_rounded = np.round(U_try * 2) / 2
                        V_rounded = np.round(V_try * 2) / 2
                        W_rounded = np.round(W_try * 2) / 2
                        T_rounded = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                        if np.allclose(T_rounded, T, atol=1e-8):
                            U_list = U_rounded.tolist()
                            V_list = V_rounded.tolist()
                            W_list = W_rounded.tolist()
                            best_valid = (U_list, V_list, W_list, r)
                            break
                        break
                
                if best_valid is not None:
                    break
            
            if best_valid is not None:
                return best_valid
        
        return None
    
    # Try to reduce rank
    reduced = try_reduce_rank(U, V, W, max_rank=27, time_limit=110)
    
    if reduced is not None:
        U_list, V_list, W_list, r = reduced
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to the original rank-28 decomposition
    return {"U": U, "V": V, "W": W}
