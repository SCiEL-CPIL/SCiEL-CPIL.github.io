"""Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

def construct():
    """Find a low-rank exact tensor decomposition for (3,3,8) matrix multiplication.
    
    The multiplication tensor T has T[i,j,k] = 1 if the (i,j) entry of A multiplies 
    the (j,k) entry of B to contribute to C[i,k]. In the standard indexing:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We need U (n*m x rank), V (m*p x rank), W (p*n x rank) such that
    for each rank element c: U[:,c] ⊗ V[:,c] ⊗ W[:,c] sums to T.
    
    Indexing convention:
    - U indexed by (i*m+j), row i*m+j
    - V indexed by (j*p+k), row j*p+k  
    - W indexed by (k*n+i), row k*n+i
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    
    # The multiplication tensor: T[(i,j), (j,k), (k,i)] = 1 for valid i,j,k
    # In our indexing:
    # U row = i*m + j
    # V row = j*p + k
    # W row = k*n + i
    # So the tensor is a 3-way tensor of size (nm, mp, pn)
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Schoolbook rank = n*m*p = 72
    schoolbook_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # Strategy: Use the structure that m=3 is small
    # For each j (0,1,2), the "slice" involves U rows with that j, 
    # V rows with that j, and W rows spanning all k,i
    
    # Let's try a structured search:
    # The key insight: for fixed j, we have n*p = 3*8 = 24 terms
    # Total schoolbook: 3 * 24 = 72
    
    # Can we share terms between different j values?
    # A rank-1 term is (u_vec, v_vec, w_vec) where:
    #   u_vec is nonzero only at positions (i*m+j) for some i
    #   v_vec is nonzero only at positions (j'*p+k) for some j',k
    #   w_vec is nonzero only at positions (k'*n+i') for some k',i'
    
    # For the term to contribute to T, we need consistency:
    # If u has support at (i,m+j), v at (j',p+k), w at (k',n+i')
    # Then it contributes to T[i*m+j, j'*p+k, k'*n+i']
    # This should equal 1 only if j=j' and k=k' and i=i'
    
    # So a valid rank-1 term must have:
    # u nonzero at exactly one (i,j) position
    # v nonzero at exactly one (j,k) position  
    # w nonzero at exactly one (k,i) position
    # AND j must be consistent across all three
    
    # This means each rank-1 term is "pinned" to a specific j value
    # So we can't easily share between different j values...
    
    # UNLESS we allow u, v, w to have multiple nonzeros!
    # A rank-1 term CAN have u with multiple nonzeros, as long as
    # the outer product u ⊗ v ⊗ w matches a subset of T entries
    
    # Let me try: allow u to be nonzero at multiple (i,j) positions,
    # v at multiple (j,k) positions, w at multiple (k,i) positions
    # such that u[i*m+j] * v[j'*p+k] * w[k'*n+i'] = 1 
    # only when j=j'=... and the indices are consistent
    
    # Actually, let's just do a randomized search for a decomposition
    # with rank < 72
    
    deadline = time.time() + 120  # 120 second budget
    
    # First, let's try a systematic approach:
    # Since m=3, group by j. For each j, we need to decompose the 
    # "j-slice" which is a rank-1 outer product structure.
    
    # For fixed j: the slice is sum over i,k of e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # This is a tensor of size (n, p, p*n) restricted to j
    # Actually it's: for each (i,k), one entry at (i*m+j, j*p+k, k*n+i)
    
    # The j-slice tensor has n*p = 24 nonzero entries, all 1
    # Can we decompose this 24-entry tensor with fewer than 24 rank-1 terms?
    
    # The j-slice is: S_j = sum_{i=0}^{n-1} sum_{k=0}^{p-1} e_i ⊗ e_k ⊗ f_{k,i}
    # where f_{k,i} is the vector with 1 at position k*n+i
    
    # This is actually a "matrix multiplication" like structure but smaller
    # S_j[i, k, l] = 1 if l = k*n + i, else 0
    # So S_j is the "identity-like" tensor mapping (i,k) to position k*n+i
    
    # This tensor S_j has a natural rank: it's like an embedding, so rank = n*p = 24
    # But can we do better? S_j[i,k,l] = delta_{l, k*n+i}
    # This is a 3-way tensor. Its rank... 
    
    # Let me think differently. Let's just try random searches.
    
    # Approach: random rank-r decomposition, check if it matches T
    # Use integer/half-integer entries for efficiency
    
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Start with schoolbook as baseline
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank by searching
    # Strategy: for each candidate rank r < 72, try random initializations
    # and verify exact match
    
    # A faster approach: use the fact that entries should be small integers or half-integers
    # Try ranks from 65 down to some limit
    
    # Actually, let's try a smarter approach:
    # The tensor T has a specific structure. Let's see if we can find
    # a decomposition where some rank-1 terms cover multiple entries.
    
    # Key: a rank-1 term (u, v, w) contributes u[i*m+j] * v[j'*p+k] * w[k'*n+i']
    # to T[i*m+j, j'*p+k, k'*n+i']
    # For this to be valid, we need j = j' (the middle index must match)
    # AND the contribution should be 1 at valid positions, 0 elsewhere
    
    # So u, v, w must be "compatible" in their j-index
    
    # Let's try: for a given j, allow u to have support on multiple i values,
    # v on multiple k values, w on multiple (k,i) values
    # such that u[i] * v[k] * w[k,i] = 1 for all (i,k) in the support
    
    # This means w[k,i] = 1/(u[i]*v[k]) for supported (i,k)
    # And w must be zero at unsupported (k,i) positions
    
    # So for a fixed j, if we choose u (length n) and v (length p) with 
    # u[i]*v[k] != 0, then w is determined at positions (k,i) for supported (i,k)
    # and must be 0 elsewhere
    
    # The number of rank-1 terms needed for the j-slice is at least 
    # the "rank" of the pattern of nonzero (i,k) pairs
    
    # For the full j-slice (all i,k), we need u and v to be nonzero everywhere
    # (or use multiple terms to cover different subsets)
    
    # If we use one term with u and v all nonzero, that covers all n*p entries
    # But w would need to be nonzero at all (k,i) positions, which is fine
    # So ONE rank-1 term can cover the entire j-slice!
    
    # Wait, that can't be right. Let me check:
    # Term: u = [u_0, u_1, u_2], v = [v_0, ..., v_7], w[k*n+i] = 1/(u_i * v_k)
    # Then the contribution to T at position (i*m+j, j*p+k, k'*n+i') is:
    #   u_i * v_k * w[k'*n+i']
    # For k'=k, i'=i: u_i * v_k * (1/(u_i*v_k)) = 1 ✓
    # For k'!=k or i'!=i: u_i * v_k * w[k'*n+i']
    #   If (k',i') is in the support (all are, since u,v nonzero everywhere):
    #   u_i * v_k * (1/(u_{i'}*v_{k'})) which is generally NOT 0!
    
    # So this doesn't work - it creates spurious nonzero entries.
    
    # The issue: w[k'*n+i'] is nonzero for ALL (k',i'), so we get cross-terms.
    # We need w to be zero at positions where (k',i') != (k,i) for the given (i,k)
    # But a single w vector can't be zero at all other positions unless it's a unit vector
    
    # So we're back to needing one term per (i,k) pair, i.e., n*p per j-slice
    # Total: m * n * p = 72
    
    # UNLESS we allow the terms to have overlapping supports in a controlled way
    # such that cross-terms cancel out!
    
    # This is the key insight: use cancellation!
    # Multiple rank-1 terms can have overlapping (i,k) supports, and their
    # cross-terms can cancel if we choose coefficients carefully
    
    # Let's try: for a fixed j, use r_j terms. Each term t has:
    #   u_t[i] for i=0..2
    #   v_t[k] for k=0..7  
    #   w_t[k*n+i] = some value
    # Sum over t: u_t[i] * v_t[k] * w_t[k'*n+i'] should equal:
    #   1 if k'=k and i'=i, else 0
    
    # This is a system of equations. For the diagonal (k'=k, i'=i):
    #   sum_t u_t[i] * v_t[k] * w_t[k*n+i] = 1
    # For off-diagonal (k'!=k or i'!=i):
    #   sum_t u_t[i] * v_t[k] * w_t[k'*n+i'] = 0
    
    # Let's try small r_j. Start with r_j = 1:
    # u_0[i] * v_0[k] * w_0[k*n+i] = 1 for all (i,k)
    # u_0[i] * v_0[k] * w_0[k'*n+i'] = 0 for (k',i') != (k,i)
    # From the first: w_0[k*n+i] = 1/(u_0[i]*v_0[k])
    # From the second: w_0[k'*n+i'] = 0 for (k',i') != (k,i)
    # But w_0[k*n+i] is nonzero for all (i,k), and w_0[k'*n+i'] = 0 for others
    # Contradiction unless n*p = 1
    
    # r_j = 2: more variables, might work
    # This is getting complex. Let me just do a numerical search.
    
    # Better approach: since the problem is small, let's use a randomized
    # search with exact verification
    
    # For each candidate rank r, try random decompositions
    # Use the structure: group terms by j-index
    
    # Actually, let me try a completely different approach:
    # Use known results or try to find a decomposition with rank < 72
    
    # For (3,3,8), let me try rank 60, 54, 48, etc.
    
    # I'll use a randomized approach: initialize U, V, W randomly,
    # then try to solve for one factor given the others
    
    # Given U and V, can we find W?
    # The constraint is: sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T
    # This is: for each (a,b): sum_c U[a,c] * V[b,c] * W[?,c] ... 
    # Actually it's a 3-way tensor equality, which is hard to solve directly
    
    # Let me try: fix U and V, then for each row of W, solve linear system
    # W[w_idx, c] appears in T[a,b,c] as U[a,c]*V[b,c]*W[w_idx,c]
    # For fixed w_idx, the equation is:
    #   sum_c U[a,c] * V[b,c] * W[w_idx,c] = T[a,b,w_idx] for all (a,b)
    # This is: (U^T * V) * W[w_idx,:] = T[:, :, w_idx] flattened
    # Where (U^T * V) is the "Khatri-Rao" product... 
    
    # Actually, let me define M_c = U[:,c] ⊗ V[:,c] (outer product, size nm*mp)
    # Then T_flat = sum_c M_c * W[:,c] (matrix product)
    # So [M_0, M_1, ..., M_{r-1}] * W^T = T_flat
    # This is a linear system! If we fix U and V, W is determined (if consistent)
    
    # So: random U, V of shape (nm, r) and (mp, r), form the Khatri-Rao matrix
    # K = [U[:,0]⊗V[:,0], U[:,1]⊗V[:,1], ..., U[:,r-1]⊗V[:,r-1]] of shape (nm*mp, r)
    # Then solve K * W^T = T_flat for W
    
    # If the solution W has small integer/half-integer entries, we're done!
    
    T_flat = T.reshape(nm * mp, pn)  # shape (216, 24)
    
    # Try various ranks with improved search strategy
    def try_rank(target_rank, deadline):
        """Try to find a decomposition of given rank using alternating least squares."""
        T_flat = T.reshape(nm * mp, pn)  # shape (216, 24)
        
        # Multiple random restarts with alternating least squares
        for attempt in range(20):
            if time.time() > deadline - 30:
                return None
            
            # Initialize with random values from a continuous distribution
            # then project to half-integers
            np.random.seed(attempt * 12345 + 67890)
            U_rand = np.random.randn(nm, target_rank)
            V_rand = np.random.randn(mp, target_rank)
            W_rand = np.random.randn(pn, target_rank)
            
            # Alternating least squares iterations
            for it in range(50):
                if time.time() > deadline - 30:
                    return None
                
                # Update W: for each column w_idx of W, solve
                # K * W[:, w_idx] = T_flat[:, w_idx]
                # where K[:, c] = outer(U[:, c], V[:, c])
                K = np.zeros((nm * mp, target_rank))
                for c in range(target_rank):
                    K[:, c] = np.outer(U_rand[:, c], V_rand[:, c]).flatten()
                
                W_sol, res, rk, sv = np.linalg.lstsq(K, T_flat, rcond=None)
                W_rand = W_sol.T  # shape (pn, target_rank)
                
                # Update U: for each row a of U, solve
                # sum_c U[a, c] * V[:, c] ⊗ W[:, c] = T[a, :, :] flattened
                # This is: for fixed a, the equation is:
                # sum_c U[a, c] * (V[:, c] ⊗ W[:, c]) = T_flat[a, :]
                # where T_flat[a, :] is a vector of length mp*pn = 24*24 = 576
                # And (V[:, c] ⊗ W[:, c]) is a vector of length 576
                # So: [V[:,0]⊗W[:,0], V[:,1]⊗W[:,1], ...] @ U[a, :] = T_flat[a, :]
                # This is a system of 576 equations in target_rank unknowns
                KW = np.zeros((mp * pn, target_rank))
                for c in range(target_rank):
                    KW[:, c] = np.outer(V_rand[:, c], W_rand[:, c]).flatten()
                
                # T_flat needs to be reshaped: T[a, b, w] where b indexes (j,k) and w indexes (k',i')
                # T_flat currently has shape (nm*mp, pn), so T_flat[a*mp+b, w] = T[a, b, w]
                # For fixed a, we need T[a, :, :] which is T_flat[a*mp:(a+1)*mp, :] reshaped to (mp*pn,)
                U_sol = np.zeros((nm, target_rank))
                for a in range(nm):
                    rhs = T_flat[a*mp:(a+1)*mp, :].flatten()
                    U_sol[a, :], _, _, _ = np.linalg.lstsq(KW, rhs, rcond=None)
                U_rand = U_sol
                
                # Update V: for each row b of V, solve
                # sum_c V[b, c] * U[:, c] ⊗ W[:, c] = T[:, b, :] flattened
                UW = np.zeros((nm * pn, target_rank))
                for c in range(target_rank):
                    UW[:, c] = np.outer(U_rand[:, c], W_rand[:, c]).flatten()
                
                V_sol = np.zeros((mp, target_rank))
                for b in range(mp):
                    # T[:, b, :] = T_flat[?, b*? + ?, :]
                    # T_flat index: a*mp + b, where a indexes U rows and b indexes V rows
                    # For fixed b, we need all a: T_flat[a*mp + b, :] for a in range(nm)
                    # This is not contiguous, so extract manually
                    rhs_parts = []
                    for a in range(nm):
                        rhs_parts.append(T_flat[a*mp + b, :])
                    rhs = np.concatenate(rhs_parts)  # shape (nm*pn,)
                    V_sol[b, :], _, _, _ = np.linalg.lstsq(UW, rhs, rcond=None)
                V_rand = V_sol
            
            # After ALS, check if we have a good solution
            K = np.zeros((nm * mp, target_rank))
            for c in range(target_rank):
                K[:, c] = np.outer(U_rand[:, c], V_rand[:, c]).flatten()
            W_sol, res, rk, sv = np.linalg.lstsq(K, T_flat, rcond=None)
            W_rand = W_sol.T
            
            residual = np.abs(K @ W_sol - T_flat).max()
            if residual < 1e-8:
                # Check if entries are close to half-integers
                U_check = U_rand * 2
                V_check = V_rand * 2
                W_check = W_rand * 2
                if (np.allclose(U_check, np.round(U_check), atol=1e-6) and
                    np.allclose(V_check, np.round(V_check), atol=1e-6) and
                    np.allclose(W_check, np.round(W_check), atol=1e-6)):
                    U_final = np.round(U_check) / 2.0
                    V_final = np.round(V_check) / 2.0
                    W_final = np.round(W_check) / 2.0
                    # Verify exact solution
                    K_final = np.zeros((nm * mp, target_rank))
                    for c in range(target_rank):
                        K_final[:, c] = np.outer(U_final[:, c], V_final[:, c]).flatten()
                    if np.abs(K_final @ W_final.T - T_flat).max() < 1e-10:
                        return (U_final, V_final, W_final)
        
        return None
    
    # Try ranks from 71 down to 40
    for target_rank in range(71, 39, -1):
        if time.time() > deadline - 30:
            break
        
        result = try_rank(target_rank, deadline)
        if result is not None:
            best_rank = target_rank
            best_decomp = result
            break
    
    # Safety: if no improvement found, return schoolbook
    if best_rank >= schoolbook_rank:
        U_out, V_out, W_out = best_decomp
    else:
        U_out, V_out, W_out = best_decomp
    
    # Verify the solution is valid before returning
    U_np = np.array(U_out)
    V_np = np.array(V_out)
    W_np = np.array(W_out)
    rank = U_np.shape[1]
    
    # Quick verification
    K = np.zeros((nm * mp, rank))
    for c in range(rank):
        K[:, c] = np.outer(U_np[:, c], V_np[:, c]).flatten()
    residual = np.abs(K @ W_np.T - T_flat).max()
    
    if residual > 1e-8:
        # Fall back to schoolbook
        U_out, V_out, W_out = U_sb, V_sb, W_sb
        rank = schoolbook_rank
    
    U_list = [[float(x) for x in row] for row in U_out]
    V_list = [[float(x) for x in row] for row in V_out]
    W_list = [[float(x) for x in row] for row in W_out]
    
    return {"U": U_list, "V": V_list, "W": W_list}
