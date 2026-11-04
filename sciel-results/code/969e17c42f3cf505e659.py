import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i
    # C[0,0]: i=0,k=0 -> W[0]; C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]; C[1,1]: i=1,k=1 -> W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _block_outer_decomp():
    """
    Decompose the outer 2x2 * 2x3 -> 2x3 block multiplication.
    
    Returns a list of (U_coeff, V_coeff, W_coeff) for the outer structure.
    Each entry describes how to combine the 2x2 block entries.
    
    For the 2x2 * 2x2 part (cols 0,1): Strassen (7 terms)
    For the 2x2 * 1 part (col 2): schoolbook (4 terms)
    
    Returns: list of dicts with keys 'U', 'V', 'W' where each is a dict
    mapping (row_block, col_block) -> coefficient
    """
    sU, sV, sW = _strassen_2x2()
    
    terms = []
    
    # 7 Strassen terms for cols 0,1
    for t in range(7):
        U_dict = {}
        V_dict = {}
        W_dict = {}
        for a in range(2):
            for b in range(2):
                coeff = sU[a*2+b, t]
                if coeff != 0:
                    U_dict[(a,b)] = U_dict.get((a,b), 0) + coeff
        for b in range(2):
            for c in range(2):
                coeff = sV[b*2+c, t]
                if coeff != 0:
                    V_dict[(b,c)] = V_dict.get((b,c), 0) + coeff
        for a in range(2):
            for c in range(2):
                coeff = sW[c*2+a, t]
                if coeff != 0:
                    W_dict[(a,c)] = W_dict.get((a,c), 0) + coeff
        terms.append({'U': U_dict, 'V': V_dict, 'W': W_dict})
    
    # 4 schoolbook terms for col 2
    for a in range(2):
        for b in range(2):
            U_dict = {(a,b): 1}
            V_dict = {(b,2): 1}
            W_dict = {(a,2): 1}
            terms.append({'U': U_dict, 'V': V_dict, 'W': W_dict})
    
    return terms


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: Use 2x2 block structure with Kronecker product decomposition.
    # A is 4x4: 2x2 blocks of size 2x2. A[ai][bj] where ai,bj in {0,1}
    #   A[i,j] with i=2*ai+ai2, j=2*bj+bj2, ai2,bj2 in {0,1}
    # B is 4x6: 2x3 blocks of size 2x2. B[bj][ck] where bj in {0,1}, ck in {0,1,2}
    #   B[j,k] with j=2*bj+bj2, k=2*ck+ck2, bj2,ck2 in {0,1}
    # C is 4x6: 2x3 blocks of size 2x2. C[ai][ck] where ai in {0,1}, ck in {0,1,2}
    #   C[i,k] with i=2*ai+ai2, k=2*ck+ck2, ai2,ck2 in {0,1}
    #
    # C[ai][ck] = sum_bj A[ai][bj] * B[bj][ck]  (block matrix multiplication)
    # Each block product is 2x2 * 2x2 -> 2x2, decomposed via Strassen (rank 7).
    #
    # Outer structure: 2x2 * 2x3 -> 2x3 block multiplication
    # Use 11 outer terms: 7 (Strassen for cols 0,1) + 4 (schoolbook for col 2)
    # Each outer term involves one 2x2*2x2 block product (rank 7)
    # Total rank: 11 * 7 = 77
    
    outer_terms = _block_outer_decomp()
    sU, sV, sW = _strassen_2x2()
    
    total_rank = len(outer_terms) * 7
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    for ot_idx, outer in enumerate(outer_terms):
        for t in range(7):
            col = ot_idx * 7 + t
            
            # U part: for each (ai, bj) block position
            for (ai, bj), coeff_outer in outer['U'].items():
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff_inner = sU[ai2*2 + bj2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            # A[i,j] where i=2*ai+ai2, j=2*bj+bj2
                            u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                            U[u_row, col] += coeff_outer * coeff_inner
            
            # V part: for each (bj, ck) block position
            for (bj, ck), coeff_outer in outer['V'].items():
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sV[bj2*2 + ck2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            # B[j,k] where j=2*bj+bj2, k=2*ck+ck2
                            v_row = (2*bj + bj2) * 6 + (2*ck + ck2)
                            V[v_row, col] += coeff_outer * coeff_inner
            
            # W part: for each (ai, ck) block position
            for (ai, ck), coeff_outer in outer['W'].items():
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sW[ck2*2 + ai2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            # C[i,k] where i=2*ai+ai2, k=2*ck+ck2
                            # Trace convention: W[k*n+i] = W[k*4+i]
                            w_row = (2*ck + ck2) * 4 + (2*ai + ai2)
                            W[w_row, col] += coeff_outer * coeff_inner
    
    # SVD-based rank reduction: identify linearly dependent rank-1 terms
    # If column j can be expressed as a linear combination of other columns
    # in ALL three factor matrices (U, V, W) simultaneously, we can eliminate it.
    
    def _verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        """Verify that sum_k U[:,k]*V[:,k]*W[:,k] reconstructs the multiplication tensor."""
        # Build the multiplication tensor T[i,j,k] = delta[i,k] * A[i,j]*B[j,k]
        # For matrix mult C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor: T[i,j,k] = 1 if k==i... no, T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k]
        # T[i,j,k] = 1 if i==k... no.
        # C[i,k] = sum_j A[i,j] * B[j,k], so T[i,j,k] = 1 for all i,j,k where the output is C[i,k]
        # Actually T[i,j,k] = delta_{i,k} is wrong.
        # The multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
        # contribute to the (i,k) entry of C. So T[i,j,k] = 1 for all valid i,j,k.
        # Wait, that's not right either. The tensor is defined such that
        # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all i,j,k.
        # 
        # But in the factorization: C[i,k] = sum_r U[i*m+j... no.
        # U has shape (n*m, rank), indexed by (i,j) -> i*m+j
        # V has shape (m*p, rank), indexed by (j,k) -> j*p+k  
        # W has shape (p*n, rank), indexed by (k,i) -> k*n+i (trace convention)
        # 
        # The reconstruction: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal T[i,j,k]
        # where T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
        
        # Let's verify by checking a few entries
        # Actually, let's just compute the full tensor and compare
        # This is expensive for large dimensions, so we'll do a spot check
        
        n, m, p = 4, 4, 6
        # Sample check: verify that the decomposition works for random inputs
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, (n, m))
        B_test = np.random.randint(-5, 6, (m, p))
        C_expected = A_test @ B_test
        
        # Compute via decomposition
        C_computed = np.zeros((n, p), dtype=float)
        U_f = U_mat.astype(float)
        V_f = V_mat.astype(float)
        W_f = W_mat.astype(float)
        
        for r in range(U_f.shape[1]):
            u_col = U_f[:, r]  # (n*m,)
            v_col = V_f[:, r]  # (m*p,)
            w_col = W_f[:, r]  # (p*n,)
            
            # Reshape
            u_mat = u_col.reshape(n, m)  # (n, m)
            v_mat = v_col.reshape(m, p)  # (m, p)
            w_mat = w_col.reshape(p, n)  # (p, n) -> transpose to (n, p)
            
            # Contribution: sum over j of u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # This is trace-like: sum_{i,j,k} u_mat[i,j]*v_mat[j,k]*w_mat[k,i] * delta output
            # Actually the contribution to C[i,k] from term r is:
            # sum_j u_mat[i,j] * v_mat[j,k] * (w_mat[k, i] if we interpret W correctly)
            
            # W row index = k*n + i, so W_f[k*n+i, r] is the coefficient for output (i,k)
            # The term r contributes: for each i,k: sum_j U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            
            # Let's compute directly
            for i in range(n):
                for k in range(p):
                    contrib = 0.0
                    for j in range(m):
                        contrib += U_f[i*m+j, r] * V_f[j*p+k, r] * W_f[k*n+i, r]
                    C_computed[i, k] += contrib
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
        return True
    
    # Try to reduce rank by eliminating linearly dependent columns
    # Strategy: for each column, check if removing it and adjusting others still works
    # Use SVD to find near-dependencies
    
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = total_rank
    
    # Build the "flattened" representation for SVD analysis
    # For each rank-1 term r, we have vectors U[:,r], V[:,r], W[:,r]
    # A column r is redundant if there exist coefficients c_s (s != r) such that:
    # U[:,r] = sum_{s!=r} c_s * U[:,s]
    # V[:,r] = sum_{s!=r} c_s * V[:,s]  
    # W[:,r] = sum_{s!=r} c_s * W[:,s]
    
    # This is a system of 3*(n*m + m*p + p*n) equations in (rank-1) unknowns
    # We can set up a least squares problem
    
    n, m, p = 4, 4, 6
    dim_U = n * m  # 16
    dim_V = m * p  # 24
    dim_W = p * n  # 24
    
    # For each candidate column to remove, solve for coefficients
    # To save time, we'll process in batches and use a deadline guard
    
    removed = set()
    remaining = list(range(total_rank))
    
    # Try removing columns one by one (greedy)
    # For efficiency, we'll limit the number of attempts
    max_attempts = 200
    attempts = 0
    
    for _ in range(max_attempts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        if not remaining:
            break
        
        # Pick a column to try removing (try the last one first, or random)
        # Try each remaining column
        found_removal = False
        for idx in remaining:
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            attempts += 1
            
            # Set up the linear system
            # Other columns
            others = [x for x in remaining if x != idx]
            if not others:
                break
            
            # Build A_matrix: (dim_U + dim_V + dim_W) x len(others)
            A_mat = np.zeros((dim_U + dim_V + dim_W, len(others)))
            b_vec = np.zeros(dim_U + dim_V + dim_W)
            
            for oi, other_idx in enumerate(others):
                A_mat[:dim_U, oi] = current_U[:, other_idx]
                A_mat[dim_U:dim_U+dim_V, oi] = current_V[:, other_idx]
                A_mat[dim_U+dim_V:, oi] = current_W[:, other_idx]
            
            b_vec[:dim_U] = current_U[:, idx]
            b_vec[dim_U:dim_U+dim_V] = current_V[:, idx]
            b_vec[dim_U+dim_V:] = current_W[:, idx]
            
            # Solve least squares
            try:
                coeffs, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
            
            # Check if the solution is exact (residuals should be ~0)
            # residuals is only populated if len(others) < (dim_U+dim_V+dim_W)
            # which is always true here since len(others) <= 76 < 64
            # Actually 76 > 64, so we need to check differently
            
            # Compute residual directly
            residual = b_vec - A_mat @ coeffs
            if np.max(np.abs(residual)) < 1e-8:
                # Column idx can be eliminated
                # Update: current_U[:, others] += current_U[:, idx][:, None] * coeffs[None, :]
                # Wait, we need to add the contribution of column idx to the other columns
                # new_U[:, other_j] = U[:, other_j] + coeffs[j] * U[:, idx]
                # But we're removing column idx, so we need to redistribute its contribution
                # Actually, the decomposition sum_r U[:,r]*V[:,r]*W[:,r] must be preserved
                # If U[:,idx] = sum_j coeffs[j]*U[:,others[j]], etc., then
                # The term idx contributes: U[:,idx]*V[:,idx]*W[:,idx]
                # = (sum_j c_j U[:,o_j]) * (sum_j c_j V[:,o_j]) * (sum_j c_j W[:,o_j])
                # This is NOT equal to sum_j c_j * U[:,o_j]*V[:,o_j]*W[:,o_j]
                # So simple linear dependency in the factor matrices does NOT preserve the tensor!
                
                # The correct condition is more subtle. The rank-1 terms are not independent
                # in the way I described. Let me reconsider.
                # 
                # Actually, for a tensor decomposition T = sum_r u_r ⊗ v_r ⊗ w_r,
                # if we have a linear relation among the u_r's, it doesn't directly
                # imply we can eliminate a term. The tensor rank is not simply
                # the rank of the factor matrices.
                #
                # However, there IS a valid approach: if the set of rank-1 terms
                # has a linear dependency in the tensor space, we can reduce rank.
                # The tensor space has dimension n*m*p = 4*4*6 = 96.
                # If we have more than 96 rank-1 terms, some are dependent.
                # But we have 77 < 96, so they could all be independent.
                #
                # A better approach: use the fact that if we can find a nonzero
                # vector (a, b, c) such that sum_r a_r * u_r ⊗ v_r ⊗ w_r = 0,
                # then we can modify the decomposition. But this is complex.
                #
                # Let me try a different approach: coordinate descent on the factors.
                # Start with the rank-77 decomposition and try to zero out entries.
                
                pass  # Skip this approach for now
            
            if found_removal:
                break
        
        if not remaining or not found_removal:
            break
    
    # Alternative: try to reduce by finding columns that are exact linear combinations
    # and using a more careful redistribution
    
    # For now, let's try a simpler approach: check if any two columns are identical
    # or negatives, which would be an obvious redundancy
    
    # Actually, let me try the SVD approach on the "flattened tensor"
    # The tensor T has entries T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # We want to find the minimal number of rank-1 terms.
    
    # Build the full tensor from our decomposition
    T = np.zeros((n, m, p))
    for r in range(total_rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # The expected tensor for matrix multiplication is T[i,j,k] = 1 for all i,j,k
    # Wait, no. C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 for all i,j,k? No!
    # T[i,j,k] is the coefficient such that C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k. Yes, that's correct.
    
    # Verify
    expected_T = np.ones((n, m, p))
    if not np.allclose(T, expected_T, atol=1e-9):
        # Decomposition is invalid, return as-is
        pass
    
    # Try to reduce rank by checking if we can express the tensor with fewer terms
    # This is the tensor rank problem, which is NP-hard in general.
    # For our structured decomposition, let's try a greedy approach.
    
    # Greedy: try removing each term and see if the remaining terms can be adjusted
    # to still produce the correct tensor. This is equivalent to checking if the
    # tensor lies in the span of the remaining rank-1 tensors.
    
    # For 77 terms in a 96-dimensional space, this is feasible.
    # For each term r, check if T - sum_{s!=r} u_s⊗v_s⊗w_s is a rank-1 tensor.
    
    # A tensor X is rank-1 if X[i,j,k] = a[i]*b[j]*c[k] for some vectors a,b,c.
    # We can check this by flattening and checking rank.
    
    # This is expensive: 77 * (96^3) operations. Let's try it with a time limit.
    
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    active = list(range(total_rank))
    
    for _ in range(len(active)):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        if not active:
            break
        
        found = False
        for pos in range(len(active)):
            if time.time() - START_TIME > TIME_LIMIT - 15:
                break
            
            r = active[pos]
            # Compute T_remaining = T - sum_{s in active, s != r} u_s⊗v_s⊗w_s
            T_rem = expected_T.copy()
            for s in active:
                if s == r:
                    continue
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_rem[i,j,k] -= U_work[i*m+j, s] * V_work[j*p+k, s] * W_work[k*n+i, s]
            
            # Check if T_rem is rank-1
            # Flatten to (n, m*p) and check rank
            T_flat = T_rem.reshape(n, m*p)
            try:
                rank_check = np.linalg.matrix_rank(T_flat, tol=1e-8)
            except:
                rank_check = n
            
            if rank_check == 1:
                # T_rem is rank-1, so we can eliminate term r
                # But we need to find the new rank-1 term that replaces the removed one
                # Actually, T_rem IS the new rank-1 term, so we just remove r
                active.remove(r)
                found = True
                break
        
        if not found:
            break
    
    # Build final matrices with only active terms
    final_rank = len(active)
    U_final = np.zeros((n*m, final_rank))
    V_final = np.zeros((m*p, final_rank))
    W_final = np.zeros((p*n, final_rank))
    
    for new_idx, old_idx in enumerate(active):
        U_final[:, new_idx] = U[:, old_idx]
        V_final[:, new_idx] = V[:, old_idx]
        W_final[:, new_idx] = W[:, old_idx]
    
    # Verify the final decomposition
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        U_out = U_final
        V_out = V_final
        W_out = W_final
    else:
        # Fallback to original
        U_out = U
        V_out = V
        W_out = W
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U_out), "V": to_list(V_out), "W": to_list(W_out)}
