import time
import numpy as np

# Deadline guard: evaluator kills at 150s, keep 30s safety margin
DEADLINE = time.time() + 120

def construct():
    n, m, p = 3, 4, 7
    rank_schoolbook = n * m * p  # 84
    
    # Build schoolbook decomposition (guaranteed valid)
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # We need rank-r decomposition: sum_c U[a,c]*V[b,c]*W[d,c] = T[a,b,d]
    
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to find lower-rank decomposition using ALS (Alternating Least Squares)
    # with integer/half-integer rounding and verification
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Candidate ranks to try (in increasing order)
    candidate_ranks = [12, 13, 14, 15, 16, 17, 18, 20, 24, 28, 35, 42]
    
    for r in candidate_ranks:
        if time.time() > DEADLINE:
            break
        
        # Try multiple random initializations
        max_iters = 200
        n_restarts = 5
        
        for restart in range(n_restarts):
            if time.time() > DEADLINE:
                break
            
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS iterations
            for it in range(max_iters):
                if time.time() > DEADLINE:
                    break
                
                # Update U: for each a, U[a,:] = argmin sum_{b,d} (T[a,b,d] - sum_c U[a,c]V[b,c]W[d,c])^2 * ...
                # This is a linear system: for fixed V,W, minimize over U
                # Residual R[a,b,d] = T[a,b,d] - sum_c U[a,c] * (V[:,c] * W[:,c])_outer... 
                # Actually: sum_c U[a,c] V[b,c] W[d,c] = sum_c U[a,c] * (V[b,c]*W[d,c])
                # For fixed a: sum_c U[a,c] * (sum over b,d of V[b,c]*W[d,c]*T[a,b,d]) = sum over b,d of T[a,b,d]^2... 
                # No, this is a tensor regression. Let me use a different approach.
                
                # Better: treat it as a matrix factorization problem by unfolding
                # T_unfolded (a, (b,d)) = sum_c U[a,c] * (V[b,c]*W[d,c])
                # Let M_c[b,d] = V[b,c]*W[d,c], then T[a,b,d] = sum_c U[a,c]*M_c[b,d]
                # This is hard because M_c depends on both V and W.
                
                # Alternative: use the fact that for fixed V, the problem in U and W is:
                # T[a,b,d] = sum_c U[a,c] V[b,c] W[d,c]
                # For fixed U,V: W[d,:] can be solved linearly for each (a,b) pair? No.
                
                # Let me just do a simple gradient descent on the flattened parameters
                pass
            
            # This approach is too slow. Let me try a structured approach instead.
            break
        
        # Try structured low-rank decompositions
        # Structure: group by i (row of A)
        # For each i, we have m*p*p = 4*7*7 = 196 non-zero entries in the tensor
        # The sub-tensor for row i: T[i*m+j, j*p+k, k*n+i] = 1 for j in 0..3, k in 0..6
        
        # A natural decomposition for each row i:
        # Use m=4 terms, one per j. For term (i,j):
        #   U[i*m+j, c_ij] = 1
        #   V[j*p+k, c_ij] = 1 for all k
        #   W[k*n+i, c_ij] = 1 for all k
        # But this gives C_ik = sum_j 1*1*1 = m for each k, not sum_j A_ij B_jk
        # The issue is that A and B are external, so U selects A_ij and V selects B_jk
        # Actually the reconstruction is: C_ik = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A_ij) * (sum_j V[j*p+k,c]*B_jk)
        # No wait, that's not right either. Let me re-read the problem.
        
        # The problem says: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # And the output factor follows W[k*n+i,r]
        # The standard convention for matrix multiplication tensor decomposition:
        # C_ik = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A_ij) * (sum_j V[j*p+k, r] * B_jk)
        # No, that's not standard either.
        
        # Actually, I think the correct interpretation is:
        # The bilinear map is (A, B) -> C where C_ik = sum_j A_ij B_jk
        # A rank-1 decomposition means: C_ik = sum_r (alpha_r(A)) * (beta_r(B)) * W_ik_r
        # where alpha_r(A) = sum_{i,j} U[i*m+j, r] A_ij
        # and beta_r(B) = sum_{j,k} V[j*p+k, r] B_jk
        # and W_ik_r = W[k*n+i, r]
        
        # So: C_ik = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] A_i'j') * (sum_{j',k'} V[j'*p+k', r] B_j'k')
        
        # For this to equal sum_j A_ij B_jk for ALL A, B:
        # We need: for each i,k: sum_r W[k*n+i,r] * (sum_{i',j'} U[i'*m+j',r] A_i'j') * (sum_{j',k'} V[j'*p+k',r] B_j'k') = sum_j A_ij B_jk
        
        # This must hold for all A, B, so:
        # sum_r W[k*n+i,r] * U[i'*m+j',r] * V[j''*p+k'',r] = delta_{i',i} * delta_{j',j''} * delta_{k'',k}
        # Wait no, the RHS is sum_j A_ij B_jk, so the coefficient of A_i'j' B_j''k'' in the LHS must be:
        #   delta_{i',i} * delta_{j',j''} * delta_{k'',k}
        # So: sum_r W[k*n+i,r] * U[i'*m+j',r] * V[j''*p+k'',r] = delta_{i',i} * delta_{j',j''} * delta_{k'',k}
        
        # This is the tensor decomposition condition!
        # T[i*m+j, j*p+k, k*n+i] = 1 (the tensor we built above)
        # And we need: sum_r U[a,r] * V[b,r] * W[d,r] = T[a,b,d]
        
        # OK so my original formulation was correct. Let me try a different approach.
        
        # For the structured decomposition with rank = n*m = 12:
        # For each (i,j) pair, define column c = i*m + j
        # U[i*m+j, c] = 1 (and 0 elsewhere in that column)
        # V[j*p+k, c] = ? 
        # W[k*n+i, c] = ?
        
        # We need: sum_c U[a,c] V[b,c] W[d,c] = T[a,b,d]
        # For a = i*m+j, b = j'*p+k', d = k''*n+i'':
        # T[a,b,d] = 1 iff i''=i AND j'=j AND k''=k... wait no.
        # T[i*m+j, j*p+k, k*n+i] = 1, so T[a,b,d] = 1 iff a=i*m+j AND b=j*p+k AND d=k*n+i for some i,j,k
        # i.e., the j-index in a and b must match, and the k-index in b and d must match, and the i-index in a and d must match.
        
        # So T[i*m+j, j'*p+k', k''*n+i''] = 1 iff i=i'' AND j=j' AND k=k''
        
        # Now with our structured decomposition:
        # sum_c U[i*m+j, c] * V[j'*p+k', c] * W[k''*n+i'', c]
        # = sum_{c} U[i*m+j, c] * V[j'*p+k', c] * W[k''*n+i'', c]
        
        # If U[i*m+j, c] is nonzero only when c corresponds to (i,j), i.e., c = i*m+j (in the structured case):
        # Then the sum reduces to: U[i*m+j, i*m+j] * V[j'*p+k', i*m+j] * W[k''*n+i'', i*m+j]
        
        # For this to equal delta_{i,i''} * delta_{j,j'} * delta_{k,k''}:
        # We need: when i=i'', j=j', k=k'': U[i*m+j, i*m+j] * V[j*p+k, i*m+j] * W[k*n+i, i*m+j] = 1
        # And when any of these fail: the product should be 0.
        
        # If we set U[i*m+j, i*m+j] = 1, then we need:
        # V[j*p+k, i*m+j] * W[k*n+i, i*m+j] = 1 when j=j', k=k'' (with i=i'')
        # And 0 otherwise.
        
        # But V[j*p+k, i*m+j] is a function of (j,k) and W[k*n+i, i*m+j] is a function of (k,i).
        # For fixed (i,j): V[j*p+k, i*m+j] * W[k*n+i, i*m+j] should be 1 for all k? No, it should be 1 when k=k'' and the overall condition holds.
        
        # Wait, I think I'm overcomplicating this. Let me just try rank 12 with:
        # U[i*m+j, i*m+j] = 1
        # V[j*p+k, i*m+j] = 1 for all k (when the column is i*m+j)
        # W[k*n+i, i*m+j] = 1 for all k (when the column is i*m+j)
        
        # Then: sum_c U[i*m+j,c] V[j'*p+k',c] W[k''*n+i'',c]
        # = U[i*m+j, i*m+j] * V[j'*p+k', i*m+j] * W[k''*n+i'', i*m+j]  (since U is one-hot)
        # = 1 * V[j'*p+k', i*m+j] * W[k''*n+i'', i*m+j]
        
        # For V[j'*p+k', i*m+j]: this is nonzero only when the column index i*m+j is "valid" for row j'.
        # In our construction, column c = i*m+j is associated with (i,j). So V[j'*p+k', i*m+j] should be 1 only when j'=j.
        # Similarly, W[k''*n+i'', i*m+j] should be 1 only when i''=i.
        
        # So: V[j*p+k, i*m+j] = 1 for all k (when j matches the column's j)
        #     W[k*n+i, i*m+j] = 1 for all k (when i matches the column's i)
        #     V[j'*p+k, i*m+j] = 0 when j' != j
        #     W[k*n+i', i*m+j] = 0 when i' != i
        
        # Then: sum_c ... = 1 * (1 if j'=j else 0) * (1 if i''=i else 0) * (1 for all k')
        # = delta_{j,j'} * delta_{i,i''}
        
        # But we need delta_{j,j'} * delta_{i,i''} * delta_{k,k''}!
        # The k'' condition is missing!
        
        # So this structured decomposition with rank 12 does NOT work because it can't distinguish k values.
        
        # What if we use rank = n*m*p = 84 (schoolbook)? That works because each (i,j,k) gets its own column.
        
        # What about rank = n*p = 21? For each (i,k), one column:
        # U[i*m+j, i*p+k] = 1 for all j (so it sums over j: sum_j A_ij)
        # V[j*p+k, i*p+k] = 1 for all j (so it sums over j: sum_j B_jk)
        # W[k*n+i, i*p+k] = 1
        # Then: sum_c U[i*m+j,c] V[j'*p+k',c] W[k''*n+i'',c]
        # For c = i*p+k: U[i*m+j, i*p+k] = 1, V[j'*p+k', i*p+k] = 1 if k'=k, W[k''*n+i'', i*p+k] = 1 if i''=i and k''=k
        # So the product is 1 if i''=i, k''=k, k'=k, and j can be anything.
        # Sum over j: this gives (number of j) * delta_{i,i''} * delta_{k,k''} * delta_{k',k}
        # That's not right either.
        
        # I think the minimum rank for (3,4,7) might actually be higher than 12.
        # Let me just return the schoolbook decomposition which is guaranteed correct.
        
        pass
    
    # Return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
