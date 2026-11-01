import time
import numpy as np
from itertools import product

def construct():
    start = time.time()
    n, m, p = 2, 3, 5
    
    # Build the tensor T of shape (n*m, m*p, p*n) = (6, 15, 10)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to find a low-rank decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Strategy: use random initialization + gradient-free optimization
    # Or: use the fact that we can fix U and V and solve for W (or vice versa)
    
    # Let's try a systematic approach:
    # Fix U to have a specific structure, then solve for V and W.
    
    # Alternative: use the "slice" structure.
    # For each V index b = j*p+k, the slice T[:, b, :] is a 6x10 matrix.
    # T[i*m+j', j*p+k, k'*n+i'] = 1 iff j'=j, k'=k, and i'=i.
    # So for fixed b = j*p+k: T[3j, b, 2k] = 1 and T[3j+1, b, 2k+1] = 1, all else 0.
    # This is a 6x10 matrix with exactly 2 non-zero entries.
    
    # The rank of each slice is 2 (if the two non-zeros are in different rows and cols).
    # Actually, it's a matrix with 2 non-zero entries at positions (3j, 2k) and (3j+1, 2k+1).
    # This has rank 2 (unless they're in the same row or column, which they're not).
    
    # So each V-slice has rank 2. The total tensor rank is at least the max slice rank = 2.
    # But that's a lower bound, not useful.
    
    # Let me try a different approach: use the "fiber" structure.
    # For fixed U index a = i*m+j: T[a, :, :] is a 15x10 matrix.
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 iff j'=j, k'=k, i'=i.
    # So for fixed a = i*m+j: T[a, j*p+k, k*n+i] = 1 for all k, 0 elsewhere.
    # That's a 15x10 matrix with 5 non-zero entries at positions (5j+k, 2k+i) for k=0..4.
    # This is a "diagonal-like" pattern. The rank of this matrix is 5 (it's like a permutation
    # matrix embedded in a larger space, with 5 non-zeros in distinct rows and columns).
    
    # Actually, the 5 non-zeros are at (5j, i), (5j+1, 2+i), (5j+2, 4+i), (5j+3, 6+i), (5j+4, 8+i).
    # These are 5 distinct rows and 5 distinct columns, so the matrix has rank 5.
    
    # Similarly, for fixed W index c = k*n+i: T[:, :, c] is a 6x15 matrix.
    # T[i'*m+j, j*p+k', k*n+i] = 1 iff i'=i, k'=k.
    # So for fixed c = k*n+i: T[i*m+j, 5j+k, c] = 1 for all j, 0 elsewhere.
    # That's a 6x15 matrix with 3 non-zero entries at positions (3j, 5j+k) for j=0,1,2.
    # These are 3 distinct rows and 3 distinct columns, so rank 3.
    
    # Lower bound on rank: max over all slices of the slice rank.
    # U-slices: rank 5 each → lower bound 5
    # V-slices: rank 2 each → lower bound 2  
    # W-slices: rank 3 each → lower bound 3
    # So rank ≥ 5. But the true rank is likely much higher.
    
    # Let me try to find a decomposition by solving a linear system.
    # Assume rank r. We have U (6×r), V (15×r), W (10×r).
    # The constraint is: for all (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c].
    
    # This is a bilinear system in (U,V,W). Hard to solve directly.
    
    # Alternative: use the "unfolding" approach.
    # Fix W, then the constraint becomes: for each c, 
    #   sum_r W[c,r] * (U[:,r] ⊗ V[:,r]) = T[:,:,c] (as a 6x15 matrix)
    # This is: T[:,:,c] = sum_r W[c,r] * U[:,r] * V[:,r]^T (outer product)
    # So T[:,:,c] is a sum of r rank-1 matrices, weighted by W[c,r].
    
    # The 6x15 matrix T[:,:,c] has rank 3 (as computed above).
    # So we need r ≥ 3 for this to work (since a sum of r rank-1 matrices has rank ≤ r).
    # But we need this for all c, so r ≥ 3 is a lower bound from this perspective.
    
    # Let me try r = 15, 20, 25, 30 and see which works.
    
    # Actually, let me try a smarter approach. 
    # The tensor has a "product" structure in some sense.
    # Let me try to use the decomposition:
    #   T = sum_{j} (something involving j)
    
    # For each j, define a sub-tensor T_j where only entries with that j are non-zero.
    # T_j has non-zeros at T[i*m+j, j*p+k, k*n+i] for all i,k.
    # That's a 2×5×10 sub-tensor (U rows {3j, 3j+1}, V rows {5j..5j+4}, W cols {0..9}).
    
    # The rank of T_j: it's a 2×5×10 tensor with 10 non-zero entries.
    # T_j[3j, 5j+k, 2k] = 1 and T_j[3j+1, 5j+k, 2k+1] = 1 for k=0..4.
    
    # Can I decompose T_j in rank 5? Let's see:
    # Term k: U[3j] = 1, V[5j+k] = 1, W[2k] = 1 → gives T[3j, 5j+k, 2k] = 1
    # But this also gives T[3j+1, 5j+k, 2k] = 0 (since U[3j+1] = 0 for this term). Good.
    # And T[3j, 5j+k, 2k+1] = 0 (since W[2k+1] = 0). Good.
    # But it doesn't contribute to T[3j+1, 5j+k, 2k+1].
    
    # So I need another term for the "off-diagonal" part:
    # Term k': U[3j+1] = 1, V[5j+k] = 1, W[2k+1] = 1 → gives T[3j+1, 5j+k, 2k+1] = 1
    
    # So T_j can be decomposed in rank 10 (5 terms for the "diagonal" + 5 for the "off-diagonal").
    # Total rank = 3 * 10 = 30 (same as naive).
    
    # Can I do better by sharing terms across different j values?
    # The issue is that different j values use different U rows and V rows, so there's
    # no overlap. The only sharing is in W columns.
    
    # Hmm, let me try a different split. What if I group by k instead?
    # For fixed k, T has non-zeros at T[i*m+j, j*p+k, k*n+i] for all i,j.
    # That's a 6×3×2 sub-tensor (all U rows, V rows {k, 5+k, 10+k}, W cols {2k, 2k+1}).
    
    # T_k[3i+j, 5j+k, 2k+i] = 1 for i∈{0,1}, j∈{0,1,2}.
    # That's 6 non-zero entries in a 6×3×2 tensor.
    
    # Can I decompose T_k in rank 3? 
    # For each j: T_k[3i+j, 5j+k, 2k+i] = 1 for i=0,1.
    # Term (j,i): U[3i+j] = 1, V[5j+k] = 1, W[2k+i] = 1.
    # That's 6 terms per k, total 30. Same as naive.
    
    # I'm stuck at 30 with this approach. Let me try to find a genuinely better decomposition.
    
    # Let me use numerical optimization to find a low-rank decomposition.
    # I'll use a random initialization and iterative refinement.
    
    def try_rank(r, max_iters=200, seed=42):
        """Try to find a rank-r decomposition."""
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Iteratively update one factor at a time (like ALS)
        for it in range(max_iters):
            if time.time() - start > 120:
                return None
            
            # Update U: for each a, solve min_U sum_b,c (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
            # This is: for each a, U[a,:] should minimize ||T[a,:,:] - V^T * diag(W^T) * U[a,:]^T||
            # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed V, W: T[a,:,:] = U[a,:] @ (V * W^T) ... no.
            # T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # So T[a] = U[a] @ M where M[b,c] = sum_r V[b,r]*W[c,r]... no.
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Let M_r[b,c] = V[b,r]*W[c,r] (outer product). Then T[a] = sum_r U[a,r] * M_r.
            # So T[a] is in the span of {M_r}. We need U[a,:] such that sum_r U[a,r]*M_r ≈ T[a].
            # This is a linear least squares: vec(T[a]) = sum_r U[a,r] * vec(M_r)
            # Let A be a (6*15*10) × r matrix where A[:,r] = vec(M_r) = vec(V[:,r] * W[:,r]^T)
            # Then for each a: A @ U[a,:] = vec(T[a,:,:])
            # vec(M_r) = np.kron(W[:,r], V[:,r]) (since M_r = V[:,r] * W[:,r]^T)
            
            # Actually, let me just do a simple gradient step or use numpy's lstsq.
            
            # Update W: for each c, W[c,:] should minimize sum_a,b (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
            # T[a,b,c] = sum_r (U[a,r]*V[b,r]) * W[c,r]
            # Let N_r[a,b] = U[a,r]*V[b,r]. Then T[:,:,c] = sum_r W[c,r] * N_r.
            # vec(T[:,:,c]) = sum_r W[c,r] * vec(N_r) = sum_r W[c,r] * np.kron(V[:,r], U[:,r])
            # Let B be a (6*15) × r matrix where B[:,r] = np.kron(V[:,r], U[:,r])
            # Then for each c: B @ W[c,:] = vec(T[:,:,c])
            
            B = np.zeros((n * m * m * p, r))
            for r_idx in range(r):
                B[:, r_idx] = np.kron(V[:, r_idx], U[:, r_idx])
            
            W_new = np.zeros((p * n, r))
            for c in range(p * n):
                rhs = T[:, :, c].flatten()
                W_new[c, :] = np.linalg.lstsq(B, rhs, rcond=None)[0]
            W = W_new
            
            # Update V: similar
            # T[a,b,c] = sum_r (U[a,r]*W[c,r]) * V[b,r]
            # For fixed U, W: T[a,b,c] = sum_r V[b,r] * (U[a,r]*W[c,r])
            # Let P_r[a,c] = U[a,r]*W[c,r]. Then T[:,:,b]... wait, T is indexed [a,b,c].
            # T[a,b,c] = sum_r V[b,r] * P_r[a,c]
            # For each b: sum_r V[b,r] * P_r = T[:,b,:] (as a 6×10 matrix)
            # vec(T[:,b,:]) = sum_r V[b,r] * vec(P_r) = sum_r V[b,r] * np.kron(W[:,r], U[:,r])
            # Let C_mat be a (6*10) × r matrix where C_mat[:,r] = np.kron(W[:,r], U[:,r])
            # Then for each b: C_mat @ V[b,:] = vec(T[:,b,:])
            
            C_mat = np.zeros((n * m * p * n, r))
            for r_idx in range(r):
                C_mat[:, r_idx] = np.kron(W[:, r_idx], U[:, r_idx])
            
            V_new = np.zeros((m * p, r))
            for b in range(m * p):
                rhs = T[:, b, :].flatten()
                V_new[b, :] = np.linalg.lstsq(C_mat, rhs, rcond=None)[0]
            V = V_new
            
            # Update U: 
            # T[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
            # For each a: T[a,:,:] = sum_r U[a,r] * Q_r where Q_r[b,c] = V[b,r]*W[c,r]
            # vec(T[a,:,:]) = sum_r U[a,r] * vec(Q_r) = sum_r U[a,r] * np.kron(W[:,r], V[:,r])
            # Let D be a (15*10) × r matrix where D[:,r] = np.kron(W[:,r], V[:,r])
            # Then for each a: D @ U[a,:] = vec(T[a,:,:])
            
            D = np.zeros((m * p * p * n, r))
            for r_idx in range(r):
                D[:, r_idx] = np.kron(W[:, r_idx], V[:, r_idx])
            
            U_new = np.zeros((n * m, r))
            for a in range(n * m):
                rhs = T[a, :, :].flatten()
                U_new[a, :] = np.linalg.lstsq(D, rhs, rcond=None)[0]
            U = U_new
        
        # Check error
        T_recon = np.zeros_like(T)
        for r_idx in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        err = np.max(np.abs(T_recon - T))
        return err, U, V, W
    
    # Try increasing ranks
    best_result = None
    for r in [15, 20, 25, 30]:
        if time.time() - start > 110:
            break
        result = try_rank(r)
        if result is not None:
            err, U, V, W = result
            if err < 1e-8:
                best_result = (r, U, V, W)
                break
    
    if best_result is None:
        # Fallback to naive rank-30
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    r, U, V, W = best_result
    
    # Try to round to rational numbers
    # Check if entries are close to integers or simple fractions
    for mat in (U, V, W):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                # Try to round to nearest 0.5
                val = mat[i, j]
                rounded = np.round(val * 2) / 2
                if abs(val - rounded) < 1e-6:
                    mat[i, j] = rounded
                else:
                    # Try rounding to nearest integer
                    rounded_int = np.round(val)
                    if abs(val - rounded_int) < 1e-6:
                        mat[i, j] = rounded_int
    
    # Verify after rounding
    T_recon = np.zeros_like(T)
    for r_idx in range(r):
        T_recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
    err = np.max(np.abs(T_recon - T))
    
    if err > 1e-6:
        # Rounding broke it, use unrounded values
        # Or fall back to naive
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
