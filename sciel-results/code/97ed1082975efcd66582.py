"""
Strategy: Numerical search for low-rank tensor decomposition.
The previous approach used a fixed schoolbook decomposition (rank 54).
This approach attempts to find a rank-50 decomposition by:
1. Initializing with a rank-54 decomposition.
2. Pruning 4 terms to get rank 50.
3. Optimizing the remaining 50 terms to minimize the reconstruction error.
4. Rounding to nearest integers/half-integers.
5. Verifying exactness.
If optimization fails to find an exact solution, it falls back to the known rank-54 solution.
"""

import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Target dimensions
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # 1. Construct the ground truth tensor T
    # T is indexed by (i, j, k) where i in [n], j in [m], k in [p]
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i, j, k] = 1 if this term contributes to the product?
    # No, the tensor T represents the bilinear map.
    # T = sum_{i,j,k} e_i^{(n)} \otimes e_j^{(m)} \otimes e_k^{(p)} ? No.
    # The matrix multiplication tensor M_{n,m,p} is defined such that
    # <M, A \otimes B> = vec(A B)
    # Specifically, the coefficient of A[i,j] B[j,k] in C[i,k] is 1.
    # So the tensor T has T[i, j, k] = 1 for all i, j, k? 
    # Let's check the flattening.
    # U is n*m x rank. Index u = i*m + j.
    # V is m*p x rank. Index v = j*p + k.
    # W is p*n x rank. Index w = k*n + i.
    
    # The reconstruction is:
    # C[i,k] = sum_{r=1}^R sum_{j=1}^m U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want this to equal sum_{j=1}^m A[i,j] B[j,k].
    # This implies that for each r, the term U_r \otimes V_r \otimes W_r
    # must sum up to the identity mapping on the middle index j.
    
    # Actually, the standard definition is:
    # T[i, j, k] = 1 if we consider the tensor in the basis of the product space?
    # Let's just build the target matrix for a specific A and B?
    # No, the decomposition must hold for ALL A and B.
    # This means the tensor T must satisfy:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{j, j'} ... wait.
    
    # Let's look at the bilinear form.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is in R^{n*m} x R^{m*p} x R^{p*n}
    # T[u, v, w] = 1 if u corresponds to (i,j), v to (j,k), w to (k,i) AND the indices match?
    # Specifically, if u = i*m+j, v = j'*p+k, w = k'*n+i'
    # The term contributes to C[i,k] only if i'=i and k'=k.
    # And the product A[i,j] B[j',k] is non-zero in the sum only if j=j'.
    # So T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i) i.e. j in u == j in v and i in u == i in w and k in v == k in w.
    # Otherwise 0.
    
    # Let's verify with rank 1.
    # If R=1, U[i*m+j, 0] = a_{ij}, V[j*p+k, 0] = b_{jk}, W[k*n+i, 0] = c_{ki}
    # C[i,k] = sum_j a_{ij} b_{jk} c_{ki}
    # We want C[i,k] = sum_j A[i,j] B[j,k].
    # This requires c_{ki} = 1 for all k,i? And a_{ij}b_{jk} to cover all terms?
    # No, rank 1 cannot do matrix multiplication unless n=m=p=1.
    
    # The correct target tensor T is:
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and i==i' and k==k'
    # i.e., T is a permutation of the identity in the middle dimension, replicated over i and k.
    # Actually, it's simpler:
    # The tensor T is the sum of n*p rank-1 tensors? No.
    # T = sum_{i=1}^n sum_{k=1}^p e_i^{(n)} \otimes e_k^{(p)} \otimes ...
    
    # Let's just build T explicitly.
    # Dimensions: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # u = i*m + j
                # v = j*p + k
                # w = k*n + i
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # 2. Initialize with Rank 54 (Schoolbook)
    R_init = n * m * p
    U = np.zeros((n*m, R_init))
    V = np.zeros((m*p, R_init))
    W = np.zeros((p*n, R_init))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    # Verify initialization
    def reconstruct(U, V, W):
        # C[i,k] = sum_r sum_j U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # This is expensive to do naively.
        # Let's compute the tensor reconstruction:
        # T_rec[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # We want T_rec to match T.
        # Efficiently: T_rec = sum_r U_r \otimes V_r \otimes W_r
        # We can check the error by computing the sum of squares of (T_rec - T).
        # Or just compute specific slices.
        # Since we need exactness, we can compute the full tensor if it's small enough.
        # n*m=6, m*p=27, p*n=18. Size = 6*27*18 = 2916.
        # R=50.
        # Computation: 50 * 6 * 27 * 18 = 145,800 ops. Very fast.
        
        T_rec = np.zeros_like(T)
        for r in range(U.shape[1]):
            T_rec += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return T_rec

    T_init = reconstruct(U, V, W)
    err_init = np.linalg.norm(T_init - T)
    if not np.allclose(T_init, T, atol=1e-9):
        # Fallback if init is wrong (should not happen)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 3. Try to reduce rank to 50
    # We will try to zero out 4 columns and optimize the rest.
    # Which columns to remove? 
    # In the schoolbook solution, each column corresponds to a unique (i,j,k) triple.
    # Removing a column means we lose the term for that specific i,j,k.
    # We need to redistribute that term's contribution to other columns.
    
    # Let's pick 4 random columns to "merge" into the others.
    # Or better, let's just optimize a rank-50 matrix initialized from rank-54 by setting 4 columns to 0?
    # No, setting to 0 breaks the sum. We need to adjust the other 50 to compensate.
    
    # Strategy:
    # 1. Pick a subset of 50 columns from the 54.
    # 2. Initialize these 50 columns with the values from the schoolbook solution.
    # 3. The "missing" 4 terms create a residual error E = T - T_current.
    # 4. Optimize the 50 columns to minimize ||T - T_rec||^2.
    # 5. Round and verify.
    
    # Let's try multiple random subsets of 50 columns.
    indices = np.arange(R_init)
    best_score = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    max_time = 100 # Leave margin
    
    # We will try 5 different random subsets
    for trial in range(5):
        if time.time() - start_time > max_time:
            break
            
        # Select 50 indices to keep
        keep_indices = np.random.choice(indices, 50, replace=False)
        
        U_curr = np.zeros((n*m, 50))
        V_curr = np.zeros((m*p, 50))
        W_curr = np.zeros((p*n, 50))
        
        # Initialize with schoolbook values for the kept indices
        for new_idx, old_idx in enumerate(keep_indices):
            U_curr[:, new_idx] = U[:, old_idx]
            V_curr[:, new_idx] = V[:, old_idx]
            W_curr[:, new_idx] = W[:, old_idx]
            
        # Current error
        T_curr = reconstruct(U_curr, V_curr, W_curr)
        E = T - T_curr
        
        # Optimization: Gradient Descent on U, V, W
        # Loss = ||T - sum U_r \otimes V_r \otimes W_r||^2
        # This is non-convex. We use simple gradient descent with small steps.
        
        lr = 0.01
        # Number of steps depends on time
        steps = 2000
        
        for step in range(steps):
            if time.time() - start_time > max_time - 10:
                break
            
            # Compute gradient
            # T_rec = sum_r U_r \otimes V_r \otimes W_r
            # dL/dU = -2 * (T - T_rec) contracted with V and W
            # Specifically, dL/dU[u, r] = -2 * sum_{v,w} (T[u,v,w] - T_rec[u,v,w]) V[v,r] W[w,r]
            
            T_rec = reconstruct(U_curr, V_curr, W_curr)
            Res = T - T_rec
            
            # Gradient for U
            # G_U[u, r] = -2 * sum_{v,w} Res[u,v,w] V[v,r] W[w,r]
            # We can compute this efficiently.
            # Res is (n*m, m*p, p*n)
            # V is (m*p, 50), W is (p*n, 50)
            
            # G_U = -2 * np.einsum('uvw,vr,wr->ur', Res, V_curr, W_curr)
            G_U = -2 * np.einsum('uvw,vr,wr->ur', Res, V_curr, W_curr)
            
            # G_V[v, r] = -2 * sum_{u,w} Res[u,v,w] U[u,r] W[w,r]
            G_V = -2 * np.einsum('uvw,ur,wr->vr', Res, U_curr, W_curr)
            
            # G_W[w, r] = -2 * sum_{u,v} Res[u,v,w] U[u,r] V[v,r]
            G_W = -2 * np.einsum('uvw,ur,vr->wr', Res, U_curr, V_curr)
            
            # Update
            U_curr -= lr * G_U
            V_curr -= lr * G_V
            W_curr -= lr * G_W
            
            # Optional: Periodically round to check for exact solution?
            # Rounding every step is expensive and might trap in local minima.
            # We will round at the end of the trial.
            
        # After optimization, round to nearest 0.5
        U_rounded = np.round(U_curr * 2) / 2
        V_rounded = np.round(V_curr * 2) / 2
        W_rounded = np.round(W_curr * 2) / 2
        
        # Verify
        T_final = reconstruct(U_rounded, V_rounded, W_rounded)
        err = np.linalg.norm(T_final - T)
        
        if err < 1e-9:
            # Found a valid rank 50 decomposition!
            # Check if all entries are integers or half-integers (they are by construction)
            # Return immediately
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
            
        # If not exact, keep the best error found so far?
        # The task requires EXACT decomposition. Invalid results score 0.
        # So we must either find an exact one or fall back to 54.
        # We don't return approximate solutions.

    # 4. Fallback: Return the rank 54 solution
    # We know this is exact.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
