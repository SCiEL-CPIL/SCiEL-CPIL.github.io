"""
Program for rank-minimal exact tensor decomposition of 3x3 by 3x7 matrix multiplication.

Strategy: Start from the schoolbook rank-63 decomposition. Then use structured
rank reduction by identifying linear dependencies in the tensor. For each rank
r from 62 down, attempt to find a rank-r decomposition by:
1. Starting from the rank-(r+1) solution (or schoolbook if r+1=63)
2. Attempting to merge/eliminate one rank-1 term using least-squares
3. Verifying the result exactly

We also try a direct optimization approach for small rank reductions.
"""

import time
import numpy as np
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Returns:
        dict: A dictionary containing 'U', 'V', 'W' matrices.
        U has shape (n*m, rank)
        V has shape (m*p, rank)
        W has shape (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21

    # Build the target tensor T[a,b,c]
    # For multiplying (n x m) by (m x p) to get (n x p):
    # C[i,l] = sum_j A[i,j] * B[j,l]
    # a = i*m + j, b = j*p + l, c = l*n + i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    nonzero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                nonzero_entries.append((a, b, c))

    # Build the set of nonzero (a,b,c) for verification
    nz_set = set(nonzero_entries)

    # All possible (a,b,c) triples
    all_triples = list(range(nm))
    # For efficient verification, we need to check that all entries match

    def verify_exact(U, V, W, tol=1e-10):
        """Check if the decomposition is valid for all entries."""
        rank = U.shape[1]
        # Check all nonzero entries
        for a, b, c in nonzero_entries:
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            if abs(val - 1.0) > tol:
                return False
        # Check all zero entries
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in nz_set:
                        val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                        if abs(val) > tol:
                            return False
        return True

    def verify_sampled(U, V, W, n_samples=2000, seed=42, tol=1e-9):
        """Fast verification using sampling."""
        rank = U.shape[1]
        rng = np.random.RandomState(seed)
        # Check all nonzero entries exactly
        for a, b, c in nonzero_entries:
            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            if abs(val - 1.0) > tol:
                return False
        # Sample zero entries
        for _ in range(n_samples):
            a = rng.randint(0, nm)
            b = rng.randint(0, mp)
            c = rng.randint(0, pn)
            if (a, b, c) not in nz_set:
                val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                if abs(val) > tol:
                    return False
        return True

    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p  # 63
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W

    U_best, V_best, W_best = schoolbook()
    best_rank = 63

    # Strategy 1: Try to reduce rank by eliminating terms
    # For each pair of terms, try to see if one can be expressed as a combination
    # of the others (this is a complex nonlinear problem)
    
    # Strategy 2: Use the fact that for fixed k, the sub-tensor is a 3x3 matrix-vector product
    # Group terms by k (the column of B)
    # For each k, we have 9 terms: (i,j) for i,j in {0,1,2}
    # The tensor entry is T[3i+j, 3j+k, 3k+i] = 1
    
    # Key insight: For different k values, the W indices 3k+i are in different ranges:
    # k=0: c in {0,1,2}
    # k=1: c in {3,4,5}
    # k=2: c in {6,7,8}
    # k=3: c in {9,10,11}
    # k=4: c in {12,13,14}
    # k=5: c in {15,16,17}
    # k=6: c in {18,19,20}
    
    # Since the W indices for different k are disjoint, the rank is at least the sum
    # of ranks of each sub-tensor. Each sub-tensor for fixed k is the tensor for
    # multiplying a 3x3 matrix by a 3x1 vector, which has rank 9 (since it's a
    # linear map from R^9 x R^3 -> R^3, and the rank of the corresponding
    # 9x3x3 tensor is 9).
    
    # Wait, let me reconsider. The tensor for 3x3 * 3x1 is:
    # T_k[a, b, c] where a = i*3+j (i,j in 0..2), b = j*1+k = j (since p=1 for this subproblem), c = k*3+i = i
    # So T_k[i*3+j, j, i] = 1
    # This is the matrix multiplication tensor for 3x3 * 3x1, which has rank 9.
    
    # Since the W-index sets for different k are disjoint, the total rank is exactly
    # 7 * 9 = 63. So the schoolbook rank is minimal!
    
    # Let me verify this claim more carefully.
    # The full tensor T has entries T[a,b,c] = 1 iff there exist i,j,k such that
    # a = 3i+j, b = 3j+k, c = 3k+i
    # with i,j in {0,1,2} and k in {0,...,6}.
    
    # For a fixed k, the support is {(3i+j, 3j+k, 3k+i) : i,j in {0,1,2}}
    # The c-values for fixed k are {3k, 3k+1, 3k+2}, which are disjoint across k.
    
    # The rank of a tensor is the minimum number of rank-1 tensors needed to sum to it.
    # If two rank-1 tensors have disjoint support in the third index, they cannot
    # cancel each other. So the rank of the direct sum is the sum of the ranks.
    
    # More precisely: T = sum_{k=0}^{6} T_k where T_k is supported on c in {3k,3k+1,3k+2}.
    # Since these c-ranges are disjoint, any rank-r decomposition of T must have
    # rank at least sum_k rank(T_k).
    
    # Each T_k is the 3x3 * 3x1 matrix multiplication tensor, which has rank 9.
    # (Proof: The tensor T_k[i*3+j, j, i] = 1 for i,j in 0..2.
    #  Reslicing along the first index gives a 3x3 matrix for each (j,i) pair... 
    #  Actually, let me just verify numerically that rank 9 is needed and achievable.)
    
    # The schoolbook gives rank 9 for each T_k, so total rank 63.
    # If the rank of each T_k is exactly 9, then the total rank is exactly 63.
    
    # Let's verify the rank of a single T_k (say k=0) is 9.
    # T_0[i*3+j, j, i] = 1 for i,j in 0..2.
    # This is the tensor for (3x3) * (3x1) matrix multiplication.
    # The rank of matrix multiplication M_{p,q,r} is at least p*q*r / max(p,q,r) ... no.
    # Actually, the rank of M_{3,3,1} is known to be 9 (it's just the standard
    # matrix-vector product, which requires 9 multiplications and no fewer).
    # This is because the 9 entries of the output are all independent bilinear forms.
    
    # So the rank is exactly 63, and the schoolbook is optimal.
    
    # But wait - maybe I'm wrong about the disjointness argument. Let me reconsider.
    # In a rank-r decomposition T = sum_{s=1}^{r} u_s ⊗ v_s ⊗ w_s, the vectors
    # u_s, v_s, w_s can have non-zero components in any coordinate. The condition
    # that T[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s] must equal the target tensor.
    
    # The fact that different T_k have disjoint c-support doesn't immediately imply
    # that the rank is additive, because a single rank-1 term u ⊗ v ⊗ w can have
    # w with non-zero components in multiple c-ranges. However, if w has non-zero
    # components in two different c-ranges (say c in {0,1,2} and c in {3,4,5}),
    # then the term contributes to both T_0 and T_1, which could potentially allow
    # for a lower total rank.
    
    # So the disjointness argument doesn't directly give additivity. Let me try
    # to actually find a lower-rank decomposition.
    
    # Let me try an optimization approach to see if rank < 63 is achievable.
    
    # For now, let's just return the schoolbook since we suspect it's optimal.
    # But let's do a quick check to make sure our schoolbook is correct.
    
    if verify_exact(U_best, V_best, W_best):
        # The schoolbook is valid. Let's try to improve it.
        pass

    # Attempt to find lower rank using iterative methods
    # We'll try ranks from 62 down to 50
    
    for target_rank in range(62, 49, -1):
        if time.time() > deadline - 20.0:
            break
        
        success = False
        for attempt in range(2):
            if time.time() > deadline - 20.0:
                break
                
            rng = np.random.RandomState(42 + attempt * 1000)
            U = rng.randn(nm, target_rank) * 0.01
            V = rng.randn(mp, target_rank) * 0.01
            W = rng.randn(pn, target_rank) * 0.01
            
            # Use alternating least squares / gradient-based optimization
            # Objective: minimize sum_{a,b,c} (T[a,b,c] - sum_s U[a,s]V[b,s]W[c,s])^2
            
            lr = 0.001
            for step in range(100):
                if time.time() > deadline - 20.0:
                    break
                
                # Compute gradients using a subset of entries for speed
                # Sample a mix of nonzero and zero entries
                n_sample = 200
                sample_a = rng.randint(0, nm, n_sample)
                sample_b = rng.randint(0, mp, n_sample)
                sample_c = rng.randint(0, pn, n_sample)
                
                # Compute current tensor values at these points
                T_cur = np.array([
                    np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                    for a, b, c in zip(sample_a, sample_b, sample_c)
                ])
                
                T_tgt = T[sample_a, sample_b, sample_c]
                
                diff = T_tgt - T_cur  # shape (n_sample,)
                
                # Gradient w.r.t. U: dU[a,s] = sum_{(a',b',c') in sample: a'=a} diff * V[b',s] * W[c',s]
                # This is expensive to compute directly. Let's use a vectorized approach.
                
                # For each sample point s_idx, the contribution to grad_U[a, r] is
                # diff[s_idx] * V[b_idx, r] * W[c_idx, r]
                
                # Vectorized: for each r, grad_U[:, r] = sum over samples where a matches
                # This is hard to vectorize efficiently. Let's use a different approach.
                
                # Instead, let's update one variable at a time using least squares.
                break  # Skip the gradient approach, use ALS instead
            
            # Use Alternating Least Squares
            for it in range(50):
                if time.time() > deadline - 20.0:
                    break
                
                # Update U: for each a, solve min_U_row sum_{b,c} (T[a,b,c] - U_row @ V[b,:] @ W[c,:])^2
                # This is equivalent to: U_row = argmin sum_{b,c} (T[a,b,c] - U_row @ M_bc)^2
                # where M_bc = V[b,:] * W[c,:] (outer product-like)
                # Actually: sum_{b,c} T[a,b,c] V[b,r] W[c,r] = U_row[r]
                # So we solve: U_row @ A = y where A[r1,r2] = sum_{b,c} V[b,r1] W[c,r2] ... no.
                
                # Let me reformulate. For fixed a:
                # min_{U_row} sum_{b,c} (T[a,b,c] - sum_r U_row[r] V[b,r] W[c,r])^2
                # = min_{U_row} || T_row - U_row @ (V^T * W^T flattened) ||^2
                # where the "flattened" matrix has entries M[b*c_idx, r] = V[b,r]*W[c,r]
                
                # Build M: shape (mp*pn, target_rank)
                # M[b*pn+c, r] = V[b,r] * W[c,r]
                M = np.outer(V, np.ones(pn))  # This is wrong, let me do it properly
                # M[i, r] where i = b*pn + c, M[i,r] = V[b,r] * W[c,r]
                M = np.zeros((mp * pn, target_rank))
                for b in range(mp):
                    for c in range(pn):
                        M[b * pn + c, :] = V[b, :] * W[c, :]
                
                # For each a, y_a = T[a, :, :] flattened = T[a, b, c] for all b,c
                T_flat = T.reshape(nm, mp * pn)
                
                # Solve: U[a, :] = argmin || T_flat[a, :] - U[a, :] @ M ||^2
                # This is a least squares problem: U[a, :] = T_flat[a, :] @ M^+ (pseudo-inverse)
                # Or equivalently, solve M^T M @ u = M^T @ T_flat[a, :]
                
                if target_rank < mp * pn:
                    # Use least squares
                    try:
                        U, residuals, rank_m, sv = np.linalg.lstsq(M.T, T_flat.T, rcond=None)
                    except np.linalg.LinAlgError:
                        U, residuals, rank_m, sv = np.linalg.lstsq(M.T, T_flat.T, rcond=None)
                else:
                    U = T_flat.T @ np.linalg.pinv(M.T)
                
                # Now update V and W similarly
                # For V: fix U and W, for each b:
                # min_{V_row} sum_{a,c} (T[a,b,c] - sum_r U[a,r] V_row[r] W[c,r])^2
                # N[a*pn+c, r] = U[a,r] * W[c,r]
                N = np.zeros((nm * pn, target_rank))
                for a in range(nm):
                    for c in range(pn):
                        N[a * pn + c, :] = U[a, :] * W[c, :]
                
                T_flat2 = np.zeros((mp, nm * pn))
                for b in range(mp):
                    for a in range(nm):
                        for c in range(pn):
                            T_flat2[b, a * pn + c] = T[a, b, c]
                
                try:
                    V = np.linalg.lstsq(N.T, T_flat2.T, rcond=None)[0]
                except:
                    V = T_flat2.T @ np.linalg.pinv(N.T)
                
                # For W: fix U and V
                # O[a*mp+b, r] = U[a,r] * V[b,r]
                O = np.zeros((nm * mp, target_rank))
                for a in range(nm):
                    for b in range(mp):
                        O[a * mp + b, :] = U[a, :] * V[b, :]
                
                T_flat3 = np.zeros((pn, nm * mp))
                for c in range(pn):
                    for a in range(nm):
                        for b in range(mp):
                            T_flat3[c, a * mp + b] = T[a, b, c]
                
                try:
                    W = np.linalg.lstsq(O.T, T_flat3.T, rcond=None)[0]
                except:
                    W = T_flat3.T @ np.linalg.pinv(O.T)
            
            # Verify
            if verify_sampled(U, V, W, n_samples=1000):
                if verify_exact(U, V, W):
                    U_best = U
                    V_best = V
                    W_best = W
                    best_rank = target_rank
                    success = True
                    break
        
        if success:
            break
    
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
# EVOLVE-BLOCK-END
