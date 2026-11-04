"""Matrix multiplication 2x4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np
import itertools

DIMENSIONS = [2, 4, 5]
_DEADLINE = time.time() + 140

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    dim_a, dim_b, dim_c = n * m, m * p, p * n
    rank = U.shape[1]
    T_recon = np.zeros((dim_a, dim_b, dim_c))
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    T_target = _build_tensor(n, m, p)
    return np.allclose(T_recon, T_target, atol=tol)


def _try_rank_r(n, m, p, target_rank, max_iters=200, seed=42):
    """
    Try to find a rank-target_rank decomposition using structured random search
    with local refinement.
    
    Key insight: For matrix multiplication A(n×m) @ B(m×p) = C(p×n),
    the tensor has a specific sparsity pattern. We use the fact that:
    - The U matrix has structure: U[i*m+j, r] can be written as u_i[j, r]
    - Similarly V[j*p+k, r] = v_j[k, r], W[k*n+i, r] = w_k[i, r]
    
    We parametrize using the block structure and search for solutions.
    """
    dim_a, dim_b, dim_c = n * m, m * p, p * n
    
    T = _build_tensor(n, m, p)
    rng = np.random.RandomState(seed)
    
    # Strategy: Use the constraint that T is sparse and has a "cycle" structure.
    # For each term r, U[:,r], V[:,r], W[:,r] must be such that their
    # outer product contributes to the right places.
    
    # Approach: Random initialization with block structure, then iterative
    # refinement using the fact that we need exact zeros in wrong positions.
    
    for attempt in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        
        # Initialize with random half-integer values in {-1, 0, 0.5, 1}
        U = np.zeros((dim_a, target_rank))
        V = np.zeros((dim_b, target_rank))
        W = np.zeros((dim_c, target_rank))
        
        # For each rank term, assign random sparse patterns
        for r in range(target_rank):
            # Choose which (i,j) pairs this term affects in U
            # Each term should be "localized" in some sense
            u_mask = rng.random(dim_a) < 0.3
            v_mask = rng.random(dim_b) < 0.3
            w_mask = rng.random(dim_c) < 0.3
            
            for idx in np.where(u_mask)[0]:
                U[idx, r] = rng.choice([-1, -0.5, 0.5, 1])
            for idx in np.where(v_mask)[0]:
                V[idx, r] = rng.choice([-1, -0.5, 0.5, 1])
            for idx in np.where(w_mask)[0]:
                W[idx, r] = rng.choice([-1, -0.5, 0.5, 1])
        
        # Compute reconstruction error
        T_recon = np.zeros((dim_a, dim_b, dim_c))
        for r in range(target_rank):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        error = np.max(np.abs(T_recon - T))
        if error < 1e-9:
            return {"U": U, "V": V, "W": W}
        
        # Local refinement: try to fix the largest errors
        if attempt % 10 == 0 and attempt > 0:
            # Find worst positions
            diff = T_recon - T
            # Try to adjust some entries
            for _ in range(20):
                # Pick a random position with large error
                flat_idx = rng.randint(dim_a * dim_b * dim_c)
                a, b, c = np.unravel_index(flat_idx, (dim_a, dim_b, dim_c))
                
                # Pick a random rank term
                r = rng.randint(target_rank)
                
                # Try adjusting U[a,r], V[b,r], or W[c,r]
                which = rng.randint(3)
                if which == 0:
                    old = U[a, r]
                    U[a, r] = old + rng.choice([-1, -0.5, 0.5, 1])
                    if U[a, r] not in [-1, -0.5, 0, 0.5, 1]:
                        U[a, r] = old
                elif which == 1:
                    old = V[b, r]
                    V[b, r] = old + rng.choice([-1, -0.5, 0.5, 1])
                    if V[b, r] not in [-1, -0.5, 0, 0.5, 1]:
                        V[b, r] = old
                else:
                    old = W[c, r]
                    W[c, r] = old + rng.choice([-1, -0.5, 0.5, 1])
                    if W[c, r] not in [-1, -0.5, 0, 0.5, 1]:
                        W[c, r] = old
        
    return None


def _try_structured_rank20(n, m, p):
    """
    Attempt a structured rank-20 decomposition for 2x4x5.
    
    The key structural insight: the tensor T = sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ki}
    has a "3-cycle" structure. For 2x4x5, we can try to exploit the fact that
    n=2 is small.
    
    Idea: Split the 20 rank terms into groups associated with different j values.
    For each j in {0,1,2,3}, we have 5 terms (one for each k).
    
    Term for (j, k): 
      U[:, r_{j,k}] should have support on rows i*m+j for all i
      V[:, r_{j,k}] should have support on rows j*p+k' for all k'
      W[:, r_{j,k}] should have support on rows k'*n+i' where k'=k
    
    But this doesn't work directly because the k in U's row doesn't match.
    
    Alternative: Use the fact that for fixed (i, j), the contribution to W
    is over all k. Let's try a different grouping.
    
    Group by (i, k): n*p = 10 terms
    Group by (j, k): m*p = 20 terms - this is our target!
    
    For each (j, k) pair, define a term that:
    - In U: has 1 at position (i, j) for all i (but this would give 2 per j)
    - In V: has 1 at position (j, k) 
    - In W: has 1 at position (k, i) for all i
    
    Problem: The i in U and W must be the same for a given term, but we're
    summing over i. So a single term can't handle both i values.
    
    Better idea: For each (j, k, i), we need a term. That's 40 terms (schoolbook).
    To reduce to 20, we need to "share" terms between different i values.
    
    For n=2, maybe we can use a trick: have 20 terms where each term handles
    both i=0 and i=1 in a clever way.
    
    Let me try: r = 0..19, r = 4*j + k for j in 0..3, k in 0..4
    U[i*m+j, r] = 1 if j matches (regardless of i)
    V[j'*p+k', r] = 1 if j'=j and k'=k
    W[k''*n+i', r] = ??? 
    
    For this to work, we need sum over r of U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    where a=i*m+j, b=j'*p+k', c=k''*n+i'
    
    For nonzero T, we need i=i', j=j', k=k''.
    
    With our ansatz:
    U[i*m+j, r_{j,k}] = 1 for all i
    V[j*p+k, r_{j,k}] = 1
    W[k'*n+i', r_{j,k}] = w_{k', i', j, k}
    
    Then sum over r of U[a,r]*V[b,r]*W[c,r] 
    = sum over (j,k) where j=j', k=k' of U[i*m+j', r_{j',k'}]*V[j'*p+k', r_{j',k'}]*W[k'*n+i', r_{j',k'}]
    = U[i*m+j', r_{j',k'}]*V[j'*p+k', r_{j',k'}]*W[k'*n+i', r_{j',k'}]
    = 1 * 1 * W[k'*n+i', r_{j',k'}]
    
    We need this to equal 1 when i=i' and 0 otherwise.
    So W[k'*n+i', r_{j',k'}] should be 1 if i'=i and 0 otherwise.
    
    But W[k'*n+i', r_{j',k'}] doesn't depend on i (the first index of U).
    So we can't distinguish i=0 from i=1 in W.
    
    This approach fails. We need W to depend on i somehow.
    
    New idea: Use 20 terms but with W having structure that encodes i.
    Let r = 10*j + k for j in 0..1, k in 0..4 (only 10 terms this way)
    No, that's only 10.
    
    Let me try r = 5*i + k for i in 0..1, k in 0..4 (10 terms)
    U[i'*m+j', r_{i,k}] = 1 if i'=i
    V[j'*p+k', r_{i,k}] = 1 if k'=k
    W[k''*n+i'', r_{i,k}] = 1 if k''=k and i''=i
    
    Then sum = U[i*m+j, r_{i',k'}]*V[j*p+k, r_{i',k'}]*W[k'*n+i', r_{i',k'}]
    This is nonzero only if i=i', k=k', and the value is 1*1*1=1.
    But we also need j=j' and the sum should be over all (i',k') with i'=i, k'=k.
    Actually for fixed (i,j,k), the sum over r gives:
    Only r_{i,k} contributes, giving 1.
    For wrong positions, we need the sum to be 0.
    
    Let's check: a=i*m+j, b=j'*p+k', c=k''*n+i''
    T[a,b,c] = 1 if i=i', j=j', k=k'', else 0.
    
    Sum_r U[a,r]*V[b,r]*W[c,r]:
    U[i*m+j, r_{i',k'}] = 1 if i'=i else 0
    V[j'*p+k', r_{i',k'}] = 1 if k'=k' else 0 (always true)
    W[k''*n+i'', r_{i',k'}] = 1 if k''=k' and i''=i' else 0
    
    So the sum is 1 if i'=i, k'=k'', i''=i' => i'=i, k'=k'', i''=i.
    And we need j=j' for T to be 1.
    
    The sum gives 1 when i'=i, k'=k'', i''=i, regardless of j, j'.
    But T is 1 only when j=j'.
    
    So we get extra nonzero terms when j≠j' but i=i', k=k''.
    This doesn't work.
    
    Let's give up on rank 20 and try rank 30 or something, or just return schoolbook.
    """
    return None


def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 40
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to find lower rank
    if time.time() < _DEADLINE - 20:
        # Try rank 20 (theoretical lower bound)
        result = _try_rank_r(n, m, p, 20, max_iters=50, seed=42)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
        
    if time.time() < _DEADLINE - 15:
        # Try rank 25
        result = _try_rank_r(n, m, p, 25, max_iters=50, seed=123)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
    
    if time.time() < _DEADLINE - 10:
        # Try rank 30
        result = _try_rank_r(n, m, p, 30, max_iters=50, seed=456)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
    
    # Return schoolbook decomposition
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
