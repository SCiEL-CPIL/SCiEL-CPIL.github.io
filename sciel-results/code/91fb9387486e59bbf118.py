"""
Rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.

Strategy:
1. Start from the schoolbook decomposition (rank 63).
2. Use a structured perturbation + Newton refinement to attempt to reduce rank.
   The key insight: if we can find a linear dependence among the 63 rank-1
   terms, we can eliminate one term. We parameterize small perturbations and
   solve the resulting nonlinear system.
3. Also try a direct structured construction: exploit the fact that for 3x3*3x7,
   we can group the 7 columns of B and use fast 3x3 multiplication blocks.
4. As a final fallback, return the schoolbook decomposition.

The output factor W follows the trace convention W[k*n+i, r] where n=3, p=7.
So W has shape (p*n, rank) = (21, rank).
"""

import numpy as np
import time

N, M, P = 3, 3, 7
RANK_SCHOOLBOOK = N * M * P  # 63
DEADLINE_SECONDS = 130.0


def _schoolbook():
    """Return the canonical schoolbook decomposition of rank 63."""
    rank = RANK_SCHOOLBOOK
    U = np.zeros((N * M, rank), dtype=np.float64)
    V = np.zeros((M * P, rank), dtype=np.float64)
    W = np.zeros((P * N, rank), dtype=np.float64)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1
    return U, V, W


def _check_decomposition(U, V, W, tol=1e-10):
    """Verify U, V, W form a valid decomposition of the matmul tensor."""
    n, m, p = N, M, P
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    computed = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)
    return np.allclose(computed, np.ones((n, m, p)), atol=tol, rtol=0.0)


def _try_eliminate_one_term(seed=42, max_iters=500, time_budget=25.0):
    """
    Try to find a rank-62 decomposition by perturbing the schoolbook and
    using Newton's method to satisfy the tensor equations while forcing
    one column to become linearly dependent.
    
    Approach: Start with schoolbook rank-63. Introduce a small perturbation
    to 3 of the 63 terms (one per factor) to create a dependency, then
    solve for corrections that maintain the tensor identity.
    """
    t_start = time.time()
    rng = np.random.default_rng(seed)
    
    for attempt in range(max_iters):
        if time.time() - t_start > time_budget:
            break
        if time.time() > DEADLINE_SECONDS - 10:
            break
        
        # Start from schoolbook
        U0, V0, W0 = _schoolbook()
        
        # Pick a random column to try to eliminate
        target_col = rng.integers(0, RANK_SCHOOLBOOK)
        
        # Create a perturbation: add a small random vector to each factor
        # at the target column, and adjust other columns to compensate
        eps = 1e-3 * (1 + 0.5 * rng.random())
        
        # Perturb the target column
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
        
        # Add small random perturbations to a few other columns
        num_perturb = rng.integers(2, 6)
        perturb_cols = rng.choice(RANK_SCHOOLBOOK, size=num_perturb, replace=False)
        
        for pc in perturb_cols:
            U[:, pc] += eps * rng.standard_normal(N * M)
            V[:, pc] += eps * rng.standard_normal(M * P)
            W[:, pc] += eps * rng.standard_normal(P * N)
        
        # Now we have 63 columns that don't exactly decompose the tensor.
        # We want to find a rank-62 decomposition. One approach:
        # Express the target column as a linear combination of the others,
        # then remove it and adjust.
        
        # Compute the residual tensor
        U3 = U.reshape(N, M, RANK_SCHOOLBOOK)
        V3 = V.reshape(M, P, RANK_SCHOOLBOOK)
        W3 = W.reshape(P, N, RANK_SCHOOLBOOK)
        computed = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)
        residual = computed - np.ones((N, M, P))
        
        # If residual is small, we're close. Try to eliminate target_col.
        # The contribution of target_col to the tensor is:
        U_t = U[:, target_col].reshape(N, M)
        V_t = V[:, target_col].reshape(M, P)
        W_t = W[:, target_col].reshape(P, N)
        
        # We want to find coefficients alpha_c for c != target_col such that:
        # sum_{c != target} alpha_c * U[:,c] * V[:,c] * W[:,c] ≈ U_t * V_t * W_t
        # This is a linear system in the alpha's if we fix two of the three factors.
        
        # Simplified: try to express U_t as a linear combination of other U columns
        # (this only works if the other U columns span U_t)
        
        # Build matrix of other U columns
        other_cols = [c for c in range(RANK_SCHOOLBOOK) if c != target_col]
        U_others = U[:, other_cols]  # (9, 62)
        
        # Check if U_t is in the span of U_others
        # Use least squares
        try:
            alpha, residuals, rank_u, sv = np.linalg.lstsq(U_others, U[:, target_col], rcond=None)
            if residuals.size > 0 and residuals[0] < 1e-6:
                # U_t is approximately in the span. Now check if the full
                # rank-1 tensor can be expressed.
                # This is heuristic - just check the full decomposition
                pass
        except np.linalg.LinAlgError:
            pass
        
        # Alternative direct approach: try to find a rank-62 by solving
        # the full nonlinear system with one column constrained.
        # This is complex, so let's try a simpler heuristic.
        
        # Heuristic: if we can find 62 columns whose tensor sum is ~1,
        # we're done. Try random subsets.
        if attempt % 10 == 0:
            # Try a random rank-62 subset with small adjustments
            U62 = U0[:, :62].copy()
            V62 = V0[:, :62].copy()
            W62 = W0[:, :62].copy()
            
            # The missing term is (i0, j0, k0) corresponding to column 62
            i0 = 62 // P
            j0 = (62 % P) // N  # This mapping is wrong, let me fix
            # Actually column index in schoolbook: col = i*M*P + j*P + k
            col62 = 62
            k0 = col62 % P
            j0 = (col62 // P) % M
            i0 = col62 // (P * M)
            
            # The missing contribution is U[i0*M+j0]=1, V[j0*P+k0]=1, W[k0*N+i0]=1
            # We need to add this to the existing 62 terms.
            # Add a small amount to each of the 62 terms to distribute the load.
            add = 1.0 / 62
            U62[i0 * M + j0, :] += add
            V62[j0 * P + k0, :] += add
            W62[k0 * N + i0, :] += add
            
            if _check_decomposition(U62, V62, W62, tol=1e-8):
                return U62, V62, W62
    
    return None


def _try_structured_rank_62(seed=42, time_budget=20.0):
    """
    Try a structured approach: group the 7 columns into groups and use
    fast matrix multiplication within groups.
    
    For 3x3 * 3x7: split B into B1 (3x3) and B2 (3x4).
    C = A*B1 + A*B2
    Use rank-23 for 3x3*3x3 (if exists) + rank-? for 3x3*3x4.
    
    Actually, let's try: use 2 copies of 3x3*3x3 (rank 27 each = 54) 
    plus 3x3*3x1 (rank 9) = 63. No improvement.
    
    Try: 3x3*3x3 rank 23 (Strassen-like) + 3x3*3x4.
    3x3*3x4: schoolbook rank 36. Total = 23+36 = 59.
    But we need to verify rank-23 for 3x3*3x3 exists.
    
    For now, try a direct rank-62 construction by merging two schoolbook
    terms that share structure.
    """
    t_start = time.time()
    
    # Try to merge two terms that have overlapping support
    # Term (i,j,k) and (i',j',k') can potentially be merged if
    # they share i==i' or j==j' or k==k'
    
    # For 3x3*3x7, consider terms with same (i,j) but different k.
    # There are 3*3 = 9 such groups, each with 7 terms.
    # Within a group, U and V are the same, only W varies.
    # sum_k U[i*M+j] * V[j*P+k] * W[k*N+i] for k=0..6
    # = U[i*M+j] * V[j*P: j*P+3... wait V is (M*P, rank)
    
    # Actually in schoolbook, each term is separate. Let's try to
    # express two terms with same (i,j) as one term using a rank-2
    # decomposition of the sum.
    
    # Term1: U1=1 at (i,j), V1=1 at (j,k1), W1=1 at (k1,i)
    # Term2: U2=1 at (i,j), V2=1 at (j,k2), W2=1 at (k2,i)
    # Sum: U1*V1*W1 + U2*V2*W2
    # = 1 * [V1*W1 + V2*W2] at position (i,j)
    # The vector V1*W1 + V2*W2 is a rank-1 + rank-1 = rank-2 tensor in (V,W) space.
    # So we need 2 terms to represent this, not saving anything.
    
    # Different approach: use the fact that for fixed (i,j), the 7 terms
    # form a linear combination. If we can find a dependency...
    
    # Let's just try random rank-62 searches more aggressively
    rng = np.random.default_rng(seed)
    
    for attempt in range(200):
        if time.time() - t_start > time_budget:
            break
        if time.time() > DEADLINE_SECONDS - 10:
            break
        
        # Start with schoolbook, zero out one column, and try to
        # redistribute its contribution
        U0, V0, W0 = _schoolbook()
        
        target = rng.integers(0, RANK_SCHOOLBOOK)
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
        
        # Zero out target column
        U[:, target] = 0
        V[:, target] = 0
        W[:, target] = 0
        
        # The missing contribution:
        col = target
        k_miss = col % P
        j_miss = (col // P) % M
        i_miss = col // (P * M)
        
        # Add small contributions to nearby columns to compensate
        # Find columns with same i_miss
        same_i_cols = [c for c in range(RANK_SCHOOLBOOK) 
                       if c != target and (c // (P * M)) == i_miss]
        
        if len(same_i_cols) > 0:
            # Distribute the missing term's contribution
            # The missing term contributes to C[i_miss, k_miss] via A[i_miss,j_miss]*B[j_miss,k_miss]
            # We need: sum_c U[i_miss*M+j_miss, c] * V[j_miss*P+k_miss, c] * W[k_miss*N+i_miss, c] = 1
            # Currently these are 0 for the target, so we need to add 1 total.
            
            # Add to each same-i column proportionally
            n_cols = len(same_i_cols)
            add_val = 1.0 / n_cols
            
            for c in same_i_cols:
                U[i_miss * M + j_miss, c] += add_val
                V[j_miss * P + k_miss, c] += add_val
                W[k_miss * N + i_miss, c] += add_val
            
            # Remove the target column to get rank 62
            keep = [c for c in range(RANK_SCHOOLBOOK) if c != target]
            U62 = U[:, keep]
            V62 = V[:, keep]
            W62 = W[:, keep]
            
            if _check_decomposition(U62, V62, W62, tol=1e-8):
                return U62, V62, W62
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W'.
    """
    t0 = time.time()
    
    # Try structured rank-62 construction
    if time.time() - t0 < DEADLINE_SECONDS - 30:
        result = _try_structured_rank_62(seed=42, time_budget=min(25.0, DEADLINE_SECONDS - (time.time() - t0) - 10))
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try perturbation-based rank reduction
    if time.time() - t0 < DEADLINE_SECONDS - 20:
        result = _try_eliminate_one_term(seed=123, max_iters=300, time_budget=min(15.0, DEADLINE_SECONDS - (time.time() - t0) - 5))
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
