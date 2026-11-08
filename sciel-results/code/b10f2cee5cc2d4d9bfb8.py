"""Matrix multiplication 3x3x6 rank-minimal tensor decomposition."""
import time
import random
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 20s safety margin before 160s kill

DIMENSIONS = [3, 3, 6]


def _time_left():
    return _DEADLINE - time.time()


def _verify(U, V, W, n, m, p, seed=42, atol=1e-8):
    """Verify: C[i,k] = sum_c W[k*n+i,c] * (U_c . A_flat) * (V_c . B_flat)."""
    rank = U.shape[1]
    rng = np.random.RandomState(seed)
    A = rng.rand(n, m)
    B = rng.rand(m, p)
    C_expected = A @ B
    A_flat = A.flatten()
    B_flat = B.flatten()
    C_computed = np.zeros((n, p))
    # Vectorized: S1 = U^T A_flat (rank,), S2 = V^T B_flat (rank,)
    S1 = U.T @ A_flat
    S2 = V.T @ B_flat
    # C = sum_c S1[c]*S2[c] * W_c reshaped
    # W is (p*n, rank), reshape each column to (p, n) -> transpose to (n, p)
    # C = (S1 * S2) @ W_reshaped  where W_reshaped is (rank, n*p) with W_c as (n,p)
    W_reshaped = W.T.reshape(rank, p, n).transpose(0, 2, 1).reshape(rank, n * p)
    C_computed = (S1 * S2) @ W_reshaped
    C_computed = C_computed.reshape(n, p)
    return np.allclose(C_computed, C_expected, atol=atol)


def _verify_multi(U, V, W, n, m, p, n_trials=3):
    """Verify with multiple random matrices."""
    for seed in range(n_trials):
        if not _verify(U, V, W, n, m, p, seed=seed):
            return False
    return True


def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _rank1_terms(n, m, p):
    """Generate all rank-1 terms: each term is (U_vec, V_vec, W_vec) with entries in {-1,0,1}.
    For efficiency, we use a randomized/greedy approach rather than full enumeration."""
    pass  # We'll use a different strategy below


def _greedy_search(n, m, p, max_rank_target, max_time=120.0):
    """
    Greedy search for low-rank decomposition.
    Strategy: Try to find rank-27 solution using block structure.
    For 3x3x6: split B into two 3x3 blocks. Use 3 products per output column.
    
    Actually, let's try a direct approach: enumerate small-rank candidates
    by random initialization + local search.
    
    Better: Use the known structure. For 3x3x6 = 3x3x(2*3):
    If we can do 3x3x3 at rank R, then 3x3x6 needs 2R terms.
    Best known for 3x3x3 is 23 (Laderman). So 3x3x6 could be 46.
    But actually we might do better with joint structure.
    
    Let me try a randomized approach: initialize with random sparse terms
    and greedily add terms that reduce residual error.
    """
    start = time.time()
    budget = min(max_time, _time_left() - 5.0)
    if budget < 10:
        return None
    
    # Target: try to find rank < 54
    best_rank = n * m * p  # schoolbook
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    
    # Strategy 1: Try rank 27 via structured approach
    # For each output (i,k), we need C[i,k] = sum_j A[i,j]*B[j,k]
    # A rank-1 term contributes: (sum_{i,j} U[i*m+j]*A[i,j]) * (sum_{j',k'} V[j'*p+k']*B[j',k']) * W[k*n+i]
    # 
    # Key insight: if we restrict W to have at most one nonzero per column,
    # each term contributes to exactly one output entry.
    # Then we need: for each (i,k), C[i,k] = W[k*n+i] * sA * sB where
    # sA = sum_{i,j} U[i*m+j] A[i,j], sB = sum_{j',k'} V[j'*p+k'] B[j',k']
    # And we need sA * sB to contain sum_j A[i,j] B[j,k].
    # This is hard for rank-1 since sA*sB is a product of two linear forms.
    #
    # Alternative: allow W to have multiple nonzeros. Then one term can
    # contribute to multiple outputs.
    #
    # Let's try: for each term c, U_c picks a subset of A entries,
    # V_c picks a subset of B entries, W_c distributes to outputs.
    #
    # For rank 27: 27 terms for 18 outputs. Average 1.5 outputs per term.
    # Each term: sA * sB * W_c. If sA = sum of 3 A entries (one row),
    # sB = sum of 3 B entries (one column), then sA*sB = 9 cross terms.
    # W_c can pick which of these go to which outputs.
    #
    # Actually the cleanest: use the "border basis" or "small matrix" approach.
    #
    # Let me just do a randomized local search:
    # 1. Start with random rank-30 solution (overcomplete)
    # 2. Greedily zero out columns that don't hurt accuracy
    # 3. Try random restarts
    
    # Actually, let's try a more principled approach:
    # The tensor T has entries T[(i,j),(j',k),(i'',k'')] = delta(i,i'')*delta(j,j')*delta(k,k'')
    # We want sum_c U_c[i*m+j] * V_c[j'*p+k] * W_c[k''*n+i''] = T[(i,j),(j',k),(i'',k'')]
    #
    # For a valid decomposition, for each (i'',k''):
    # sum_c W_c[k''*n+i''] * (sum_{i,j} U_c[i*m+j]*delta(i,i'')*delta(j,j')) ... 
    # No, the tensor equation is:
    # sum_c U_c[i*m+j] * V_c[j'*p+k] * W_c[k'*n+i'] = delta(i,i')*delta(j,j')*delta(k,k')
    # for all i,j,j',k,i',k'.
    #
    # This means: for fixed (i',k'), the 2D array in (j,j') must satisfy:
    # sum_c W_c[k'*n+i'] * U_c[i*m+j] * V_c[j'*p+k] = delta(i,i')*delta(j,j')*delta(k,k')
    #
    # If i' != i, RHS = 0. If i' = i, then for (j,k'):
    # sum_c W_c[k'*n+i'] * U_c[i*m+j] * V_c[j'*p+k'] = delta(j,j')*delta(k,k')
    #
    # This is a system of bilinear equations. Let's set up and solve.
    #
    # Number of unknowns: rank * (n*m + m*p + p*n) = rank * (9 + 18 + 18) = rank * 45
    # Number of equations: n*m*m*p*p*n = 3*3*3*6*6*3 = 2916? No...
    # Indices: i,j in A (9 pairs), j',k in B (18 pairs), i',k' in C (18 pairs)
    # Total: 9 * 18 * 18 = 2916 equations.
    # But many are trivially satisfied.
    #
    # Non-trivial constraints: For (i',k'), the matrix M_{jk'} = sum_c W_c[k'*n+i']*U_c[i*m+j]*V_c[j'*p+k]
    # must equal delta(i,i')*outer(e_j, e_k) when j=j', k=k'.
    #
    # For i' = i: M must be identity-like: M[j,k] = delta(j,j')*delta(k,k')... 
    # Wait, M is indexed by (j, j', k) where j from A, j' from B, k from B.
    # M[j, j', k] = sum_c W_c[k'*n+i] * U_c[i*m+j] * V_c[j'*p+k] = delta(j,j') * delta(k,k)
    # But k is free... this doesn't make sense. Let me re-read.
    #
    # The tensor: T_{(i,j),(j',k),(i'',k'')} = delta(i,i'') * delta(j,j') * delta(k,k'')
    # So for fixed (i'',k''): T[(i,j),(j',k),(i'',k'')] = delta(i,i'')*delta(j,j')*delta(k,k'')
    # If i'' != i: 0
    # If i'' = i: delta(j,j')*delta(k,k'')
    #
    # The decomposition: sum_c U_c[i*m+j]*V_c[j'*p+k]*W_c[k''*n+i''] = T[(i,j),(j',k),(i'',k'')]
    #
    # For i'' = i: sum_c W_c[k''*n+i]*U_c[i*m+j]*V_c[j'*p+k] = delta(j,j')*delta(k,k'')
    # This must hold for ALL j, j', k.
    #
    # Let's define for fixed i, k'': M_{j, j', k} = sum_c W_c[k''*n+i]*U_c[i*m+j]*V_c[j'*p+k]
    # We need M_{j,j',k} = delta(j,j')*delta(k,k'')
    #
    # This is a rank-1 constraint on M as a 3-tensor, but more usefully:
    # For fixed i, k'': the 3x3 matrix M[:, :, k''] in (j, j') must be:
    # M[j, j', k''] = delta(j,j') * delta(k'',k'') = delta(j,j') if k''=k''... 
    # Wait, delta(k,k'') with k being the B index. If k = k'', then 1, else 0.
    # But k' is the fixed output column. So:
    # M[j, j', k] = delta(j,j') * delta(k, k'')
    #
    # So for the slice at column k'': M[:,:,k''] = I_3 (identity in j,j'), and M[:,:,k≠k''] = 0.
    #
    # This means: for each (i, k''), the vector (over c) W_c[k''*n+i] * U_c[i*m+j] for fixed j
    # times V_c[j'*p+k] must reconstruct I_3 delta(k,k'').
    #
    # This is getting complex. Let me just do numerical optimization.
    
    # Numerical approach: random initialization + gradient-free local search
    # Try to minimize the residual of the tensor equation.
    
    rng = np.random.RandomState(0)
    max_attempts = 20
    attempt = 0
    
    while attempt < max_attempts and time.time() < start + budget:
        attempt += 1
        # Try different rank targets
        for target_rank in [24, 27, 30, 33, 36]:
            if time.time() >= start + budget:
                break
            U = rng.randn(n * m, target_rank) * 0.5
            V = rng.randn(m * p, target_rank) * 0.5
            W = rng.randn(p * n, target_rank) * 0.5
            
            # Local search: try to zero out columns
            for col in range(target_rank - 1, -1, -1):
                if time.time() >= start + budget:
                    break
                # Try zeroing this column
                old_W = W[:, col].copy()
                W[:, col] = 0
                if _verify_multi(U, V, W, n, m, p, n_trials=2):
                    target_rank = col  # successfully removed
                else:
                    W[:, col] = old_W
            
            if target_rank < best_rank:
                if _verify_multi(U, V, W, n, m, p, n_trials=5):
                    best_rank = target_rank
                    best_U = U[:, :target_rank].copy()
                    best_V = V[:, :target_rank].copy()
                    best_W = W[:, :target_rank].copy()
                    print(f"Found rank {best_rank}")
                    if best_rank <= 27:
                        return best_U, best_V, best_W
    
    if best_rank < n * m * p:
        return best_U, best_V, best_W
    return None


def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a better solution
    result = _greedy_search(n, m, p, max_rank_target=30, max_time=115.0)
    
    if result is not None:
        U, V, W = result
        # Final verification
        if _verify_multi(U, V, W, n, m, p, n_trials=10):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    assert _verify_multi(U, V, W, n, m, p, n_trials=5), "Schoolbook verification failed"
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
