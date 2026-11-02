import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is correct."""
        for a in range(n * m):
            i, j = divmod(a, m)
            for b in range(m * p):
                j2, k = divmod(b, p)
                if j2 != j:
                    continue
                for c in range(p * n):
                    k2, i2 = divmod(c, n)
                    expected = 1 if (k2 == k and i2 == i) else 0
                    actual = 0
                    for r in range(rank):
                        actual += U[a, r] * V[b, r] * W[c, r]
                    if actual != expected:
                        return False
        return True

    def build_trivial(rank):
        """Build the trivial decomposition with given rank."""
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)

        i_idx, j_idx, k_idx = np.indices((n, m, p))
        i_flat = i_idx.flatten()
        j_flat = j_idx.flatten()
        k_flat = k_idx.flatten()

        u_rows = i_flat * m + j_flat
        v_rows = j_flat * p + k_flat
        w_rows = k_flat * n + i_flat
        r_indices = np.arange(rank)

        U[u_rows, r_indices] = 1
        V[v_rows, r_indices] = 1
        W[w_rows, r_indices] = 1

        return U, V, W

    # First, get the trivial decomposition (rank 63)
    U63, V63, W63 = build_trivial(63)
    
    if not verify_decomposition(U63, V63, W63, 63):
        raise ValueError("Trivial decomposition verification failed")
    
    if time.time() > deadline:
        return {"U": U63.tolist(), "V": V63.tolist(), "W": W63.tolist()}

    # Try to find a rank-62 decomposition
    # Strategy: Use the trivial 63-term decomposition and try to merge two terms
    # or find a linear dependency to eliminate one term
    
    # Known result: For M_{n,m,p} with min(n,m,p) >= 2, the rank is at most n*m*p - 1
    # We can try to find a non-trivial linear combination
    
    # Approach: Start with the 63-term decomposition and look for a way to reduce
    # One approach: find a rank-1 matrix R such that we can subtract it and 
    # still have a valid decomposition
    
    # Alternative: Use the fact that for certain parameter ranges, we can reduce rank
    # by exploiting symmetries or known bounds
    
    # Let's try a direct search: modify the last term to be a combination
    # of the others, hoping to maintain correctness
    
    # Actually, let's try a simpler approach: use the known formula
    # For M_{n,m,p}, one can achieve rank n*m*p - 1 when n,m,p are all >= 2
    # by replacing one term with a clever combination
    
    # The standard trick: for the term corresponding to (i0, j0, k0),
    # replace it with a sum that still contributes correctly
    
    # Let's try: take the trivial decomposition and modify it
    # U', V', W' with rank 62
    
    U62 = np.zeros((n * m, 62), dtype=int)
    V62 = np.zeros((m * p, 62), dtype=int)
    W62 = np.zeros((p * n, 62), dtype=int)
    
    # Use the first 62 terms from the trivial decomposition
    # and see if we can make them work
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    i_flat = i_idx.flatten()[:62]
    j_flat = j_idx.flatten()[:62]
    k_flat = k_idx.flatten()[:62]
    
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    r_indices = np.arange(62)
    
    U62[u_rows, r_indices] = 1
    V62[v_rows, r_indices] = 1
    W62[w_rows, r_indices] = 1
    
    if verify_decomposition(U62, V62, W62, 62):
        if time.time() > deadline:
            return {"U": U62.tolist(), "V": V62.tolist(), "W": W62.tolist()}
        return {"U": U62.tolist(), "V": V62.tolist(), "W": W62.tolist()}
    
    # If that didn't work, try a different modification
    # Use the full 63 terms but combine the last two into one
    U_try = np.zeros((n * m, 62), dtype=int)
    V_try = np.zeros((m * p, 62), dtype=int)
    W_try = np.zeros((p * n, 62), dtype=int)
    
    # Use first 61 terms as-is
    i_idx_full, j_idx_full, k_idx_full = np.indices((n, m, p))
    i_full = i_idx_full.flatten()
    j_full = j_idx_full.flatten()
    k_full = k_idx_full.flatten()
    
    for r in range(61):
        U_try[i_full[r] * m + j_full[r], r] = 1
        V_try[j_full[r] * p + k_full[r], r] = 1
        W_try[k_full[r] * n + i_full[r], r] = 1
    
    # For the 62nd term (index 61 in 0-based), try to combine terms 61 and 62
    # Term 61: (i61, j61, k61) = (i_full[61], j_full[61], k_full[61])
    # Term 62: (i62, j62, k62) = (i_full[62], j_full[62], k_full[62])
    
    # We need: U62[:,61] * V62[:,61] * W62[:,61] to cover both terms
    # This is generally not possible with a single rank-1 term
    # Unless the two terms can be expressed as a single rank-1 tensor
    
    # Check if terms 61 and 62 are "compatible"
    # Term r1: (i1, j1, k1) and term r2: (i2, j2, k2)
    # They can be merged if:
    # - They share the same i (for U)
    # - They share the same j (for V)  
    # - They share the same k (for W)
    # But then they'd be the same term!
    
    # So direct merging won't work. Let's try a different approach.
    
    # Try: can we use a non-integer coefficient?
    # The problem says "integer/half-integer entries"
    
    # Let's try the standard rank reduction technique:
    # Replace two terms with a combination that uses the other terms
    
    # Actually, let's just return the trivial decomposition if we can't find better
    if time.time() > deadline:
        return {"U": U63.tolist(), "V": V63.tolist(), "W": W63.tolist()}
    
    return {"U": U63.tolist(), "V": V63.tolist(), "W": W63.tolist()}
# EVOLVE-BLOCK-END
