"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    size_a = n * m  # 4
    size_b = m * p  # 28
    size_c = p * n  # 28
    
    # Build the true tensor T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0

    def verify(U, V, W, atol=1e-9):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T_true[a,b,c]."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(T_recon, T_true, atol=atol)

    def schoolbook():
        """Return the standard rank-56 schoolbook decomposition."""
        rank = n * m * p
        U = np.zeros((size_a, rank), dtype=int)
        V = np.zeros((size_b, rank), dtype=int)
        W = np.zeros((size_c, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def try_strassen_block_decomposition():
        """
        Try a Strassen-like block decomposition.
        Partition the 14 columns into 7 pairs and apply 7-term Strassen per pair.
        Total rank: 7 * 7 = 49.
        """
        rank = 49
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)
        
        for l in range(7):
            col_start = 7 * l
            k1 = 2 * l
            k2 = 2 * l + 1
            
            # M1: (A00+A11)(B0,k1 + B1,k2)
            U[0, col_start] = 1
            U[3, col_start] = 1
            V[k1, col_start] = 1
            V[p + k2, col_start] = 1
            W[k1 * n, col_start] = 1
            W[k2 * n + 1, col_start] = 1
            
            # M2: (A10+A11) * B0,k1
            U[2, col_start+1] = 1
            U[3, col_start+1] = 1
            V[k1, col_start+1] = 1
            W[k1 * n + 1, col_start+1] = 1
            W[k2 * n + 1, col_start+1] = -1
            
            # M3: A00 * (B0,k2 - B1,k2)
            U[0, col_start+2] = 1
            V[k2, col_start+2] = 1
            V[p + k2, col_start+2] = -1
            W[k2 * n, col_start+2] = 1
            W[k2 * n + 1, col_start+2] = 1
            
            # M4: A11 * (B1,k1 - B0,k1)
            U[3, col_start+3] = 1
            V[p + k1, col_start+3] = 1
            V[k1, col_start+3] = -1
            W[k1 * n, col_start+3] = 1
            W[k1 * n + 1, col_start+3] = 1
            
            # M5: (A00+A01) * B1,k2
            U[0, col_start+4] = 1
            U[1, col_start+4] = 1
            V[p + k2, col_start+4] = 1
            W[k1 * n, col_start+4] = -1
            W[k2 * n, col_start+4] = 1
            
            # M6: (A10-A00) * (B0,k1 + B0,k2)
            U[2, col_start+5] = 1
            U[0, col_start+5] = -1
            V[k1, col_start+5] = 1
            V[k2, col_start+5] = 1
            W[k2 * n + 1, col_start+5] = 1
            
            # M7: (A01-A11) * (B1,k1 + B1,k2)
            U[1, col_start+6] = 1
            U[3, col_start+6] = -1
            V[p + k1, col_start+6] = 1
            V[p + k2, col_start+6] = 1
            W[k1 * n, col_start+6] = 1
        
        return U, V, W

    def try_rank_reduction():
        """
        Try to reduce rank below 49 by finding linear dependencies among the 49 rank-1 tensors.
        Uses SVD-based null space detection and proper tensor rank reduction.
        """
        if time.time() > deadline:
            return None
        
        # Start with the Strassen-49 decomposition
        U_curr, V_curr, W_curr = try_strassen_block_decomposition()
        
        if not verify(U_curr, V_curr, W_curr):
            return None
        
        current_rank = 49
        reduced_U = U_curr.astype(np.float64).copy()
        reduced_V = V_curr.astype(np.float64).copy()
        reduced_W = W_curr.astype(np.float64).copy()
        
        # Iteratively try to reduce rank
        while current_rank > 1 and time.time() < deadline:
            # Build the tensor matrix: each column r is the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # Shape: (current_rank, size_a * size_b * size_c)
            tensors = np.zeros((current_rank, size_a * size_b * size_c), dtype=np.float64)
            for r in range(current_rank):
                T_r = np.einsum('a,b,c->abc', reduced_U[:,r], reduced_V[:,r], reduced_W[:,r])
                tensors[r] = T_r.flatten()
            
            # Find the numerical rank and null space
            try:
                U_svd, s, Vt = np.linalg.svd(tensors, full_matrices=True)
            except np.linalg.LinAlgError:
                break
            
            # Determine numerical rank
            tol = 1e-10
            if len(s) == 0:
                break
            num_rank = np.sum(s > tol * s[0])
            
            if num_rank >= current_rank:
                # No null space, cannot reduce further
                break
            
            # Get null space vectors: rows num_rank..end of Vt
            null_dim = current_rank - num_rank
            null_vecs = Vt[num_rank:, :]
            
            # Try each null vector: if c is in null space, then sum_r c[r] * T_r = 0
            # We can eliminate one term by absorbing it into others
            reduced = False
            for nv in null_vecs:
                if time.time() > deadline:
                    break
                
                # Find index k with largest |c[k]|
                k = np.argmax(np.abs(nv))
                if abs(nv[k]) < 1e-8:
                    continue
                
                # Scale so nv[k] = 1: then T_k = -sum_{i≠k} nv[i] * T_i
                # New decomposition: replace term k with absorption
                # U_new[:,i] = U[:,i] for i≠k, and we need to "remove" term k
                # 
                # Proper approach: if sum_r c[r] U_r ⊗ V_r ⊗ W_r = 0,
                # we can set one coefficient to 0 and adjust others.
                # But for rank-1 tensors, absorption requires special structure.
                #
                # Simpler valid approach: if the tensors are linearly dependent,
                # we can find a basis of the span with fewer vectors.
                # Use the row space: keep only the first num_rank rows of Vt (transposed).
                
                # Actually, the cleanest approach: project onto the column space
                # The span of {T_0, ..., T_{r-1}} has dimension num_rank
                # We can re-express using a basis, but the new basis vectors won't be rank-1
                # in general. So this only works if we can find a rank-1 combination.
                
                # Alternative: try to find if any single tensor is a linear combination
                # of the others in a way that preserves rank-1 structure.
                # This is hard. Let's try a different strategy:
                # 
                # If there's a null vector c, then for any perturbation, we can
                # shift along c without changing the sum. But to reduce rank,
                # we need one coefficient to become exactly 0.
                
                # Try: set c[k] to 0 by absorbing into other terms.
                # This works if we can write T_k as a sum of rank-1 tensors
                # that are combinations of the other T_i's. Generally not possible.
                
                # So the only guaranteed reduction is if num_rank < current_rank
                # AND we can find a subset of size num_rank that spans the same space
                # with each remaining tensor still being rank-1.
                
                # For now, let's just check: can we drop term k and rescale others?
                # sum_r c[r] T_r = 0 means T_k = -sum_{i≠k} (c[i]/c[k]) T_i
                # This is a sum of rank-1 tensors, not a single rank-1 tensor.
                # So we can't simply drop it.
                
                # The rank of the tensor is the minimum number of rank-1 tensors needed.
                # Linear dependence among the rank-1 terms doesn't automatically reduce
                # the tensor rank unless the dependence has special structure.
                
                # Let's try a different approach: use the fact that our tensors have
                # a block structure (7 blocks of 7 Strassen terms). Check if there are
                # cross-block dependencies.
                
                # For now, break if we can't find a clean reduction
                break
            
            if not reduced:
                break
        
        # If no reduction found, return None
        if current_rank >= 49:
            return None
            
        if verify(reduced_U, reduced_V, reduced_W):
            # Trim the zero columns
            mask = np.any(np.abs(reduced_U) > 1e-10, axis=0) | np.any(np.abs(reduced_V) > 1e-10, axis=0) | np.any(np.abs(reduced_W) > 1e-10, axis=0)
            idx = np.where(mask)[0]
            return reduced_U[:, idx], reduced_V[:, idx], reduced_W[:, idx]
            
        return None

    # First try the constructive rank-49 Strassen decomposition
    U, V, W = try_strassen_block_decomposition()
    if verify(U, V, W):
        # Try to reduce rank further
        reduced = try_rank_reduction()
        if reduced is not None:
            U_r, V_r, W_r = reduced
            if verify(U_r, V_r, W_r):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If that didn't work, check with looser tolerance
    if verify(U, V, W, atol=1e-6):
        # Try to clean up
        U_clean = np.round(U * 2) / 2
        V_clean = np.round(V * 2) / 2
        W_clean = np.round(W * 2) / 2
        if verify(U_clean, V_clean, W_clean):
            return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 56
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
