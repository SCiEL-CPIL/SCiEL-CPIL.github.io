def _als_step(U, V, W, T, T0, T1, T2, rank, max_iter=200, deadline=None, round_step=False):
    """Alternating least squares. Returns (U, V, W, converged)."""
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            return U, V, W, False
        # Solve for U
        B = (V[:, None, :] * W[None, :, :]).reshape(rank, -1)
        U, *_ = np.linalg.lstsq(B.T, T0.T, rcond=None)
        # Solve for V
        B = (U[:, None, :] * W[None, :, :]).reshape(rank, -1)
        V, *_ = np.linalg.lstsq(B.T, T1.T, rcond=None)
        # Solve for W
        B = (U[:, None, :] * V[None, :, :]).reshape(rank, -1)
        W, *_ = np.linalg.lstsq(B.T, T2.T, rcond=None)
        
        if round_step:
            # Hard rounding to nearest 0.5 to encourage integer/half-integer solutions
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
    return U, V, W, True

def _try_rank16(deadline):
    """Bounded randomized search for a rank-16 decomposition. Returns (U,V,W) or None."""
    T = _build_tensor()
    T0 = T.reshape(U_DIM, -1)
    T1 = T.transpose(1, 0, 2).reshape(V_DIM, -1)
    T2 = T.transpose(2, 0, 1).reshape(W_DIM, -1)
    rank = 16

    # Strategy 1: Start from rank-17, zero out one column, then ALS
    U17, V17, W17 = _rank17()
    for drop_col in range(17):
        if time.time() > deadline:
            return None
        U = U17[:, :drop_col].copy()
        U = np.hstack([U, U17[:, drop_col+1:]]) if drop_col < 16 else U17[:, :16].copy()
        V = V17[:, :drop_col].copy()
        V = np.hstack([V, V17[:, drop_col+1:]]) if drop_col < 16 else V17[:, :16].copy()
        W = W17[:, :drop_col].copy()
        W = np.hstack([W, W17[:, drop_col+1:]]) if drop_col < 16 else W17[:, :16].copy()
        U, V, W, _ = _als_step(U, V, W, T, T0, T1, T2, rank, max_iter=100, deadline=deadline)
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W

    # Strategy 2: Sparse, integer-biased initializations
    # Hypothesis: The solution has entries in {0, 0.5, 1, -1}
    for trial in range(100):
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(10000 + trial)
        
        # Initialize with sparse 0/1/-1 values
        # Density around 0.2-0.4
        density = 0.2 + 0.2 * (trial % 5) / 5.0
        U = (rng.rand(U_DIM, rank) < density) * (2 * (rng.rand(U_DIM, rank) < 0.5) - 1)
        V = (rng.rand(V_DIM, rank) < density) * (2 * (rng.rand(V_DIM, rank) < 0.5) - 1)
        W = (rng.rand(W_DIM, rank) < density) * (2 * (rng.rand(W_DIM, rank) < 0.5) - 1)
        
        # Run ALS with rounding every few steps
        U, V, W, _ = _als_step(U, V, W, T, T0, T1, T2, rank, max_iter=150, deadline=deadline)
        
        # Final polish: Round to nearest 0.5 and check
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if _validate(U_r, V_r, W_r, T, atol=1e-6):
            return U_r, V_r, W_r
        
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W

    # Strategy 3: "Algebraic" pruning - Fix Strassen block
    # We know the top-left 2x2 block of A and B can be handled by Strassen.
    # Let's try to force the first 7 columns to be the Strassen pattern 
    # (or a permutation of it) and solve for the remaining 9 columns.
    # This reduces the search space significantly.
    
    # Permute Strassen columns to align with different subsets of indices
    # This is complex, so we stick to random restarts but with tighter ALS.
    
    for trial in range(50):
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(20000 + trial)
        # Small random perturbations of the rank-17 solution
        U = U17[:, :16].copy() + rng.randn(U_DIM, 16) * 0.1
        V = V17[:, :16].copy() + rng.randn(V_DIM, 16) * 0.1
        W = W17[:, :16].copy() + rng.randn(W_DIM, 16) * 0.1
        
        U, V, W, _ = _als_step(U, V, W, T, T0, T1, T2, rank, max_iter=200, deadline=deadline)
        
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if _validate(U_r, V_r, W_r, T, atol=1e-6):
            return U_r, V_r, W_r

        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W

    return None
