"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _build_block_strassen():
    """Return (U, V, W) as lists of lists, rank = 42 (2x2 Strassen on j-blocks)."""
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for U (first matrix M, 2x2 local)
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1},
    ]
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ]
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]

    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), c in U_coeffs[s].items():
                    U[i_loc * m + (a * 2 + j_loc)][col] += c
                for (j_loc, k_loc), c in V_coeffs[s].items():
                    V[(a * 2 + j_loc) * p + (b * 2 + k_loc)][col] += c
                for (i_loc, k_loc), c in W_contrib[s].items():
                    W[(b * 2 + k_loc) * n + i_loc][col] += c
                col += 1
    return U, V, W


def _build_4x2x2_blocks():
    """
    Return (U, V, W) for a 4x4x4-style decomposition on 4 j-rows, 2 i, 2 k
    using a 4x4 Strassen-like scheme (rank 16 per 2x2 i-k block).
    This gives 4 i-rows * 2 k-cols * 16 = 128 terms. Not useful, but we
    use it only as a starting point for pruning.
    """
    n, m, p = 2, 4, 6
    rank = 128
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Standard 4x4 Strassen uses 16 mults (7 for 2x2, but 4x4 naive is 16)
    # We'll just do naive 4x4: 16 products
    # But to make it "Strassen-like" we'll use 7x7=49 for 8x8, not applicable.
    # Actually let's just do the naive 4x4 for the j dimension, giving rank 16 per 2x2 block
    # That's the same as naive. Not helpful.
    # Instead, let's try a different approach: split into 4x2x2 with rank 5 (optimal for 2x2x2 is 7,
    # but 2x2x4 might have lower rank per unit). This is getting complex.
    # For now, return None to indicate this is not implemented.
    return None


def _prune(U, V, W, rank, tol=1e-9, max_iter=50):
    """
    Try to reduce rank by identifying dependent columns.
    We'll use a simple approach: for each column, check if it can be represented
    as a linear combination of previous columns.
    """
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    nU, nV, nW = U_np.shape[0], V_np.shape[0], W_np.shape[0]
    
    kept = []
    for c in range(rank):
        u_col = U_np[:, c]
        v_col = V_np[:, c]
        w_col = W_np[:, c]
        
        # Check if this column is a linear combination of previously kept columns
        # We'll use a simple rank check on the combined matrix
        if not kept:
            kept.append(c)
            continue
        
        # Build matrix from kept columns
        kept_U = U_np[:, kept]
        kept_V = V_np[:, kept]
        kept_W = W_np[:, kept]
        
        # Check if [u_col, v_col, w_col] is in the span of the kept columns
        # This is a system of linear equations. We'll use least squares.
        # Concatenate all three matrices
        A = np.concatenate([kept_U, kept_V, kept_W], axis=0)
        b = np.concatenate([u_col, v_col, w_col], axis=0)
        
        # Solve least squares
        try:
            sol, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=tol)
            pred = A @ sol
            if np.linalg.norm(b - pred) < tol * max(1.0, np.linalg.norm(b)):
                # This column is redundant
                continue
        except np.linalg.LinAlgError:
            pass
        
        kept.append(c)
    
    # Rebuild with kept columns
    if len(kept) < rank:
        new_U = [U_np[:, c] for c in kept]
        new_V = [V_np[:, c] for c in kept]
        new_W = [W_np[:, c] for c in kept]
        return (
            [[round(x, 10) for x in col] for col in np.array(new_U).T],
            [[round(x, 10) for x in col] for col in np.array(new_V).T],
            [[round(x, 10) for x in col] for col in np.array(new_W).T]
        )
    return U, V, W


def construct():
    start_time = time.time()
    
    # Start with block Strassen (rank 42)
    U, V, W = _build_block_strassen()
    rank = len(U[0]) if U else 0
    
    # Try to prune
    if rank > 0:
        U, V, W = _prune(U, V, W, rank)
        rank = len(U[0]) if U else 0
    
    # Safety check
    if time.time() - start_time > 130:
        # Return the best we have
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
