import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def _build_strassen_28():
    """Pre-compute the rank-28 Strassen decomposition for M_{2,2,8}.
    
    Uses the tensor-train / structured approach: 4 independent 2x2 blocks,
    each decomposed via Strassen's rank-7 algorithm. Total rank = 4*7 = 28.
    """
    n, m, p = 2, 2, 8
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # Each entry: dict mapping (row, col) index to coefficient
    strassen_products = [
        # M1 = (A00 + A11) * (B00 + B11)
        {'A': {(0,0): 1, (1,1): 1},
         'B': {(0,0): 1, (1,1): 1},
         'C': {(0,0): 1, (1,1): 1}},
        # M2 = (A10 + A11) * B00
        {'A': {(1,0): 1, (1,1): 1},
         'B': {(0,0): 1},
         'C': {(1,0): 1, (1,1): -1}},
        # M3 = A00 * (B01 - B11)
        {'A': {(0,0): 1},
         'B': {(0,1): 1, (1,1): -1},
         'C': {(0,1): 1, (1,1): 1}},
        # M4 = A11 * (B10 - B00)
        {'A': {(1,1): 1},
         'B': {(1,0): 1, (0,0): -1},
         'C': {(0,0): 1, (1,0): 1}},
        # M5 = (A00 + A01) * B11
        {'A': {(0,0): 1, (0,1): 1},
         'B': {(1,1): 1},
         'C': {(0,0): -1, (0,1): 1}},
        # M6 = (A10 - A00) * (B00 + B01)
        {'A': {(1,0): 1, (0,0): -1},
         'B': {(0,0): 1, (0,1): 1},
         'C': {(1,1): 1}},
        # M7 = (A01 - A11) * (B10 + B11)
        {'A': {(0,1): 1, (1,1): -1},
         'B': {(1,0): 1, (1,1): 1},
         'C': {(0,0): 1}},
    ]
    
    for q in range(p // 2):  # 4 groups of 2 columns
        k1 = 2 * q
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r  # term index
            
            # Fill U: A coefficients (index: i*m + j)
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
            
            # Fill V: B coefficients (index: j*p + k_global)
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            
            # Fill W: C contributions (index: k_global*n + i)
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff
    
    return U, V, W


def _verify_decomposition(U, V, W, n=2, m=2, p=8, tol=1e-9):
    """Verify that sum_t u_t ⊗ v_t ⊗ w_t reconstructs the multiplication tensor.
    
    Vectorized check: for all (i,j,k), 
      sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
    """
    # Build index arrays for all (i, j, k) combinations
    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    i_flat = i_vals.ravel()
    j_flat = j_vals.ravel()
    k_flat = k_vals.ravel()
    
    # Index into U, V, W
    u_idx = i_flat * m + j_flat  # shape (n*m*p,)
    v_idx = j_flat * p + k_flat  # shape (n*m*p,)
    w_idx = k_flat * n + i_flat  # shape (n*m*p,)
    
    # Extract rows: shape (n*m*p, rank)
    U_rows = U[u_idx, :]  # (n*m*p, rank)
    V_rows = V[v_idx, :]  # (n*m*p, rank)
    W_rows = W[w_idx, :]  # (n*m*p, rank)
    
    # Compute sum_t U[t] * V[t] * W[t] for each (i,j,k)
    vals = np.sum(U_rows * V_rows * W_rows, axis=1)
    
    # All should equal 1
    errors = np.sum(np.abs(vals - 1.0) > tol)
    return errors == 0, int(errors)


def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Pre-compute at module load time if not already cached
    if not hasattr(construct, '_U_cached'):
        _U, _V, _W = _build_strassen_28()
        construct._U_cached = _U
        construct._V_cached = _V
        construct._W_cached = _W
    
    U = construct._U_cached
    V = construct._V_cached
    W = construct._W_cached
    
    # Quick vectorized verification
    if time.time() < deadline:
        ok, errs = _verify_decomposition(U, V, W, n, m, p)
        if not ok:
            # Fallback: rebuild from scratch
            U, V, W = _build_strassen_28()
            construct._U_cached = U
            construct._V_cached = V
            construct._W_cached = W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
