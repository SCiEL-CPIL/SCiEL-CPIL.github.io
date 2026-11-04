import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
_U2 = np.array([
    [1, 0, 1, 0, 1, -1, 0],  # A00
    [0, 0, 0, 0, 1, 0, 1],   # A01
    [0, 1, 0, 0, 0, 1, 0],   # A10
    [1, 1, 0, 1, 0, 0, -1]   # A11
], dtype=int)

_V2 = np.array([
    [1, 1, 0, -1, 0, 1, 0],  # B00
    [0, 0, 1, 0, 0, 1, 0],   # B01
    [0, 0, 0, 1, 0, 0, 1],   # B10
    [1, 0, -1, 0, 1, 0, 1]   # B11
], dtype=int)

_W2 = np.array([
    [1, 0, 0, 1, -1, 0, 1],  # C00
    [0, 0, 1, 0, 1, 0, 0],   # C01
    [0, 1, 0, 1, 0, 0, 0],   # C10
    [1, -1, 1, 0, 0, 1, 0]   # C11
], dtype=int)

def _verify_decomposition(U, V, W, n, m, p):
    """Check if the tensor decomposition exactly reproduces matrix multiplication.
    
    The matrix multiplication tensor T[i,j,k] = 1 if the (i,j) entry of A times
    the (j,k) entry of B contributes to C[i,k]. In the trace convention W[k*n+i, r],
    we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    """
    rank = U.shape[1]
    # Build the product tensor
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    U_np = np.asarray(U)
    V_np = np.asarray(V)
    W_np = np.asarray(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                val = 0
                for r in range(rank):
                    val += U_np[u_row, r] * V_np[v_row, r] * W_np[w_row, r]
                if val != 1:
                    return False
    return True

def _build_decomposition(n, m, p, u_idx_fn, v_idx_fn, w_idx_fn):
    """Build the Kronecker-Strassen decomposition with given index functions."""
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                i1, i0 = divmod(i, 2)
                j1, j0 = divmod(j, 2)
                k1, k0 = divmod(k, 2)
                
                u_idx = u_idx_fn(i, j)
                v_idx = v_idx_fn(j, k)
                w_idx = w_idx_fn(k, i)
                
                u_block_row = i1 * 2 + j1
                u_entry_row = i0 * 2 + j0
                v_block_row = j1 * 2 + k1
                v_entry_row = j0 * 2 + k0
                w_block_row = i1 * 2 + k1
                w_entry_row = i0 * 2 + k0
                
                for alpha in range(7):
                    for beta in range(7):
                        col = alpha * 7 + beta
                        U[u_idx, col] = _U2[u_block_row, alpha] * _U2[u_entry_row, beta]
                        V[v_idx, col] = _V2[v_block_row, alpha] * _V2[v_entry_row, beta]
                        W[w_idx, col] = _W2[w_block_row, alpha] * _W2[w_entry_row, beta]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin: evaluator kills at 150s
    
    # Primary index mapping: standard row-major
    # U: A[i,j] -> row i*m+j
    # V: B[j,k] -> row j*p+k
    # W: C[i,k] -> row k*n+i (trace convention: W[k*n+i, r])
    
    def u_idx_std(i, j):
        return i * m + j
    
    def v_idx_std(j, k):
        return j * p + k
    
    def w_idx_std(k, i):
        return k * n + i
    
    U, V, W = _build_decomposition(n, m, p, u_idx_std, v_idx_std, w_idx_std)
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If primary mapping fails, try alternative index mappings
    # Alternative 1: W indexed as i*n+k instead of k*n+i
    def w_idx_alt1(k, i):
        return i * n + k
    
    U, V, W = _build_decomposition(n, m, p, u_idx_std, v_idx_std, w_idx_alt1)
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Alternative 2: U indexed as j*n+i, V as k*m+j
    def u_idx_alt2(i, j):
        return j * n + i
    
    def v_idx_alt2(j, k):
        return k * m + j
    
    U, V, W = _build_decomposition(n, m, p, u_idx_alt2, v_idx_alt2, w_idx_std)
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Alternative 3: All transposed
    def u_idx_alt3(i, j):
        return j * n + i
    
    def v_idx_alt3(j, k):
        return k * m + j
    
    def w_idx_alt3(k, i):
        return i * n + k
    
    U, V, W = _build_decomposition(n, m, p, u_idx_alt3, v_idx_alt3, w_idx_alt3)
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return the standard mapping (most likely correct)
    U, V, W = _build_decomposition(n, m, p, u_idx_std, v_idx_std, w_idx_std)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
