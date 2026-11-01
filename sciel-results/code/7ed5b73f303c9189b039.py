"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W matrices of shape (4,7) for a single 2x2x2 block.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a00 + a11) * (b00 + b11)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # M2 = (a10 + a11) * b00
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = -1.0
    
    # M3 = a00 * (b10 - b11)
    U[0, 2] = 1.0
    V[2, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0; W[3, 2] = 1.0
    
    # M4 = a11 * (b01 - b00)
    U[3, 3] = 1.0
    V[1, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = -1.0; W[2, 3] = 1.0
    
    # M5 = (a00 + a01) * b11
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = 1.0; W[1, 4] = 1.0
    
    # M6 = (a10 - a00) * (b00 + b10)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[2, 5] = 1.0
    W[3, 5] = 1.0
    
    # M7 = (a01 - a11) * (b01 + b11)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[1, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = -1.0
    
    return U, V, W

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    u_val = U[i * m + j, c]
                    v_val = V[j * p + k, c]
                    w_val = W[k * n + i, c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        val += u_val * v_val * w_val
                if abs(val - 1.0) > tol:
                    return False
    return True

def _block_decomposition_strassen(n, m, p):
    """
    Block decomposition using 2x2x2 Strassen blocks + schoolbook for remainder.
    p=15: 7 blocks of 2 + 1 block of 1. Rank = 7*7 + 4 = 53.
    """
    if n != 2 or m != 2:
        return None
    
    block_sizes = []
    remaining = p
    while remaining > 0:
        if remaining >= 2:
            block_sizes.append(2)
            remaining -= 2
        else:
            block_sizes.append(1)
            remaining -= 1
            
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Schoolbook for 2x2x1
    U_1 = np.zeros((4, 4), dtype=np.float64)
    V_1 = np.zeros((2, 4), dtype=np.float64)
    W_1 = np.zeros((2, 4), dtype=np.float64)
    for r in range(4):
        i, j = divmod(r, 2)
        U_1[i*2+j, r] = 1.0
        V_1[j, r] = 1.0
        W_1[i, r] = 1.0
    
    total_rank = sum(7 if bs == 2 else 4 for bs in block_sizes)
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    k_offset = 0
    
    for bs in block_sizes:
        if bs == 2:
            for r in range(7):
                U[:, col_offset] = U_s[:, r]
                V[0 * p + k_offset, col_offset] = V_s[0, r]
                V[0 * p + k_offset + 1, col_offset] = V_s[1, r]
                V[1 * p + k_offset, col_offset] = V_s[2, r]
                V[1 * p + k_offset + 1, col_offset] = V_s[3, r]
                W[k_offset * n + 0, col_offset] = W_s[0, r]
                W[k_offset * n + 1, col_offset] = W_s[1, r]
                W[(k_offset + 1) * n + 0, col_offset] = W_s[2, r]
                W[(k_offset + 1) * n + 1, col_offset] = W_s[3, r]
                col_offset += 1
            k_offset += 2
        elif bs == 1:
            for r in range(4):
                U[:, col_offset] = U_1[:, r]
                V[0 * p + k_offset, col_offset] = V_1[0, r]
                V[1 * p + k_offset, col_offset] = V_1[1, r]
                W[k_offset * n + 0, col_offset] = W_1[0, r]
                W[k_offset * n + 1, col_offset] = W_1[1, r]
                col_offset += 1
            k_offset += 1
            
    return U, V, W

def _global_iterative_refinement(n, m, p, rank, deadline, rng_seed=42):
    """
    Global iterative refinement for the full 2x2x15 tensor.
    Uses Bini's method: alternately solve for U, V, W rows given the others.
    """
    start = time.time()
    rng = np.random.default_rng(rng_seed)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Try multiple initializations
    n_init = 20
    max_iters = 500
    
    for trial in range(n_init):
        if time.time() > deadline:
            break
        
        # Initialize
        U = rng.standard_normal((n * m, rank))
        V = rng.standard_normal((m * p, rank))
        W = rng.standard_normal((p * n, rank))
        
        # Normalize columns to prevent blowup
        for c in range(rank):
            norm_u = np.linalg.norm(U[:, c])
            norm_v = np.linalg.norm(V[:, c])
            norm_w = np.linalg.norm(W[:, c])
            if norm_u > 1e-12:
                U[:, c] /= norm_u
            if norm_v > 1e-12:
                V[:, c] /= norm_v
            if norm_w > 1e-12:
                W[:, c] /= norm_w
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    target = np.zeros(rank)
                    for k in range(p):
                        target += V[j * p + k, :] * W[k * n + i, :]
                    dot = target @ target
                    if dot > 1e-12:
                        U[i * m + j, :] = target / dot
                    else:
                        U[i * m + j, :] = 0.0
            
            # Update V
            for j in range(m):
                for k in range(p):
                    target = np.zeros(rank)
                    for i in range(n):
                        target += U[i * m + j, :] * W[k * n + i, :]
                    dot = target @ target
                    if dot > 1e-12:
                        V[j * p + k, :] = target / dot
                    else:
                        V[j * p + k, :] = 0.0
            
            # Update W
            for i in range(n):
                for k in range(p):
                    target = np.zeros(rank)
                    for j in range(m):
                        target += U[i * m + j, :] * V[j * p + k, :]
                    dot = target @ target
                    if dot > 1e-12:
                        W[k * n + i, :] = target / dot
                    else:
                        W[k * n + i, :] = 0.0
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        err += (val - 1.0) ** 2
            err = np.sqrt(err)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-8:
            break
    
    if best_err < 1e-8 and best_U is not None:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err

def _construct_rank_r(n, m, p, rank, deadline):
    """Try to construct a rank-r decomposition using global iterative refinement."""
    if rank >= n * m * p:
        # Schoolbook
        return _schoolbook(n, m, p)
    
    U, V, W, err = _global_iterative_refinement(n, m, p, rank, deadline, rng_seed=rank * 1000 + 42)
    if U is not None:
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Best result so far
    best_result = None
    best_rank = float('inf')
    
    # Option 1: Strassen block decomposition (rank 53)
    U, V, W = _block_decomposition_strassen(n, m, p)
    if U is not None and _is_valid(U, V, W, n, m, p):
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        best_rank = U.shape[1]
    
    # Option 2: Try global iterative refinement for lower ranks
    # Try ranks from 49 down to 40
    for rank in [49, 48, 47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline - 10.0:
            break
        
        if rank >= best_rank:
            continue
        
        U, V, W = _construct_rank_r(n, m, p, rank, deadline - 5.0)
        if U is not None and _is_valid(U, V, W, n, m, p):
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
