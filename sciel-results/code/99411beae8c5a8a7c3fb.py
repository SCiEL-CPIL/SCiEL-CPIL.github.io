import time
import numpy as np

DIMENSIONS = [2, 5, 5]
DEADLINE_SECONDS = 135.0  # Safety margin before 150s kill


def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T


def _tensor_error(U, V, W, T):
    """
    Compute ||U V^T W^T - T||_F^2
    T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    """
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.sum((reconstructed - T) ** 2)


def _max_abs_residual(U, V, W, T):
    """Compute max absolute residual."""
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(reconstructed - T))


def _schoolbook_solution(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2x2_decomp():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    U: (i, j) -> i*2+j
    V: (j, k) -> j*2+k
    W: (k, i) -> k*2+i
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _block_strassen_init(n, m, p, rank, seed):
    """
    Initialize U, V, W using a block structure.
    Splits m=5 into 2+3, p=5 into 2+3.
    Uses Strassen for the 2x2x2 block.
    Uses Schoolbook for the rest.
    Pads with zeros or random noise to match 'rank'.
    """
    rng = np.random.RandomState(seed)
    
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # We need to map the block decomposition into the global indices.
    # Global U index: i*m + j
    # Global V index: j*p + k
    # Global W index: k*n + i
    
    # Block 1: i in [0,2), j in [0,2), k in [0,2)
    # This is a 2x2x2 multiplication.
    # We use Strassen for this.
    U_s, V_s, W_s = _strassen_2x2x2_decomp()
    
    col = 0
    # Map Strassen indices to global indices
    # Strassen U row r_s corresponds to (i_s, j_s). Global row = i_s*m + j_s.
    # Strassen V row r_s corresponds to (j_s, k_s). Global row = j_s*p + k_s.
    # Strassen W row r_s corresponds to (k_s, i_s). Global row = k_s*n + i_s.
    
    for r in range(7):
        if col >= rank: break
        # U
        for i_s in range(2):
            for j_s in range(2):
                idx_u = i_s * 2 + j_s
                global_u = i_s * m + j_s
                U[global_u, col] = U_s[idx_u, r]
        
        # V
        for j_s in range(2):
            for k_s in range(2):
                idx_v = j_s * 2 + k_s
                global_v = j_s * p + k_s
                V[global_v, col] = V_s[idx_v, r]
        
        # W
        for k_s in range(2):
            for i_s in range(2):
                idx_w = k_s * 2 + i_s
                global_w = k_s * n + i_s
                W[global_w, col] = W_s[idx_w, r]
        
        col += 1
    
    # Block 2: i in [0,2), j in [2,5), k in [0,2)
    # Dimensions: A(2x3), B(3x2), C(2x2)
    # Schoolbook rank: 2*3*2 = 12
    # Indices:
    # i: 0,1
    # j: 2,3,4
    # k: 0,1
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                if col >= rank: break
                global_u = i * m + j
                global_v = j * p + k
                global_w = k * n + i
                U[global_u, col] = 1.0
                V[global_v, col] = 1.0
                W[global_w, col] = 1.0
                col += 1
    
    # Block 3: i in [0,2), j in [0,2), k in [2,5)
    # Dimensions: A(2x2), B(2x3), C(2x3)
    # Schoolbook rank: 2*2*3 = 12
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                if col >= rank: break
                global_u = i * m + j
                global_v = j * p + k
                global_w = k * n + i
                U[global_u, col] = 1.0
                V[global_v, col] = 1.0
                W[global_w, col] = 1.0
                col += 1
    
    # Block 4: i in [0,2), j in [2,5), k in [2,5)
    # Dimensions: A(2x3), B(3x3), C(2x3)
    # Schoolbook rank: 2*3*3 = 18
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                if col >= rank: break
                global_u = i * m + j
                global_v = j * p + k
                global_w = k * n + i
                U[global_u, col] = 1.0
                V[global_v, col] = 1.0
                W[global_w, col] = 1.0
                col += 1
    
    # If rank is larger than the structured part, add random noise to remaining columns
    if col < rank:
        U[:, col:] = rng.randn(n*m, rank-col) * 0.1
        V[:, col:] = rng.randn(m*p, rank-col) * 0.1
        W[:, col:] = rng.randn(p*n, rank-col) * 0.1
        
    # Normalize columns
    for r in range(rank):
        max_u = np.max(np.abs(U[:, r]))
        max_v = np.max(np.abs(V[:, r]))
        max_w = np.max(np.abs(W[:, r]))
        if max_u > 0: U[:, r] /= max_u
        if max_v > 0: V[:, r] /= max_v
        if max_w > 0: W[:, r] /= max_w
        
    return U, V, W


def _als_refine(U, V, W, T, Nu, Nv, Nw, max_iter=200, tol=1e-10, deadline=None):
    """
    Alternating Least Squares refinement with best-tracking.
    U: (Nu, R), V: (Nv, R), W: (Nw, R)
    T: (Nu, Nv, Nw)
    """
    R = U.shape[1]
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 2:
            break
        
        # Update U
        M_U = (V[:, :, None] * W[None, :, :]).reshape(Nv * Nw, R)
        for u in range(Nu):
            b_u = T[u, :, :].flatten()
            try:
                U[u, :] = np.linalg.lstsq(M_U.T, b_u, rcond=None)[0]
            except:
                pass
        
        # Update V
        M_V = (U[:, :, None] * W[None, :, :]).reshape(Nu * Nw, R)
        for v in range(Nv):
            b_v = T[:, v, :].flatten()
            try:
                V[v, :] = np.linalg.lstsq(M_V.T, b_v, rcond=None)[0]
            except:
                pass
        
        # Update W
        M_W = (U[:, :, None] * V[None, :, :]).reshape(Nu * Nv, R)
        for w in range(Nw):
            b_w = T[:, :, w].flatten()
            try:
                W[w, :] = np.linalg.lstsq(M_W.T, b_w, rcond=None)[0]
            except:
                pass
        
        err = _max_abs_residual(U, V, W, T)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err


def _round_to_half_int(arr, tol=1e-6):
    """Round array to nearest half-integer if close enough."""
    rounded = np.round(arr * 2) / 2
    if np.max(np.abs(arr - rounded)) < tol:
        return rounded
    return arr


def _try_rank(n, m, p, rank, T, Nu, Nv, Nw, seed=0, time_limit=2.0, use_block_init=False):
    """
    Attempt to find a decomposition of the given rank.
    """
    rng = np.random.RandomState(seed)
    
    if use_block_init and rank >= 49:
        U, V, W = _block_strassen_init(n, m, p, rank, seed)
    else:
        U = rng.randn(Nu, rank)
        V = rng.randn(Nv, rank)
        W = rng.randn(Nw, rank)
        
        for r in range(rank):
            max_u = np.max(np.abs(U[:, r]))
            max_v = np.max(np.abs(V[:, r]))
            max_w = np.max(np.abs(W[:, r]))
            if max_u > 0: U[:, r] /= max_u
            if max_v > 0: V[:, r] /= max_v
            if max_w > 0: W[:, r] /= max_w

    # Phase 1: Coarse ALS
    U, V, W, err = _als_refine(U, V, W, T, Nu, Nv, Nw, max_iter=100, tol=1e-9, deadline=time.time() + time_limit * 0.6)
    
    if err < 1e-7:
        # Phase 2: Fine ALS
        U, V, W, err = _als_refine(U, V, W, T, Nu, Nv, Nw, max_iter=100, tol=1e-12, deadline=time.time() + time_limit * 0.3)
    
    if err < 1e-8:
        # Try rounding to half-integers
        U_r = _round_to_half_int(U)
        V_r = _round_to_half_int(V)
        W_r = _round_to_half_int(W)
        err_r = _max_abs_residual(U_r, V_r, W_r, T)
        if err_r < 1e-6:
            return U_r, V_r, W_r, err_r
        return U, V, W, err
    
    return None


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    Nu = n * m  # 10
    Nv = m * p  # 25
    Nw = p * n  # 10
    
    T = _build_tensor(n, m, p)
    
    U_fb, V_fb, W_fb = _schoolbook_solution(n, m, p)
    U_best, V_best, W_best = U_fb.copy(), V_fb.copy(), W_fb.copy()
    best_rank = n * m * p  # 50
    
    # Try ranks from 46 to 50
    candidate_ranks = [46, 47, 48, 49, 50]
    
    for rank in candidate_ranks:
        if time.time() - start_time > DEADLINE_SECONDS - 8:
            break
        
        # More seeds for most promising ranks
        num_seeds = 80 if rank == 46 else 40 if rank == 47 else 20
        
        found = False
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE_SECONDS - 4:
                break
            
            remaining_seeds = num_seeds - seed
            time_limit = min(2.0, (DEADLINE_SECONDS - 4 - (time.time() - start_time)) / max(remaining_seeds, 1))
            if time_limit < 0.3:
                break
                
            try:
                use_block = rank >= 49
                result = _try_rank(n, m, p, rank, T, Nu, Nv, Nw, seed=seed * 7 + 3, time_limit=time_limit, use_block_init=use_block)
                if result is not None:
                    U_new, V_new, W_new, err = result
                    if _tensor_error(U_new, V_new, W_new, T) < 1e-6:
                        U_best, V_best, W_best = U_new, V_new, W_new
                        best_rank = rank
                        found = True
                        break
            except Exception:
                continue
        
        if found:
            break
    
    # Final rounding to half-integers
    U_best = _round_to_half_int(U_best)
    V_best = _round_to_half_int(V_best)
    W_best = _round_to_half_int(W_best)
    
    # Verify; fall back to schoolbook if rounding broke it
    if _tensor_error(U_best, V_best, W_best, T) > 1e-6:
        U_best, V_best, W_best = U_fb.copy(), V_fb.copy(), W_fb.copy()
    
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
