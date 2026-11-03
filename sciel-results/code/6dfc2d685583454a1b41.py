import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    # Build target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1
                
    T_flat = T.flatten()
    
    def check_decomposition(U, V, W, tol=1e-9):
        r = U.shape[1]
        # Reconstruct using einsum for efficiency
        T_recon = np.einsum('il,jl,kl->ijk', U, V, W)
        max_diff = np.max(np.abs(T_recon - T))
        return max_diff < tol
    
    def try_integerize(W_float):
        """Try to convert float W to integer or half-integer"""
        W_int = np.round(W_float).astype(np.int64)
        if check_decomposition(U_current, V_current, W_int):
            return W_int, "int"
        W_half = np.round(W_float * 2) / 2.0
        if check_decomposition(U_current, V_current, W_half):
            return W_half, "half"
        return None, None
    
    # Initialize with schoolbook (Rank 60)
    rank_schoolbook = n * m * p
    U_school = np.zeros((dim_u, rank_schoolbook), dtype=np.int64)
    V_school = np.zeros((dim_v, rank_schoolbook), dtype=np.int64)
    W_school = np.zeros((dim_w, rank_schoolbook), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Time management
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # Precompute M matrix structure for speed
    # M maps W_flat to T_flat
    # M[row_idx, col_idx] where row_idx = (i*dim_v + j)*dim_w + k
    # col_idx = w_idx * rank + l
    # M[row_idx, col_idx] = U[i, l] * V[j, l]
    
    def build_M(U, V, rank):
        """Build the M matrix for given U, V and rank"""
        M = np.zeros((dim_u * dim_v * dim_w, dim_w * rank), dtype=np.float64)
        for i in range(dim_u):
            for j in range(dim_v):
                row_base = (i * dim_v + j) * dim_w
                for k in range(dim_w):
                    row_idx = row_base + k
                    M[row_idx, k*rank : (k+1)*rank] = U[i, :] * V[j, :]
        return M
    
    # Search strategy: try various ranks with diverse initializations
    # Focus on ranks that might be achievable
    
    # Try a range of ranks, prioritizing those closer to known bounds
    # Lower bound is 20, schoolbook is 60
    # Try ranks in descending order from 59 down to 30
    
    ranks_to_try = list(range(59, 29, -1))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try multiple diverse initializations
        max_attempts = 8
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Different initialization strategies
            if attempt % 3 == 0:
                # Small integers
                U = np.random.randint(-2, 3, size=(dim_u, target_rank))
                V = np.random.randint(-2, 3, size=(dim_v, target_rank))
            elif attempt % 3 == 1:
                # Binary with some -1
                U = np.random.randint(0, 2, size=(dim_u, target_rank))
                V = np.random.randint(0, 2, size=(dim_v, target_rank))
                # Add some -1 for diversity
                for _ in range(target_rank // 3):
                    idx = random.randint(0, dim_u - 1)
                    jdx = random.randint(0, target_rank - 1)
                    U[idx, jdx] = -1
                    idx2 = random.randint(0, dim_v - 1)
                    jdx2 = random.randint(0, target_rank - 1)
                    V[idx2, jdx2] = -1
            else:
                # Gaussian, then round
                U = np.random.randn(dim_u, target_rank)
                V = np.random.randn(dim_v, target_rank)
                U = np.round(U).astype(np.int64)
                V = np.round(V).astype(np.int64)
                
            # Build M and solve
            M = build_M(U, V, target_rank)
            
            try:
                W_flat, residuals, rank_M, s = np.linalg.lstsq(M, T_flat, rcond=None)
            except:
                continue
                
            W_float = W_flat.reshape(dim_w, target_rank)
            
            # Check if solution is exact
            if check_decomposition(U, V, W_float):
                # Try to integerize
                W_int, wtype = try_integerize(W_float)
                if W_int is not None:
                    best_U, best_V, best_W = U, V, W_int
                    best_rank = target_rank
                    break
                else:
                    # Accept float if it's very close
                    if check_decomposition(U, V, W_float, tol=1e-6):
                        best_U, best_V, best_W = U, V, W_float
                        best_rank = target_rank
                        break
            else:
                # Check if close, then try integerization
                W_int, wtype = try_integerize(W_float)
                if W_int is not None:
                    best_U, best_V, best_W = U, V, W_int
                    best_rank = target_rank
                    break

    # Final check and return
    # Ensure we have a valid solution
    if not check_decomposition(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_schoolbook
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
