"""Strassen-based rank reduction for 4x4 * 4x9 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    The tensor T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,9)
    must be decomposed as:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Returns dict with U, V, W as lists of lists with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook baseline (rank 144)
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    def is_valid(U, V, W, rank):
        """Check if the decomposition exactly reproduces T[i,j,k]=1 for all i,j,k."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i*m+j], V_np[j*p+k])
                    # Wait, that's not right. Let me fix.
                    val = 0.0
                    for r in range(rank):
                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def is_valid_fast(U_np, V_np, W_np, rank):
        """Vectorized validation."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Strassen 2x2x2 coefficients
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    strassen_w = [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ]
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    U_44 = []
    V_44 = []
    W_44 = []
    
    for s in range(7):
        for t in range(7):
            u_vec = [0] * 16
            v_vec = [0] * 16
            w_vec = [0] * 16
            
            u_block = strassen_u[s]
            v_block = strassen_v[s]
            w_block = strassen_w[s]
            u_in = strassen_u[t]
            v_in = strassen_v[t]
            w_in = strassen_w[t]
            
            for bi in range(2):
                for bj in range(2):
                    a_coeff = u_block[bi * 2 + bj]
                    if a_coeff == 0:
                        continue
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            in_coeff = u_in[local_idx]
                            if in_coeff == 0:
                                continue
                            gi = 2 * bi + li
                            gj = 2 * bj + lj
                            gidx = gi * 4 + gj
                            u_vec[gidx] += a_coeff * in_coeff
            
            for bi in range(2):
                for bj in range(2):
                    b_coeff = v_block[bi * 2 + bj]
                    if b_coeff == 0:
                        continue
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            in_coeff = v_in[local_idx]
                            if in_coeff == 0:
                                continue
                            gi = 2 * bi + li
                            gj = 2 * bj + lj
                            gidx = gi * 4 + gj
                            v_vec[gidx] += b_coeff * in_coeff
            
            for bi in range(2):
                for bj in range(2):
                    c_coeff = w_block[bi * 2 + bj]
                    if c_coeff == 0:
                        continue
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            in_coeff = w_in[local_idx]
                            if in_coeff == 0:
                                continue
                            gi_row = 2 * bi + li
                            gi_col = 2 * bj + lj
                            gidx = gi_col * 4 + gi_row
                            w_vec[gidx] += c_coeff * in_coeff
            
            U_44.append(u_vec)
            V_44.append(v_vec)
            W_44.append(w_vec)
    
    U_44_np = np.array(U_44, dtype=float).T
    V_44_np = np.array(V_44, dtype=float).T
    W_44_np = np.array(W_44, dtype=float).T
    
    # Verify 4x4x4 Strassen
    valid_44 = is_valid_fast(U_44_np, V_44_np, W_44_np, 49)
    
    if not valid_44:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Build the 102-rank solution as starting point
    rank_total = 49 + 49 + 4
    U_full = np.zeros((16, rank_total))
    V_full = np.zeros((36, rank_total))
    W_full = np.zeros((36, rank_total))
    
    for r in range(49):
        U_full[:, r] = U_44_np[:, r]
        for j in range(4):
            for k in range(4):
                V_full[j * 9 + k, r] = V_44_np[j * 4 + k, r]
        for k in range(4):
            for i in range(4):
                W_full[k * 4 + i, r] = W_44_np[k * 4 + i, r]
    
    offset = 49
    for r in range(49):
        U_full[:, offset + r] = U_44_np[:, r]
        for j in range(4):
            for k in range(4):
                V_full[j * 9 + (4 + k), offset + r] = V_44_np[j * 4 + k, r]
        for k in range(4):
            for i in range(4):
                W_full[(4 + k) * 4 + i, offset + r] = W_44_np[k * 4 + i, r]
    
    offset2 = 98
    for i in range(4):
        for j in range(4):
            U_full[i * 4 + j, offset2] = 1
            V_full[j * 9 + 8, offset2] = 1
            W_full[8 * 4 + i, offset2] = 1
    
    if not is_valid_fast(U_full, V_full, W_full, rank_total):
        return {"U": best_U, "V": best_V, "W": best_W}
    
    best_rank = rank_total
    best_U = U_full.copy()
    best_V = V_full.copy()
    best_W = W_full.copy()
    
    # Improved ALS refinement
    def als_refine(U_np, V_np, W_np, target_rank, max_iter=500, tol=1e-12):
        U_cur = U_np.copy()
        V_cur = V_np.copy()
        W_cur = W_np.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Solve for U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V_cur[j*p+k, :target_rank] * W_cur[k*n+i, :target_rank]
                    try:
                        u_sol = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        U_cur[i*m+j, :target_rank] = u_sol
                    except:
                        pass
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, target_rank))
                    for i in range(n):
                        A[i, :] = U_cur[i*m+j, :target_rank] * W_cur[k*n+i, :target_rank]
                    try:
                        v_sol = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        V_cur[j*p+k, :target_rank] = v_sol
                    except:
                        pass
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        A[j, :] = U_cur[i*m+j, :target_rank] * V_cur[j*p+k, :target_rank]
                    try:
                        w_sol = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        W_cur[k*n+i, :target_rank] = w_sol
                    except:
                        pass
            
            # Check convergence
            max_res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_cur[i*m+j] * V_cur[j*p+k] * W_cur[k*n+i])
                        res = val - 1.0
                        max_res = max(max_res, abs(res))
            
            if max_res < tol:
                return U_cur, V_cur, W_cur, True
        
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_cur[i*m+j] * V_cur[j*p+k] * W_cur[k*n+i])
                    res = val - 1.0
                    max_res = max(max_res, abs(res))
        
        return U_cur, V_cur, W_cur, max_res < tol

    # Try reducing rank
    np.random.seed(42)
    for target_rank in range(best_rank - 1, max(best_rank - 20, 80), -1):
        if time.time() > deadline - 5:
            break
        
        success = False
        for attempt in range(15):
            if time.time() > deadline - 5:
                break
            
            U_init = best_U.copy()
            V_init = best_V.copy()
            W_init = best_W.copy()
            
            cols_to_zero = np.random.choice(best_rank, best_rank - target_rank, replace=False)
            U_init[:, cols_to_zero] = 0
            V_init[:, cols_to_zero] = 0
            W_init[:, cols_to_zero] = 0
            
            noise_scale = 0.01
            U_init += np.random.randn(16, best_rank) * noise_scale
            V_init += np.random.randn(36, best_rank) * noise_scale
            W_init += np.random.randn(36, best_rank) * noise_scale
            
            U_new, V_new, W_new, converged = als_refine(
                U_init, V_init, W_init, target_rank, max_iter=200, tol=1e-10
            )
            
            if converged and is_valid_fast(U_new, V_new, W_new, target_rank):
                best_rank = target_rank
                best_U = U_new.copy()
                best_V = V_new.copy()
                best_W = W_new.copy()
                success = True
                break
        
        if success:
            continue
        else:
            break
    
    # Convert to lists
    best_U_list = best_U.tolist()
    best_V_list = best_V.tolist()
    best_W_list = best_W.tolist()
    
    # Final validation
    if not is_valid(best_U_list, best_V_list, best_W_list, best_rank):
        # Fallback to schoolbook
        best_rank = n * m * p
        best_U = [[0] * best_rank for _ in range(n * m)]
        best_V = [[0] * best_rank for _ in range(m * p)]
        best_W = [[0] * best_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j][col] = 1
                    best_V[j * p + k][col] = 1
                    best_W[k * n + i][col] = 1
                    col += 1
        best_U_list = best_U
        best_V_list = best_V
        best_W_list = best_W
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
# EVOLVE-BLOCK-END
