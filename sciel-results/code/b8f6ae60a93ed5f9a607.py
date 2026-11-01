import random
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def validate_exact(U, V, W, n, m, p):
    """Validate with exact fraction arithmetic."""
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = Fraction(0, 1)
                for c in range(r):
                    val += Fraction(U[idx_u][c]) * Fraction(V[idx_v][c]) * Fraction(W[idx_w][c])
                if val != Fraction(1, 1):
                    return False
    return True

def validate_float(U, V, W, n, m, p, tol=1e-9):
    """Fast float validation."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = W.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def schoolbook_decomposition(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def build_strassen_blocks(n, m, p):
    """
    Build a decomposition using Strassen's algorithm for 2x2x2 blocks
    where possible, and schoolbook for the rest.
    """
    terms = []
    
    def add_strassen(i_start, j_start, k_start):
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))

    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))

    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_schoolbook(0, 2, 0, 2, 4, 5)
    
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_schoolbook(0, 2, 2, 4, 4, 5)
    
    add_schoolbook(0, 2, 4, 5, 0, 2)
    add_schoolbook(0, 2, 4, 5, 2, 4)
    add_schoolbook(0, 2, 4, 5, 4, 5)

    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
            
    return U, V, W

def tensor_error(U, V, W, n, m, p):
    """Calculate L2 error of the tensor decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = W.shape[1]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                error += (val - 1.0) ** 2
    return error

def svd_based_init(n, m, p, rank, seed=0):
    """
    Initialize U, V, W using SVD of tensor matricizations.
    The all-1 tensor of shape (2,5,5) can be decomposed via its matricizations.
    """
    np.random.seed(seed)
    
    # For the all-1 tensor, any rank-r decomposition works if we can find
    # U, V, W such that the contraction gives 1 everywhere.
    # Use structured initialization based on the tensor's symmetry.
    
    # Initialize with small random values scaled appropriately
    scale = 1.0 / (rank ** 0.5)
    U = np.random.randn(n * m, rank) * scale
    V = np.random.randn(m * p, rank) * scale
    W = np.random.randn(p * n, rank) * scale
    
    # Try to align with the structure: for each j, the U entries for row j
    # should correlate with V entries for row j
    for j in range(m):
        # U rows: i*m + j for i in range(n)
        # V rows: j*p + k for k in range(p)
        u_rows = [i * m + j for i in range(n)]
        v_rows = [j * p + k for k in range(p)]
        
        # Make U and V entries for same j more correlated
        for c in range(rank):
            avg_u = np.mean(U[u_rows, c])
            avg_v = np.mean(V[v_rows, c])
            U[u_rows, c] += 0.1 * (avg_u - U[u_rows, c])
            V[v_rows, c] += 0.1 * (avg_v - V[v_rows, c])
    
    return U, V, W

def try_optimize_rank(n, m, p, rank, deadline, max_iter=2000, lr=0.01, init_seed=0):
    """
    Try to find a decomposition of given rank using gradient descent.
    Returns (U, V, W) as lists if found, else None.
    """
    U, V, W = svd_based_init(n, m, p, rank, seed=init_seed)
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        # Compute error and gradients
        error = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                    err = val - 1.0
                    error += err * err
                    
                    # Gradients
                    dU[idx_u, :] += 2 * err * (V[idx_v, :] * W[idx_w, :])
                    dV[idx_v, :] += 2 * err * (U[idx_u, :] * W[idx_w, :])
                    dW[idx_w, :] += 2 * err * (U[idx_u, :] * V[idx_v, :])
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-12:
            break
        
        # Adaptive learning rate with cosine annealing
        progress = it / max_iter
        step_size = lr * (0.5 + 0.5 * np.cos(np.pi * progress))
        
        U -= step_size * dU
        V -= step_size * dV
        W -= step_size * dW
        
        # Occasionally restart from best if stuck
        if it > 0 and it % 300 == 0 and error > 1e-4:
            noise_scale = 0.01 * (1 + error)
            U = best_U + np.random.randn(n * m, rank) * noise_scale
            V = best_V + np.random.randn(m * p, rank) * noise_scale
            W = best_W + np.random.randn(p * n, rank) * noise_scale
    
    # Final check on best found
    if best_error < 1e-8:
        # Try to snap to rational values with increasing precision
        for mult in [10, 100, 1000, 10000]:
            U_snap = np.round(best_U * mult) / mult
            V_snap = np.round(best_V * mult) / mult
            W_snap = np.round(best_W * mult) / mult
            
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            
            if validate_exact(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    
    return None

def try_merge_terms(U, V, W, n, m, p, tol=1e-10):
    """
    Try to reduce rank by finding linear dependencies among rank-1 terms.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = W.shape[1]
    
    kept_U = []
    kept_V = []
    kept_W = []
    
    for c in range(r):
        is_dup = False
        for c2 in range(len(kept_U)):
            u1, u2 = U[:, c], U[:, c2]
            v1, v2 = V[:, c], V[:, c2]
            w1, w2 = W[:, c], W[:, c2]
            
            # Check if term c is a scalar multiple of term c2
            nonzero_u2 = np.abs(u2) > tol
            if np.sum(nonzero_u2) > 0:
                denom_u = np.dot(u2[nonzero_u2], u2[nonzero_u2])
                if denom_u > tol:
                    alpha = np.dot(u1[nonzero_u2], u2[nonzero_u2]) / denom_u
                    if np.allclose(u1, alpha * u2, atol=tol, rtol=tol):
                        nonzero_v2 = np.abs(v2) > tol
                        if np.sum(nonzero_v2) > 0:
                            denom_v = np.dot(v2[nonzero_v2], v2[nonzero_v2])
                            if denom_v > tol:
                                beta = np.dot(v1[nonzero_v2], v2[nonzero_v2]) / denom_v
                                if np.allclose(v1, beta * v2, atol=tol, rtol=tol):
                                    nonzero_w2 = np.abs(w2) > tol
                                    if np.sum(nonzero_w2) > 0:
                                        denom_w = np.dot(w2[nonzero_w2], w2[nonzero_w2])
                                        if denom_w > tol:
                                            gamma = np.dot(w1[nonzero_w2], w2[nonzero_w2]) / denom_w
                                            if np.allclose(w1, gamma * w2, atol=tol, rtol=tol):
                                                if abs(alpha * beta * gamma - 1.0) < tol:
                                                    is_dup = True
                                                    break
        if not is_dup:
            kept_U.append(U[:, c])
            kept_V.append(V[:, c])
            kept_W.append(W[:, c])
    
    if len(kept_U) < r:
        U_new = np.column_stack(kept_U)
        V_new = np.column_stack(kept_V)
        W_new = np.column_stack(kept_W)
        if validate_float(U_new, V_new, W_new, n, m, p, tol=1e-8):
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer
    
    # Build Strassen-based decomposition as fallback
    U_strassen, V_strassen, W_strassen = build_strassen_blocks(n, m, p)
    best_result = (U_strassen, V_strassen, W_strassen)
    best_rank = len(W_strassen[0])
    
    # Verify the Strassen decomposition is valid
    if not validate_float(U_strassen, V_strassen, W_strassen, n, m, p):
        U_strassen, V_strassen, W_strassen = schoolbook_decomposition(n, m, p)
        best_result = (U_strassen, V_strassen, W_strassen)
        best_rank = len(W_strassen[0])
    
    # Try to find lower ranks using gradient optimization with SVD-inspired init
    candidate_ranks = [30, 28, 26, 24, 22, 20, 18, 16]
    
    for rank in candidate_ranks:
        if rank >= best_rank:
            continue
        if time.time() > deadline - 30:
            break
            
        # Try multiple random starts
        found = False
        for attempt in range(6):
            if time.time() > deadline - 15:
                break
                
            result = try_optimize_rank(n, m, p, rank, deadline, max_iter=1200, lr=0.02, init_seed=attempt)
            if result:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = rank
                found = True
                break
        
        if found:
            # Try to further reduce by merging terms
            merged = try_merge_terms(best_result[0], best_result[1], best_result[2], n, m, p)
            if merged:
                U_m, V_m, W_m = merged
                if len(W_m[0]) < best_rank:
                    best_result = (U_m, V_m, W_m)
                    best_rank = len(W_m[0])
            continue
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
