import time
import numpy as np
from scipy.optimize import minimize

DEADLINE_S = 110.0
N, M, P = 4, 5, 5

def _validate(U, V, W, tol=1e-9):
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_vectors():
    return [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]

def _build_base_85():
    """Build the base rank-85 decomposition using Strassen for 4x4x4 block."""
    n, m, p = N, M, P
    R = 85
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    c = 0
    
    s2 = _strassen_2x2_vectors()
    
    def map_2x2_to_4x4(vec, row_off, col_off):
        out = [0.0] * 16
        for r in range(2):
            for c in range(2):
                out[(row_off + r) * 4 + (col_off + c)] = vec[r * 2 + c]
        return out

    calls = [
        ([ (0,0,1), (2,2,1) ], [ (0,0,1), (2,2,1) ], [ (0,0,1), (2,2,1) ]),
        ([ (2,0,1), (2,2,1) ], [ (0,0,1) ], [ (2,0,1), (2,2,-1) ]),
        ([ (0,0,1) ], [ (0,2,1), (2,2,-1) ], [ (0,2,1), (2,2,1) ]),
        ([ (2,2,1) ], [ (2,0,1), (0,0,-1) ], [ (0,0,1), (2,0,1) ]),
        ([ (0,0,1), (0,2,1) ], [ (2,2,1) ], [ (0,0,-1), (0,2,1) ]),
        ([ (2,0,1), (0,0,-1) ], [ (0,0,1), (0,2,1) ], [ (2,2,1) ]),
        ([ (0,2,1), (2,2,-1) ], [ (2,0,1), (2,2,1) ], [ (0,0,1) ])
    ]
    
    for u_blocks, v_blocks, w_blocks in calls:
        for u_vec, v_vec, w_vec in s2:
            u_full = [0.0] * 16
            v_full = [0.0] * 16
            w_full = [0.0] * 16
            
            for row_off, col_off, sign in u_blocks:
                mapped = map_2x2_to_4x4(u_vec, row_off, col_off)
                for i in range(16):
                    u_full[i] += sign * mapped[i]
                    
            for row_off, col_off, sign in v_blocks:
                mapped = map_2x2_to_4x4(v_vec, row_off, col_off)
                for i in range(16):
                    v_full[i] += sign * mapped[i]
                    
            for row_off, col_off, sign in w_blocks:
                mapped = map_2x2_to_4x4(w_vec, row_off, col_off)
                for i in range(16):
                    w_full[i] += sign * mapped[i]
            
            for i in range(4):
                for j in range(4):
                    u_idx = i * 5 + j
                    u_val = u_full[i * 4 + j]
                    if u_val != 0:
                        U[u_idx, c] += u_val
                        
            for j in range(4):
                for k in range(4):
                    v_idx = j * 5 + k
                    v_val = v_full[j * 4 + k]
                    if v_val != 0:
                        V[v_idx, c] += v_val
                        
            for k in range(4):
                for i in range(4):
                    w_idx = k * 4 + i
                    w_val = w_full[k * 4 + i]
                    if w_val != 0:
                        W[w_idx, c] += w_val
                        
            c += 1
            
    # Part 2: A11 B112 (4x4x1) Rank 16
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, c] = 1.0
            V[j * 5 + 4, c] = 1.0
            W[4 * 4 + i, c] = 1.0
            c += 1
            
    # Part 3: A12 B21 (4x1x5) Rank 20
    for i in range(4):
        for k in range(5):
            U[i * 5 + 4, c] = 1.0
            V[4 * 5 + k, c] = 1.0
            W[k * 4 + i, c] = 1.0
            c += 1
            
    return U, V, W

def _compute_loss_and_grad(U, V, W):
    """Compute loss and gradients for the tensor decomposition constraint."""
    n, m, p = N, M, P
    R = U.shape[1]
    loss = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = s - 1.0
                loss += err * err
                
                grad_U[i*m+j, :] += 2.0 * err * V[j*p+k, :] * W[k*n+i, :]
                grad_V[j*p+k, :] += 2.0 * err * U[i*m+j, :] * W[k*n+i, :]
                grad_W[k*n+i, :] += 2.0 * err * U[i*m+j, :] * V[j*p+k, :]
    
    return loss, grad_U, grad_V, grad_W

def _try_reduce_rank(U, V, W, target_rank, max_iter=200, lr=0.01, time_limit=20.0, n_restarts=5):
    """Try to reduce rank by optimizing with better initialization from existing decomposition."""
    t0 = time.time()
    n, m, p = N, M, P
    R = U.shape[1]
    
    best_loss = np.inf
    best_U, best_V, best_W = None, None, None
    
    for restart in range(n_restarts):
        if time.time() - t0 > time_limit:
            break
        
        # Initialize from the existing decomposition by projecting to lower rank
        # Use SVD-like approach: take the first target_rank columns and scale
        if R >= target_rank:
            U_r = U[:, :target_rank].copy()
            V_r = V[:, :target_rank].copy()
            W_r = W[:, :target_rank].copy()
            # Add small random perturbation to escape local minima
            U_r += np.random.randn(n*m, target_rank) * 0.1
            V_r += np.random.randn(m*p, target_rank) * 0.1
            W_r += np.random.randn(p*n, target_rank) * 0.1
        else:
            U_r = np.random.randn(n*m, target_rank) * 0.5
            V_r = np.random.randn(m*p, target_rank) * 0.5
            W_r = np.random.randn(p*n, target_rank) * 0.5
        
        current_lr = lr
        for it in range(max_iter):
            if time.time() - t0 > time_limit:
                break
            
            loss, grad_U, grad_V, grad_W = _compute_loss_and_grad(U_r, V_r, W_r)
            
            # Adaptive learning rate
            if loss < best_loss * 0.9:
                current_lr = min(current_lr * 1.2, 0.1)
            elif loss > best_loss:
                current_lr = max(current_lr * 0.5, 0.001)
            
            U_r -= current_lr * grad_U
            V_r -= current_lr * grad_V
            W_r -= current_lr * grad_W
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
            
            if loss < 1e-12:
                break
        
        # Try a second phase with smaller learning rate if close
        if best_loss < 1e-6 and time.time() - t0 < time_limit * 0.7:
            for it in range(100):
                if time.time() - t0 > time_limit:
                    break
                loss, grad_U, grad_V, grad_W = _compute_loss_and_grad(best_U, best_V, best_W)
                best_U -= 0.001 * grad_U
                best_V -= 0.001 * grad_V
                best_W -= 0.001 * grad_W
                if loss < best_loss:
                    best_loss = loss
                if loss < 1e-12:
                    break
    
    if best_loss < 1e-9:
        return best_U, best_V, best_W
    return None

def _try_sparse_reduction(U, V, W, target_rank, time_limit=15.0):
    """Try to find a sparse decomposition of target rank by zeroing out entries."""
    t0 = time.time()
    n, m, p = N, M, P
    R = U.shape[1]
    
    # Try to project the existing decomposition to lower rank by keeping only
    # the most important columns based on their contribution
    if R < target_rank:
        return None
    
    # Compute the "importance" of each column by its norm
    U_norms = np.linalg.norm(U, axis=0)
    V_norms = np.linalg.norm(V, axis=0)
    W_norms = np.linalg.norm(W, axis=0)
    
    # Combined importance
    importance = U_norms * V_norms * W_norms
    top_indices = np.argsort(importance)[::-1][:target_rank]
    
    U_s = U[:, top_indices]
    V_s = V[:, top_indices]
    W_s = W[:, top_indices]
    
    # Check if this projection is already valid
    if _validate(U_s, V_s, W_s):
        return U_s, V_s, W_s
    
    # Try to optimize the projected decomposition
    result = _try_reduce_rank(U_s, V_s, W_s, target_rank, max_iter=100, lr=0.01, 
                              time_limit=time_limit - (time.time() - t0), n_restarts=3)
    return result

def construct():
    t0 = time.time()
    n, m, p = N, M, P
    
    # Start with the base rank-85 decomposition
    U, V, W = _build_base_85()
    
    if not _validate(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U, V, W
    best_rank = 85
    
    # Try to reduce rank from 84 downward, using both sparse and dense approaches
    for target_rank in range(84, 75, -1):
        if time.time() - t0 > DEADLINE_S - 25:
            break
        
        # First try sparse projection
        result = _try_sparse_reduction(U, V, W, target_rank, time_limit=10.0)
        if result is not None:
            rU, rV, rW = result
            if _validate(rU, rV, rW):
                best_U, best_V, best_W = rU, rV, rW
                best_rank = target_rank
                break
        
        # Then try dense optimization
        if time.time() - t0 > DEADLINE_S - 15:
            break
        result = _try_reduce_rank(U, V, W, target_rank, max_iter=150, lr=0.01, 
                                  time_limit=15.0, n_restarts=3)
        if result is not None:
            rU, rV, rW = result
            if _validate(rU, rV, rW):
                best_U, best_V, best_W = rU, rV, rW
                best_rank = target_rank
                break
    
    # Final polish: try to clean up the solution
    if time.time() - t0 < DEADLINE_S - 5:
        for i in range(3):
            loss, grad_U, grad_V, grad_W = _compute_loss_and_grad(best_U, best_V, best_W)
            if loss < 1e-12:
                break
            best_U -= 0.0001 * grad_U
            best_V -= 0.0001 * grad_V
            best_W -= 0.0001 * grad_W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
