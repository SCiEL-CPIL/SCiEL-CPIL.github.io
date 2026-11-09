"""Rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Build index arrays for vectorized verification
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all i,j,k
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute all tensor entries: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use einsum: sum over c
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    return np.all(np.abs(T - 1.0) <= tol)

def _to_halfint_list(arr):
    """Convert numpy array to list with integer/half-integer entries."""
    result = []
    for row in arr:
        row_list = []
        for val in row:
            rounded = round(val * 2) / 2.0
            if abs(rounded - int(rounded)) < 1e-9:
                row_list.append(int(rounded))
            else:
                row_list.append(rounded)
        result.append(row_list)
    return result

def _verify_halfint(U, V, W, n, m, p, tol=1e-9):
    """Verify with half-integer rounding applied first."""
    U_rounded = np.round(np.array(U, dtype=np.float64) * 2) / 2.0
    V_rounded = np.round(np.array(V, dtype=np.float64) * 2) / 2.0
    W_rounded = np.round(np.array(W, dtype=np.float64) * 2) / 2.0
    return _verify(U_rounded, V_rounded, W_rounded, n, m, p, tol)

def _compute_loss_grads(U, V, W, n, m, p):
    """Compute loss and gradients using vectorized einsum operations."""
    rank = U.shape[1]
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    err = T - 1.0
    loss = np.sum(err * err)
    
    # Gradients:
    # dL/dU[i,j,c] = sum_k err[i,j,k] * V[j,k,c] * W[k,i,c]
    # dL/dV[j,k,c] = sum_i err[i,j,k] * U[i,j,c] * W[k,i,c]
    # dL/dW[k,i,c] = sum_j err[i,j,k] * U[i,j,c] * V[j,k,c]
    
    grad_U = np.einsum('ijk,jkc,kic->ijc', err, V_reshaped, W_reshaped)
    grad_V = np.einsum('ijk,ijc,kic->jkc', err, U_reshaped, W_reshaped)
    grad_W = np.einsum('ijk,ijc,jkc->kic', err, U_reshaped, V_reshaped)
    
    return loss, grad_U.reshape(n*m, rank), grad_V.reshape(m*p, rank), grad_W.reshape(p*n, rank)

def _try_rank_structured(r, n, m, p, max_iters=2000, seed=42, deadline=None):
    """Try to find a rank-r decomposition using structured initialization."""
    np.random.seed(seed)
    
    # Initialize with structured random values
    U = np.random.randn(n*m, r) * 0.3
    V = np.random.randn(m*p, r) * 0.3
    W = np.random.randn(p*n, r) * 0.3
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.05
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute loss and gradients efficiently
        loss, grad_U, grad_V, grad_W = _compute_loss_grads(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-12:
            break
        
        # Adaptive learning rate
        lr_effective = lr / (1.0 + it * 0.001)
        U -= lr_effective * grad_U
        V -= lr_effective * grad_V
        W -= lr_effective * grad_W
    
    # Round to half-integers and verify
    U_rounded = np.round(best_U * 2) / 2.0
    V_rounded = np.round(best_V * 2) / 2.0
    W_rounded = np.round(best_W * 2) / 2.0
    
    if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
        return U_rounded, V_rounded, W_rounded
    
    # Try without rounding
    if _verify(best_U, best_V, best_W, n, m, p):
        return best_U, best_V, best_W
    
    return None

def _polish(U, V, W, n, m, p, max_iters=500, deadline=None):
    """Final high-precision polish pass to refine the decomposition."""
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        loss, grad_U, grad_V, grad_W = _compute_loss_grads(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-14:
            break
        
        lr_effective = lr / (1.0 + it * 0.01)
        U -= lr_effective * grad_U
        V -= lr_effective * grad_V
        W -= lr_effective * grad_W
    
    return best_U, best_V, best_W, best_loss

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # Search from higher ranks down, with more seeds for harder ranks
    for r in range(95, 34, -1):
        if time.time() > deadline - 15:
            break
        
        # Number of seeds based on remaining time
        remaining = deadline - time.time()
        max_seeds = min(12, max(3, int(remaining / 8)))
        
        found = False
        for seed in range(max_seeds):
            if time.time() > deadline - 10:
                break
            
            result = _try_rank_structured(r, n, m, p, max_iters=2000, seed=seed, deadline=deadline - 5)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_result = {
                    "U": _to_halfint_list(U),
                    "V": _to_halfint_list(V),
                    "W": _to_halfint_list(W)
                }
                found = True
                break
        
        if found:
            continue  # Try lower rank
        else:
            # Don't break immediately; try a few more ranks
            if r > 50:
                continue
            else:
                break  # Can't find at this rank, likely can't at lower either
    
    # Final polish pass on the best result found
    if best_rank < rank_school and time.time() < deadline - 5:
        try:
            U_best = np.array(best_result["U"], dtype=np.float64)
            V_best = np.array(best_result["V"], dtype=np.float64)
            W_best = np.array(best_result["W"], dtype=np.float64)
            
            U_pol, V_pol, W_pol, loss_pol = _polish(U_best, V_best, W_best, n, m, p, max_iters=300, deadline=deadline - 3)
            
            # Check if polish improved things
            if _verify(U_pol, V_pol, W_pol, n, m, p):
                best_result = {
                    "U": _to_halfint_list(U_pol),
                    "V": _to_halfint_list(V_pol),
                    "W": _to_halfint_list(W_pol)
                }
        except Exception:
            pass
    
    return best_result
