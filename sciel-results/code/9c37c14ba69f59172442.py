import time
import numpy as np
from itertools import product

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _build_target_tensor(n, m, p):
    """Build the matmul tensor as a list of (a, b, c) indices where T=1."""
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    return triples

def _build_non_target_set(n, m, p):
    """Build set of all non-target (a,b,c) indices that must be 0."""
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_set.add((a, b, c))
    
    non_target = []
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in target_set:
                    non_target.append((a, b, c))
    return non_target

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U,V,W decompose the matmul tensor exactly."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check all target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that non-target entries are 0
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_set.add((a, b, c))
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in target_set:
                    val = np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
                    if abs(val) > tol:
                        return False
    return True

def _construct_rank_18(n, m, p):
    """
    Constructive rank-18 decomposition for 2x3x6 matmul.
    For each (j,k) pair, create one term handling both i=0 and i=1.
    """
    rank = 18
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    t = 0
    for j in range(m):
        for k in range(p):
            a0 = 0 * m + j
            a1 = 1 * m + j
            b = j * p + k
            c0 = k * n + 0
            c1 = k * n + 1
            
            U[a0, t] = 1
            U[a1, t] = 1
            V[b, t] = 1
            W[c0, t] = 1
            W[c1, t] = 1
            t += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_rank_36(n, m, p):
    """Schoolbook rank-36 decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _snap_to_half_integers(arr, tol=0.01):
    """Snap values to nearest half-integer if close enough."""
    result = np.copy(arr)
    for idx in np.ndindex(result.shape):
        val = result[idx]
        nearest_half = round(val * 2) / 2.0
        if abs(val - nearest_half) < tol:
            result[idx] = nearest_half
    return result

def _compute_loss_and_grads(U, V, W, triples, non_triples, lambda_nt=1.0):
    """Compute combined loss and gradients for target and non-target entries."""
    n, m, p = U.shape[0] // 3, 3, W.shape[0] // 2
    r = U.shape[1]
    
    loss = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    # Target entries: should be 1
    for (a, b, c) in triples:
        current = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
        error = current - 1.0
        loss += error * error
        
        for rr in range(r):
            dU = 2 * error * V[b, rr] * W[c, rr]
            dV = 2 * error * U[a, rr] * W[c, rr]
            dW = 2 * error * U[a, rr] * V[b, rr]
            grad_U[a, rr] += dU
            grad_V[b, rr] += dV
            grad_W[c, rr] += dW
    
    # Non-target entries: should be 0
    for (a, b, c) in non_triples:
        current = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
        loss += lambda_nt * current * current
        
        for rr in range(r):
            dU = 2 * lambda_nt * current * V[b, rr] * W[c, rr]
            dV = 2 * lambda_nt * current * U[a, rr] * W[c, rr]
            dW = 2 * lambda_nt * current * U[a, rr] * V[b, rr]
            grad_U[a, rr] += dU
            grad_V[b, rr] += dV
            grad_W[c, rr] += dW
    
    return loss, grad_U, grad_V, grad_W

def _try_rank_with_optimization(n, m, p, r, max_iters=1000, lr=0.01, seed=42):
    """Try to find rank-r decomposition using optimization with both target and non-target constraints."""
    if _time_left() < 15:
        return None
    
    np.random.seed(seed)
    U = np.random.randn(n * m, r) * 0.1
    V = np.random.randn(m * p, r) * 0.1
    W = np.random.randn(p * n, r) * 0.1
    
    triples = _build_target_tensor(n, m, p)
    non_triples = _build_non_target_set(n, m, p)
    
    lambda_nt = 1.0  # Weight for non-target penalty
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if _time_left() < 15:
            break
        
        loss, grad_U, grad_V, grad_W = _compute_loss_and_grads(U, V, W, triples, non_triples, lambda_nt)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Adaptive learning rate with decay
        grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
        if grad_norm > 1e-12:
            step = lr / (1.0 + it * 0.005)
            U -= step * grad_U
            V -= step * grad_V
            W -= step * grad_W
        
        if best_loss < 1e-10:
            break
    
    # Try to snap to half-integers
    U_s = _snap_to_half_integers(best_U)
    V_s = _snap_to_half_integers(best_V)
    W_s = _snap_to_half_integers(best_W)
    
    if _verify(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p, tol=1e-6):
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-6):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def _polish(U, V, W, n, m, p, max_iters=200, lr=0.005):
    """High-precision polish pass for a near-converged decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    triples = _build_target_tensor(n, m, p)
    non_triples = _build_non_target_set(n, m, p)
    
    for it in range(max_iters):
        if _time_left() < 10:
            break
        
        loss, grad_U, grad_V, grad_W = _compute_loss_and_grads(U, V, W, triples, non_triples, 1.0)
        
        if loss < 1e-12:
            break
        
        grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
        if grad_norm > 1e-14:
            step = lr / (1.0 + it * 0.01)
            U -= step * grad_U
            V -= step * grad_V
            W -= step * grad_W
    
    U_s = _snap_to_half_integers(U)
    V_s = _snap_to_half_integers(V)
    W_s = _snap_to_half_integers(W)
    
    if _verify(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p, tol=1e-6):
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_structured_rank_12(n, m, p):
    """
    Try rank-12 decomposition using a structured initialization.
    For each k (0..5), use 2 terms to handle the 2x3 subproblem.
    """
    if _time_left() < 15:
        return None
    
    r = 12
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # For each k, use 2 terms
    for k in range(p):
        for term in range(2):
            r_idx = k * 2 + term
            if term == 0:
                # Handle j=0 for both i
                U[0 * m + 0, r_idx] = 1
                U[1 * m + 0, r_idx] = 1
                V[0 * p + k, r_idx] = 1
                W[k * n + 0, r_idx] = 1
                W[k * n + 1, r_idx] = 1
            else:
                # Handle j=1,2 for both i - use weighted combination
                U[0 * m + 1, r_idx] = 1
                U[1 * m + 1, r_idx] = 1
                U[0 * m + 2, r_idx] = 1
                U[1 * m + 2, r_idx] = 1
                V[1 * p + k, r_idx] = 1
                V[2 * p + k, r_idx] = 1
                W[k * n + 0, r_idx] = 0.5
                W[k * n + 1, r_idx] = 0.5
    
    # Now polish
    result = _polish(U.tolist(), V.tolist(), W.tolist(), n, m, p, max_iters=500, lr=0.01)
    if result is not None:
        return result
    
    # Try with random perturbation
    for seed in [1, 2, 3]:
        if _time_left() < 10:
            break
        np.random.seed(seed)
        U_pert = U + np.random.randn(n * m, r) * 0.05
        V_pert = V + np.random.randn(m * p, r) * 0.05
        W_pert = W + np.random.randn(p * n, r) * 0.05
        result = _polish(U_pert.tolist(), V_pert.tolist(), W_pert.tolist(), n, m, p, max_iters=300, lr=0.01)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = 2, 3, 6
    
    # First, try the constructive rank-18 approach
    result_18 = _construct_rank_18(n, m, p)
    if _verify(result_18["U"], result_18["V"], result_18["W"], n, m, p, tol=1e-6):
        return result_18
    
    # Try optimization for rank 18 with multiple seeds
    for seed in [42, 123, 456, 789, 1011]:
        if _time_left() < 30:
            break
        result = _try_rank_with_optimization(n, m, p, 18, max_iters=500, lr=0.02, seed=seed)
        if result is not None:
            return result
    
    # Try structured rank-12
    if _time_left() > 40:
        result_12 = _try_structured_rank_12(n, m, p)
        if result_12 is not None:
            return result_12
    
    # Try optimization for rank 12
    for seed in [42, 123, 456]:
        if _time_left() < 20:
            break
        result = _try_rank_with_optimization(n, m, p, 12, max_iters=800, lr=0.01, seed=seed)
        if result is not None:
            return result
    
    # Fallback to schoolbook rank 36
    return _construct_rank_36(n, m, p)
