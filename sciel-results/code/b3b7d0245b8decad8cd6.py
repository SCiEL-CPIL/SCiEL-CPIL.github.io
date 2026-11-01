import time
import random
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-12):
    """Check if U, V, W form a valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Reconstruct tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _try_rank_reduction(n, m, p, target_rank, max_iter=1000, deadline=None):
    """
    Try to find a decomposition of rank = target_rank using iterative local search.
    Uses the structure that the tensor is all-ones: T[i,j,k] = 1 for all i,j,k.
    So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    This is a system of n*m*p equations in rank*3*... variables (actually rank variables per matrix).
    We parameterize U, V, W as dense matrices of size (n*m x target_rank), etc.
    
    We use gradient descent on the residual.
    """
    if target_rank >= n * m * p:
        U, V, W = _schoolbook(n, m, p)
        # Pad/truncate to target_rank if needed
        if target_rank < U.shape[1]:
            U = U[:, :target_rank]
            V = V[:, :target_rank]
            W = W[:, :target_rank]
        elif target_rank > U.shape[1]:
            # This shouldn't happen if target_rank < n*m*p and we start from schoolbook
            pass
        return U, V, W, True
    
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    lr = 0.01
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        # Compute residual
        resid = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    resid[i,j,k] = 1.0 - np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
        
        # Check convergence
        if np.max(np.abs(resid)) < 1e-10:
            return U, V, W, True
        
        # Compute gradients
        # d/dU[i*m+j, c] = -resid[i,j] * V[j*p+k, c] * W[k*n+i, c] (sum over k)
        # Actually, for each (i,j,k), the term is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # d/dU[i*m+j, c] = sum_k resid[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
        # But we want to minimize sum resid^2, so gradient of 0.5*sum resid^2 w.r.t. U[i*m+j, c]
        # = sum_k resid[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
        
        # This is expensive to do naively. Let's do it in a smarter way.
        # For fixed c, U[:, c] is a vector of length n*m.
        # The residual at (i,j,k) depends on U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
        
        # Let's just do simple gradient updates.
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = resid[i,j,k]
                    for c in range(target_rank):
                        u_val = U[i*m+j, c]
                        v_val = V[j*p+k, c]
                        w_val = W[k*n+i, c]
                        U_grad[i*m+j, c] += r * v_val * w_val
                        V_grad[j*p+k, c] += r * u_val * w_val
                        W_grad[k*n+i, c] += r * u_val * v_val
        
        U -= lr * U_grad
        V -= lr * V_grad
        W -= lr * W_grad
        
        # Adaptive learning rate
        if it % 100 == 99:
            lr *= 0.9
        
    # Final check
    if _verify(U, V, W, n, m, p, tol=1e-8):
        return U, V, W, True
    return None, None, None, False

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s deadline, 20s margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    U0, V0, W0 = _schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    # Known bounds: R(2,3,10) is not known to be 60, but we try to find < 60
    # Start from just below schoolbook and go down
    # This is a long shot but we try a few target ranks
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U0, V0, W0
    
    # Try a few target ranks with limited iterations
    for target_rank in [59, 58, 57]:
        if time.time() > deadline:
            break
        max_iter = 200  # Limited iterations per target
        U, V, W, success = _try_rank_reduction(n, m, p, target_rank, max_iter=max_iter, deadline=deadline)
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break  # Found one, stop
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

DIMENSIONS = [2, 3, 10]
