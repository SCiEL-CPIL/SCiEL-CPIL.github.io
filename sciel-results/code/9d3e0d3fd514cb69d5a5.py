"""Matrix multiplication 4x5x5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Vectorized verification: for each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # Build index arrays
    for i in range(n):
        for j in range(m):
            u_row = U[i*m+j]  # shape (rank,)
            for k in range(p):
                val = np.dot(u_row, V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _gradient_descent_search(n, m, p, target_rank, deadline, max_iter=3000, lr=0.005):
    """
    Use gradient descent to find a rank-`target_rank` decomposition.
    Objective: minimize sum over (i,j,k) of (sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] - 1)^2
    """
    U = np.random.randn(n*m, target_rank) * 0.1
    V = np.random.randn(m*p, target_rank) * 0.1
    W = np.random.randn(p*n, target_rank) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 3:
            break
        
        # Compute loss and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j]
                for k in range(p):
                    v_row = V[j*p+k]
                    w_row = W[k*n+i]
                    
                    val = np.dot(u_row, v_row * w_row)
                    err = val - 1.0
                    loss += err * err
                    
                    # Gradients
                    grad_U[i*m+j] += 2 * err * (v_row * w_row)
                    grad_V[j*p+k] += 2 * err * (u_row * w_row)
                    grad_W[k*n+i] += 2 * err * (u_row * v_row)
        
        # Update with adaptive learning rate
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-10:
            break
    
    if _verify(best_U, best_V, best_W, n, m, p, tol=1e-8):
        return best_U, best_V, best_W
    return None

def _local_search_from_schoolbook(n, m, p, deadline):
    """
    Start from schoolbook decomposition and try to reduce rank by
    finding linear dependencies among the terms.
    """
    U, V, W = _build_schoolbook(n, m, p)
    current_rank = U.shape[1]
    
    # Try to eliminate terms one by one
    # For each term c, check if it can be expressed as a linear combination of others
    # This is a heuristic: try random subsets
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Try removing terms in random order
    term_indices = list(range(current_rank))
    np.random.shuffle(term_indices)
    
    for idx in term_indices:
        if time.time() > deadline - 5:
            break
        
        # Try removing term idx
        mask = np.ones(current_rank, dtype=bool)
        mask[idx] = False
        
        U_trial = U[:, mask]
        V_trial = V[:, mask]
        W_trial = W[:, mask]
        
        if _verify(U_trial, V_trial, W_trial, n, m, p, tol=1e-8):
            U, V, W = U_trial, V_trial, W_trial
            current_rank -= 1
            best_rank = current_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            # Reset term_indices for the new set
            term_indices = list(range(current_rank))
            np.random.shuffle(term_indices)
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    # 1. Schoolbook Decomposition (Rank 100)
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # 2. Local search from schoolbook to try to reduce rank
    if time.time() < deadline - 30:
        U_ls, V_ls, W_ls, rank_ls = _local_search_from_schoolbook(n, m, p, deadline)
        if rank_ls < best_rank:
            best_rank = rank_ls
            best_U, best_V, best_W = U_ls, V_ls, W_ls
    
    # 3. Attempt to find even lower rank using gradient-based search
    # Try ranks from best_rank-1 down to max(best_rank-15, 80)
    for target_rank in range(best_rank - 1, max(best_rank - 15, 79), -1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random restarts
        found = False
        for attempt in range(8):
            if time.time() > deadline - 15:
                break
            
            result = _gradient_descent_search(n, m, p, target_rank, deadline)
            if result is not None:
                U, V, W = result
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            break
    
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
