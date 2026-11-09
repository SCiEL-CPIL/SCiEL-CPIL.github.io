import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The rank is the number of columns in these matrices.
    Uses gradient descent to find a low-rank decomposition, falling back to schoolbook.
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    best_rank = rank_schoolbook

    def compute_loss_and_grad(U, V, W):
        """Compute the loss and gradients for U, V, W."""
        r = U.shape[1]
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Precompute sums for efficiency?
        # For each (i, j, k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is O(n*m*p*r).
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    
                    val = 0.0
                    for rr in range(r):
                        val += U[idx_u, rr] * V[idx_v, rr] * W[idx_w, rr]
                    
                    error = val - 1.0
                    loss += error * error
                    
                    for rr in range(r):
                        # d/dU = error * V * W
                        grad_U[idx_u, rr] += error * V[idx_v, rr] * W[idx_w, rr]
                        # d/dV = error * U * W
                        grad_V[idx_v, rr] += error * U[idx_u, rr] * W[idx_w, rr]
                        # d/dW = error * U * V
                        grad_W[idx_w, rr] += error * U[idx_u, rr] * V[idx_v, rr]
        
        return loss, grad_U, grad_V, grad_W

    def try_rank(rank, n_restarts=5, n_steps=50):
        """Try to find a rank-`rank` decomposition using gradient descent."""
        if time.time() > deadline:
            return None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            lr = 0.01
            for step in range(n_steps):
                if time.time() > deadline:
                    return None
                
                loss, gU, gV, gW = compute_loss_and_grad(U, V, W)
                
                # Simple gradient step
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
                # Optional: adaptive learning rate or decay
                if step % 10 == 0:
                    lr *= 0.9
                
                if loss < 1e-10:
                    return (U, V, W)
        
        return None

    # Try ranks from 62 down to 50
    # Start from schoolbook and go down
    for rank in range(62, 49, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(rank, n_restarts=3, n_steps=30)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank
            break
    
    return best_result
