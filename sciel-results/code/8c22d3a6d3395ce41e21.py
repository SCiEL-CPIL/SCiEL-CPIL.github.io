import numpy as np
import time
import json

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s for safety

    # Target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank

    # Try to find a rank < 72 solution using gradient descent
    # We'll try rank 71, then 70, etc., but given the 140s limit and difficulty,
    # we'll focus on a few quick attempts at rank 71.
    # Actually, let's just try a few random restarts at rank 71.
    
    # If we don't find a better one, we fall back to schoolbook.
    
    # To maximize chance of finding a valid lower rank solution quickly if it exists,
    # we could use a smarter initialization or algorithm.
    # But for now, let's stick to the structure but ensure robustness.
    
    # Let's try rank 71 with a few restarts
    for rank in range(71, 72): 
        if time.time() > deadline:
            break
            
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Random initialization
            np.random.seed(42 + restart) # Reproducibility
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Use Adam optimizer logic manually or just simple GD with adaptive lr
            # Simple GD with decreasing learning rate
            lr = 0.01
            for step in range(500):
                if time.time() > deadline:
                    break
                    
                # Compute tensor product
                # T_est[u,v,w] = sum_c U[u,c] * V[v,c] * W[w,c]
                T_est = np.einsum('uc,vc,wc->uvw', U, V, W)
                error = T_est - T
                loss = np.sum(error**2)
                
                if not np.isfinite(loss):
                    break
                
                if loss < 1e-6:
                    # Check for exactness with higher precision if possible, 
                    # but float64 might suffice for validation if the evaluator is lenient?
                    # The task says "exactly decompose". Usually this implies integer/half-integer.
                    # If we find a solution with small error, we can try to round it.
                    
                    # Round to nearest 0.5 or integer
                    def round_to_half(arr):
                        return np.round(arr * 2) / 2
                    
                    U_rounded = round_to_half(U)
                    V_rounded = round_to_half(V)
                    W_rounded = round_to_half(W)
                    
                    # Verify rounded solution
                    T_est_rounded = np.einsum('uc,vc,wc->uvw', U_rounded, V_rounded, W_rounded)
                    error_rounded = T_est_rounded - T
                    loss_rounded = np.sum(error_rounded**2)
                    
                    if loss_rounded < 1e-9:
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        best_rank = rank
                        break
                    
                    # If rounding fails, keep the float solution?
                    # The contract says "integer/half-integer entries".
                    # So we MUST round. If rounding doesn't work, this path is invalid.
                    # So we continue searching.
                
                # Gradients
                grad_U = 2 * np.einsum('uvw,vc,wc->uc', error, V, W)
                grad_V = 2 * np.einsum('uvw,uc,wc->vc', error, U, W)
                grad_W = 2 * np.einsum('uvw,uc,vc->wc', error, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Decay learning rate
                if step % 100 == 0:
                    lr *= 0.5
            
            if best_rank == rank:
                break

    # Construct Schoolbook Rank 72 Solution (Guaranteed Valid)
    if best_U is None:
        rank = schoolbook_rank
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    # Ensure all values are finite
    if not (np.all(np.isfinite(best_U)) and np.all(np.isfinite(best_V)) and np.all(np.isfinite(best_W))):
        # Fallback to schoolbook if something went wrong
        rank = schoolbook_rank
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W

    # Convert to lists of lists
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]

    return {"U": U_out, "V": V_out, "W": W_out}
