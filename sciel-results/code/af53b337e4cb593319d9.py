import numpy as np
import time

def construct():
    n, m, p = 3, 5, 7
    deadline = time.time() + 110  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n*m, rank_schoolbook))
    V_schoolbook = np.zeros((m*p, rank_schoolbook))
    W_schoolbook = np.zeros((p*n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i*m+j, col] = 1
                V_schoolbook[j*p+k, col] = 1
                W_schoolbook[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_schoolbook, V_schoolbook, W_schoolbook
    
    # Try to find a lower-rank decomposition using gradient descent
    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    
    # Precompute the target tensor as a list of (idx_u, idx_v, idx_w) tuples
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i*m+j, j*p+k, k*n+i))
    
    # Try ranks from 104 down to 50
    for rank in range(rank_schoolbook - 1, 49, -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Initialize U, V, W randomly
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Use gradient descent to minimize the error
            lr = 0.01
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Compute the current decomposition
                error = 0
                for idx_u, idx_v, idx_w in target_indices:
                    val = sum(U[idx_u, r] * V[idx_v, r] * W[idx_w, r] for r in range(rank))
                    error += (val - 1) ** 2
                
                if error < 1e-6:
                    break
                
                # Compute gradients (simplified)
                # This is computationally expensive, so I'll use a simple update
                # For each rank-1 term, update U, V, W
                for r in range(rank):
                    grad_U = np.zeros(n*m)
                    grad_V = np.zeros(m*p)
                    grad_W = np.zeros(p*n)
                    
                    for idx_u, idx_v, idx_w in target_indices:
                        val = sum(U[idx_u, rr] * V[idx_v, rr] * W[idx_w, rr] for rr in range(rank))
                        diff = val - 1
                        grad_U[idx_u] += diff * V[idx_v, r] * W[idx_w, r]
                        grad_V[idx_v] += diff * U[idx_u, r] * W[idx_w, r]
                        grad_W[idx_w] += diff * U[idx_u, r] * V[idx_v, r]
                    
                    U[:, r] -= lr * grad_U
                    V[:, r] -= lr * grad_V
                    W[:, r] -= lr * grad_W
            
            # Check if the decomposition is valid
            valid = True
            for idx_u, idx_v, idx_w in target_indices:
                val = sum(U[idx_u, r] * V[idx_v, r] * W[idx_w, r] for r in range(rank))
                if abs(val - 1) > 1e-4:
                    valid = False
                    break
            
            if valid:
                # Round to nearest half-integer
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Verify again after rounding
                valid = True
                for idx_u, idx_v, idx_w in target_indices:
                    val = sum(U[idx_u, r] * V[idx_v, r] * W[idx_w, r] for r in range(rank))
                    if abs(val - 1) > 1e-6:
                        valid = False
                        break
                
                if valid:
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    break
        
        if best_rank < rank_schoolbook:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
