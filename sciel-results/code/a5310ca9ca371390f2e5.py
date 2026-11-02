"""Find rank-minimal tensor decomposition for 2x3 x 3x8 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 3, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = int(n), int(m), int(p)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n * m, W.shape[1]):
        return False
    if V.shape != (m * p, W.shape[1]):
        return False
    if W.shape != (p * n, W.shape[1]):
        return False
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_schoolbook():
    """Build the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank_r(r, n, m, p, max_iter=500, seed=42):
    """Try to find a rank-r decomposition using random initialization and gradient descent."""
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, r) * 0.5
    V = np.random.randn(m * p, r) * 0.5
    W = np.random.randn(p * n, r) * 0.5
    
    # Define the loss: for each (i,j,k), (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    # Plus we need to handle that the decomposition should be exact
    
    # Precompute index mappings
    idx_map = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_map[(i, j, k)] = (i * m + j, j * p + k, k * n + i)
    
    def compute_loss_and_grads(U, V, W):
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for (i, j, k), (u_idx, v_idx, w_idx) in idx_map.items():
            # s = sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r]
            s = 0.0
            for r in range(r):
                s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            diff = s - 1.0
            loss += diff * diff
            
            # Gradients
            for r in range(r):
                # d_loss/d_U[u_idx,r] = 2*diff * V[v_idx,r] * W[w_idx,r]
                grad_U[u_idx, r] += 2 * diff * V[v_idx, r] * W[w_idx, r]
                # d_loss/d_V[v_idx,r] = 2*diff * U[u_idx,r] * W[w_idx,r]
                grad_V[v_idx, r] += 2 * diff * U[u_idx, r] * W[w_idx, r]
                # d_loss/d_W[w_idx,r] = 2*diff * U[u_idx,r] * V[v_idx,r]
                grad_W[w_idx, r] += 2 * diff * U[u_idx, r] * V[v_idx, r]
        
        return loss, grad_U, grad_V, grad_W
    
    # Simple gradient descent with adaptive learning rate
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        loss, gU, gV, gW = compute_loss_and_grads(U, V, W)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-12:
            # Found a solution
            return best_U.tolist(), best_V.tolist(), best_W.tolist(), True
        
        # Adaptive learning rate
        norm_g = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
        if norm_g > 1e-10:
            lr = min(0.1, 0.01 / norm_g)
        
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
    
    # Check if we found a good solution
    if best_loss < 1e-9:
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), True
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140 second deadline with safety margin
    
    # Start with schoolbook as fallback
    schoolbook = _build_schoolbook()
    
    # Try ranks from low to high, but be strategic
    # Known lower bound: max(n*m, m*p, p*n) = 24
    # Try a range of ranks
    
    # First, try some promising ranks
    ranks_to_try = []
    
    # Try ranks near the schoolbook rank first to see if we can reduce slightly
    for r in range(48, 23, -1):
        ranks_to_try.append(r)
    
    # Also try some lower ranks
    for r in range(24, 36):
        if r not in ranks_to_try:
            ranks_to_try.append(r)
    
    # Sort and deduplicate
    ranks_to_try = sorted(set(ranks_to_try))
    
    best_result = schoolbook
    best_rank = 48
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try multiple seeds
        found = False
        for seed in range(10):
            if time.time() > deadline:
                break
            
            U_list, V_list, W_list, success = _try_rank_r(r, n, m, p, max_iter=300, seed=seed)
            
            if success:
                # Verify
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
        
        if found:
            # If we found a solution at rank r, we can stop trying higher ranks
            # but we should try lower ranks
            pass
    
    return best_result
