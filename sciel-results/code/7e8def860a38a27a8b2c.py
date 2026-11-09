import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _construct_tensor(n, m, p):
    # T[i, j, k] = 1 if it's part of the multiplication tensor
    # Actually, the tensor T is defined such that C = A*B is recovered.
    # T is a 3-tensor.
    # We want to approximate T with sum_r U[:, :, r] outer V[:, :, r] outer W[:, :, r]
    # T[i, j, k] is 1 if the term a_ij b_jk c_ik exists?
    # The multiplication tensor is sum_{i,j,k} a_ij b_jk c_ik
    # So T[i, j, k] = 1 for all i, j, k?
    # No, the tensor T is in R^{n*m} x R^{m*p} x R^{p*n}.
    # The entry T[u_idx, v_idx, w_idx] is 1 if u_idx corresponds to (i,j), v_idx to (j,k), w_idx to (k,i).
    # For a fixed i,j,k, there is exactly one such triple.
    # So T is a 0-1 tensor with 1s at (i*m+j, j*p+k, k*n+i).
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1
    return T

def _try_rank(n, m, p, rank, max_iters=2000, lr=0.01):
    T = _construct_tensor(n, m, p)
    # Initialize U, V, W randomly
    # Scale initialization to help convergence
    scale = 1.0 / np.sqrt(rank)
    U = np.random.randn(n * m, rank) * scale
    V = np.random.randn(m * p, rank) * scale
    W = np.random.randn(p * n, rank) * scale
    
    def error():
        # Compute decomposed tensor
        # D[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # Using einsum
        D = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.linalg.norm(D - T)
    
    err = error()
    if err < 1e-6:
        return U, V, W
        
    # Simple gradient descent
    # Gradient of error^2 w.r.t U[i, r] = 2 * sum_{j,k} (D[i,j,k] - T[i,j,k]) * V[j,r] * W[k,r]
    # We can compute D once per step.
    
    for _ in range(max_iters):
        D = np.einsum('ir,jr,kr->ijk', U, V, W)
        diff = D - T
        
        # Gradients
        # gU[i, r] = sum_{j, k} diff[i, j, k] * V[j, r] * W[k, r]
        # gV[j, r] = sum_{i, k} diff[i, j, k] * U[i, r] * W[k, r]
        # gW[k, r] = sum_{i, j} diff[i, j, k] * U[i, r] * V[j, r]
        
        # Efficient calculation using einsum
        # gU = einsum('ijk, jr, kr -> ir', diff, V, W)
        # gV = einsum('ijk, ir, kr -> jr', diff, U, W)
        # gW = einsum('ijk, ir, jr -> kr', diff, U, V)
        
        gU = np.einsum('ijk,jr,kr->ir', diff, V, W)
        gV = np.einsum('ijk,ir,kr->jr', diff, U, W)
        gW = np.einsum('ijk,ir,jr->kr', diff, U, V)
        
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        # Periodic check
        if _ % 100 == 0:
            err = np.linalg.norm(np.einsum('ir,jr,kr->ijk', U, V, W) - T)
            if err < 1e-6:
                return U, V, W
            # Adaptive learning rate? Keep simple.
            
    err = np.linalg.norm(np.einsum('ir,jr,kr->ijk', U, V, W) - T)
    if err < 1e-6:
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Try ranks in increasing order
    # The directive suggests 30.
    # Known lower bound?
    # Try 30, 40, 50, 55, 60
    ranks_to_try = [30, 40, 50, 55, 60]
    
    best_result = None
    best_rank = float('inf')
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
            res = _try_rank(n, m, p, rank, max_iters=1000, lr=0.05)
            if res is not None:
                best_result = res
                best_rank = rank
                found = True
                break # Found valid for this rank, try next lower rank? No, we want min rank.
                # Since we iterate 30, 40..., if we find 30, we can return immediately?
                # Yes, 30 is the target.
        
        if found:
            # Convert to lists for output
            U, V, W = best_result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # If no lower rank found, return schoolbook
    sb = _schoolbook_decomposition(n, m, p)
    return {
        "U": sb["U"].tolist(),
        "V": sb["V"].tolist(),
        "W": sb["W"].tolist()
    }
