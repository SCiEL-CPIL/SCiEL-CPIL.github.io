"""Strassen-based 2x5 x 5x6 matrix multiplication decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from itertools import product

    start_time = time.time()
    DEATHLINE = 130.0
    n, m, p = 2, 5, 6

    def elapsed():
        return time.time() - start_time

    def build_trivial():
        """Build the trivial rank-60 decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}, rank

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor.
        T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r] must equal 1 if u=i*m+j, v=j*p+k, w=k*n+i, else 0.
        """
        rank = len(U[0])
        # Check all required positions equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def tensor_rank_optimize(R, max_iter=2000, lr=0.01, seed=42):
        """
        Try to find a rank-R decomposition using gradient descent.
        Returns (U, V, W) as lists or None if not found.
        """
        if elapsed() > DEATHLINE - 5:
            return None

        rng = np.random.RandomState(seed)
        
        # Initialize with random factors
        U = rng.randn(n * m, R) * 0.1
        V = rng.randn(m * p, R) * 0.1
        W = rng.randn(p * n, R) * 0.1
        
        # Precompute the target tensor indices
        target_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_indices.append((i * m + j, j * p + k, k * n + i))
        
        best_loss = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(max_iter):
            if elapsed() > DEATHLINE - 5:
                break
            
            # Compute loss: sum over target positions of (sum_r U[u,r]*V[v,r]*W[w,r] - 1)^2
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for (u, v, w) in target_indices:
                # Compute contribution
                contrib = 0.0
                for r in range(R):
                    contrib += U[u, r] * V[v, r] * W[w, r]
                
                # Gradient of (contrib - 1)^2 with respect to each factor
                err = contrib - 1.0
                for r in range(R):
                    grad_U[u, r] += 2.0 * err * V[v, r] * W[w, r]
                    grad_V[v, r] += 2.0 * err * U[u, r] * W[w, r]
                    grad_W[w, r] += 2.0 * err * U[u, r] * V[v, r]
                
                loss += err * err
            
            loss /= len(target_indices)
            
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if loss < 1e-12:
                break
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Adaptive learning rate
            if it % 100 == 0 and it > 0:
                lr *= 0.95
        
        if best_loss < 1e-9:
            # Convert to lists
            U_list = best_U.tolist()
            V_list = best_V.tolist()
            W_list = best_W.tolist()
            
            # Round to nearest half-integer if possible
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        mat[i][j] = round(mat[i][j] * 2) / 2.0
            
            # Verify
            if verify_decomposition(U_list, V_list, W_list, tol=1e-8):
                return U_list, V_list, W_list
        
        return None

    # Start with trivial decomposition
    best_result, best_rank = build_trivial()
    
    if elapsed() > DEATHLINE - 5:
        return best_result
    
    # Try to find lower-rank decompositions
    # Search from rank 40 up to 59
    found_lower = False
    for R in range(40, 60):
        if elapsed() > DEATHLINE - 10:
            break
        
        result = tensor_rank_optimize(R, max_iter=1000, lr=0.005, seed=R)
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = R
            found_lower = True
            break
    
    # If we found a lower rank, try to go even lower
    if found_lower:
        for R in range(30, best_rank):
            if elapsed() > DEATHLINE - 10:
                break
            
            result = tensor_rank_optimize(R, max_iter=1000, lr=0.005, seed=R)
            if result is not None:
                U_list, V_list, W_list = result
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = R
                break
    
    return best_result
# EVOLVE-BLOCK-END
