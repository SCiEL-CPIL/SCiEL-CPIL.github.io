"""
Strategy: Riemannian Gradient Descent for Tensor Rank Minimization.
We initialize the factor matrices with a known valid decomposition (e.g., Rank 64 schoolbook 
or Rank 49 Strassen). We then perform gradient descent on the Stiefel manifold (or simply 
Euclidean space with normalization) to minimize the reconstruction error of the matrix 
multiplication tensor. To encourage rank reduction, we add an L1 sparsity penalty to the 
loss function. This drives some columns of the factor matrices towards zero. After 
optimization, we identify and drop columns that are sufficiently close to zero, 
resulting in a lower-rank decomposition. We verify the result exactly.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _build_schoolbook_4x4():
    """
    Returns the rank-64 schoolbook decomposition for 4x4 matrix multiplication.
    This serves as a robust initial guess for the optimizer.
    """
    n, m, p = 4, 4, 4
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
                
    return U, V, W

def _build_strassen_2x2():
    """
    Returns the Strassen decomposition for 2x2 matrix multiplication.
    """
    a = np.zeros((7, 2, 2))
    b = np.zeros((7, 2, 2))
    c = np.zeros((2, 2, 7))
    
    a[0] = [[1, 0], [0, 1]]
    b[0] = [[1, 0], [0, 1]]
    a[1] = [[0, 0], [1, 1]]
    b[1] = [[1, 0], [0, 0]]
    a[2] = [[1, 0], [0, 0]]
    b[2] = [[0, 0], [1, -1]]
    a[3] = [[0, 0], [0, 1]]
    b[3] = [[-1, 1], [0, 0]]
    a[4] = [[1, 1], [0, 0]]
    b[4] = [[0, 0], [0, 1]]
    a[5] = [[-1, 0], [1, 0]]
    b[5] = [[1, 1], [0, 0]]
    a[6] = [[0, 1], [0, -1]]
    b[6] = [[0, 0], [1, 1]]
    
    c[0, 0] = [1, 0, 0, 1, -1, 0, 1]
    c[0, 1] = [0, 0, 1, 0, 1, 0, 0]
    c[1, 0] = [0, 1, 0, 1, 0, 0, 0]
    c[1, 1] = [1, -1, 1, 0, 0, 1, 0]
    
    return a, b, c

def _build_strassen_4x4():
    """
    Builds the rank-49 decomposition for 4x4 matrix multiplication
    by taking the tensor product of the rank-7 Strassen decomposition for 2x2.
    """
    a2, b2, c2 = _strassen_2x2()
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for s1 in range(7):
        for s2 in range(7):
            col = 7 * s1 + s2
            
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            U[i * m + j, col] = a2[s1, i1, j1] * a2[s2, i2, j2]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            V[j * p + k, col] = b2[s1, j1, k1] * b2[s2, j2, k2]
            
            for i1 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            i = 2 * i1 + i2
                            k = 2 * k1 + k2
                            W[k * n + i, col] = c2[i1, k1, s1] * c2[i2, k2, s2]
    
    return U, V, W

def _compute_tensor(U, V, W):
    """
    Computes the tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    """
    n, m, p = 4, 4, 4
    rank = U.shape[1]
    U4 = U.reshape(n, m, rank)
    V4 = V.reshape(m, p, rank)
    W4 = W.reshape(p, n, rank)
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U4[i, j, :], V4[j, k, :] * W4[k, i, :])
    return T

def _loss_and_grad(U, V, W, target, l1_lambda):
    """
    Computes the loss (MSE + L1 sparsity) and gradients.
    Loss = ||T - Target||_F^2 + l1_lambda * (||U||_1 + ||V||_1 + ||W||_1)
    """
    n, m, p = 4, 4, 4
    T = _compute_tensor(U, V, W)
    err = T - target
    mse_loss = np.sum(err * err)
    
    l1_loss = l1_lambda * (np.sum(np.abs(U)) + np.sum(np.abs(V)) + np.sum(np.abs(W)))
    
    # Gradient of MSE w.r.t. U
    # d/dU[i,j,r] (sum_{i,j,k} (sum_r' U[i,j,r'] V[j,k,r'] W[k,i,r'] - T[i,j,k])^2)
    # = 2 * (T[i,j,r] - Target[i,j,r]) * V[j,k,r] * W[k,i,r] summed over k? No.
    # Let S[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # d/dU[i,j,r] S[i,j,k] = V[j,k,r] W[k,i,r]
    # d/dU[i,j,r] L = 2 * sum_k (S[i,j,k] - T[i,j,k]) * V[j,k,r] * W[k,i,r]
    
    U4 = U.reshape(n, m, -1)
    V4 = V.reshape(m, p, -1)
    W4 = W.reshape(p, n, -1)
    
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                diff = S[i, j, k] - target[i, j, k]
                # d/dU[i,j,r]
                grad_U[i * m + j, :] += 2 * diff * V4[j, k, :] * W4[k, i, :]
                # d/dV[j,k,r]
                grad_V[j * p + k, :] += 2 * diff * U4[i, j, :] * W4[k, i, :]
                # d/dW[k,i,r]
                grad_W[k * n + i, :] += 2 * diff * U4[i, j, :] * V4[j, k, :]
    
    # Gradient of L1
    grad_U += l1_lambda * np.sign(U)
    grad_V += l1_lambda * np.sign(V)
    grad_W += l1_lambda * np.sign(W)
    
    return mse_loss + l1_loss, grad_U, grad_V, grad_W

def _optimize(U, V, W, target, time_limit, l1_lambda=0.01, lr=0.01, steps=1000):
    """
    Performs gradient descent to minimize loss.
    """
    start_time = time.time()
    
    for step in range(steps):
        if time.time() - start_time > time_limit:
            break
            
        loss, gU, gV, gW = _loss_and_grad(U, V, W, target, l1_lambda)
        
        # Simple gradient descent
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        # Optional: Normalize columns to prevent divergence
        # This is not strictly necessary if lr is small, but helps stability
        # for r in range(U.shape[1]):
        #     norm_U = np.linalg.norm(U[:, r])
        #     norm_V = np.linalg.norm(V[:, r])
        #     norm_W = np.linalg.norm(W[:, r])
        #     if norm_U > 0 and norm_V > 0 and norm_W > 0:
        #         scale = 1.0 / (norm_U * norm_V * norm_W) ** (1.0/3.0)
        #         U[:, r] *= scale
        #         V[:, r] *= scale
        #         W[:, r] *= scale

    return U, V, W

def _prune(U, V, W, target, threshold=0.01):
    """
    Removes columns with small norms and verifies if the remaining decomposition is valid.
    """
    n, m, p = 4, 4, 4
    rank = U.shape[1]
    
    # Calculate norms for each column
    norms = np.zeros(rank)
    for r in range(rank):
        norms[r] = np.linalg.norm(U[:, r]) + np.linalg.norm(V[:, r]) + np.linalg.norm(W[:, r])
    
    # Sort indices by norm
    indices = np.argsort(norms)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try removing the smallest columns one by one
    # We need to check if the tensor reconstruction is still valid
    # Since we are pruning, we just drop the column.
    # The decomposition T = sum U_r V_r W_r remains valid if we just sum fewer terms?
    # No, dropping a term changes the tensor.
    # The optimization was designed to make the tensor contribution of small columns close to 0.
    # So if ||U_r|| is small, then T_r is small.
    # If we drop it, the error increases by ||T_r||.
    # We need to verify the final result exactly.
    
    keep_mask = np.ones(rank, dtype=bool)
    
    # Try to prune from smallest to largest
    # We can't just greedily prune without re-verifying because dropping one might affect the validity of others?
    # Actually, the tensor is a sum. If T = sum_{r in Keep} T_r + sum_{r in Drop} T_r.
    # If sum_{r in Drop} T_r is close to 0, then sum_{r in Keep} T_r is close to T.
    # But we need EXACT decomposition.
    # The optimization minimizes ||T_calc - T_target||^2.
    # If we prune, T_calc_new = T_calc_old - sum_{r in Drop} T_r.
    # This will introduce error unless sum_{r in Drop} T_r is exactly 0.
    # Gradient descent with L1 tends to make entries exactly 0 (or very close).
    # So we check if the pruned decomposition is valid.
    
    # Let's try pruning columns where norm < threshold
    # And verify.
    
    pruned_indices = [i for i in indices if norms[i] < threshold]
    
    if not pruned_indices:
        return U, V, W, rank
    
    # Construct new matrices without pruned indices
    keep_indices = [i for i in range(rank) if i not in pruned_indices]
    
    if not keep_indices:
        return U, V, W, rank
        
    U_new = U[:, keep_indices]
    V_new = V[:, keep_indices]
    W_new = W[:, keep_indices]
    
    # Verify
    T_new = _compute_tensor(U_new, V_new, W_new)
    err = np.max(np.abs(T_new - target))
    
    if err < 1e-9:
        return U_new, V_new, W_new, len(keep_indices)
    else:
        # If pruning fails, return original
        return U, V, W, rank

def _verify(U, V, W):
    """Verify that the decomposition exactly computes 4x4 matrix multiplication."""
    n, m, p = 4, 4, 4
    target = np.ones((n, m, p))
    T = _compute_tensor(U, V, W)
    max_err = np.max(np.abs(T - target))
    return max_err < 1e-9

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    n, m, p = 4, 4, 4
    target = np.ones((n, m, p))
    
    # Strategy:
    # 1. Start with Rank 49 Strassen (known good).
    # 2. Optimize with L1 sparsity to see if we can push it to Rank 48 or lower.
    # 3. Prune small columns.
    # 4. If verification fails, fallback to Rank 49 or 64.
    
    # Initialize with Strassen 49
    U, V, W = _build_strassen_4x4()
    
    # Optimization Phase
    # We use a short optimization time to ensure we stay within limits.
    # L1 lambda is small to avoid destroying the solution, just to encourage sparsity.
    # We try multiple restarts or different lambdas if time permits.
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 49
    
    # Try optimizing
    try:
        # Use a small lambda to gently encourage sparsity
        l1_lambda = 0.005
        lr = 0.005
        steps = 2000
        
        U_opt, V_opt, W_opt = _optimize(U, V, W, target, time_limit=60, l1_lambda=l1_lambda, lr=lr, steps=steps)
        
        # Prune
        U_pruned, V_pruned, W_pruned, rank_pruned = _prune(U_opt, V_opt, W_opt, target, threshold=0.05)
        
        if _verify(U_pruned, V_pruned, W_pruned):
            if rank_pruned < best_rank:
                best_U, best_V, best_W = U_pruned, V_pruned, W_pruned
                best_rank = rank_pruned
    except Exception as e:
        pass
    
    # If we didn't improve, try a different initialization or parameters?
    # For now, if Strassen 49 is the best, we return it.
    # But wait, the task asks for rank-minimal.
    # Is 49 minimal?
    # The rank of 4x4 matrix multiplication is known to be between 49 and 64?
    # Actually, the rank of 2x2 is 7. Rank of 4x4 is at most 49.
    # Is it less than 49?
    # It is known that the rank of 4x4 matrix multiplication is exactly 49?
    # No, it's not proven to be 49. It's known to be <= 49.
    # However, it is widely believed to be 49 or close to it.
    # If we can't find a lower rank, we return 49.
    
    # Ensure the returned solution is valid.
    if not _verify(best_U, best_V, best_W):
        # Fallback to schoolbook 64
        U_fb, V_fb, W_fb = _build_schoolbook_4x4()
        if _verify(U_fb, V_fb, W_fb):
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        else:
            # This should never happen
            raise RuntimeError("No valid decomposition found")
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
