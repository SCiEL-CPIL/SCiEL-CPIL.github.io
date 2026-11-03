"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomp(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # Reshape to 3D tensors for easy einsum
    # U[i*m+j, r] -> U_3d[i, j, r]
    U_3d = U.reshape(n, m, rank)
    # V[j*p+k, r] -> V_3d[j, k, r]
    V_3d = V.reshape(m, p, rank)
    # W[k*n+i, r] -> W_3d[k, i, r]
    W_3d = W.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Using einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # Check if T is all ones
    if not np.allclose(T, 1.0, atol=1e-6):
        return False
    return True

def _compute_loss_grad(U, V, W):
    """Compute loss and gradients for U, V, W using vectorized operations."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # T[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # Loss = sum (T - 1)^2
    error = T - 1.0
    loss = np.sum(error ** 2)
    
    # Gradients
    # dL/dT[i,j,k] = 2 * error[i,j,k]
    dT = 2.0 * error
    
    # dU[i,j,r] = sum_{k} dT[i,j,k] * V[j,k,r] * W[k,i,r]
    # dV[j,k,r] = sum_{i} dT[i,j,k] * U[i,j,r] * W[k,i,r]
    # dW[k,i,r] = sum_{j} dT[i,j,k] * U[i,j,r] * V[j,k,r]
    
    # Using einsum for gradients
    # dU_3d[i,j,r] = einsum('ijk,jkr,kir->ijr', dT, V_3d, W_3d)
    dU_3d = np.einsum('ijk,jkr,kir->ijr', dT, V_3d, W_3d)
    # dV_3d[j,k,r] = einsum('ijk,ijr,kir->jkr', dT, U_3d, W_3d)
    dV_3d = np.einsum('ijk,ijr,kir->jkr', dT, U_3d, W_3d)
    # dW_3d[k,i,r] = einsum('ijk,ijr,jkr->kir', dT, U_3d, V_3d)
    dW_3d = np.einsum('ijk,ijr,jkr->kir', dT, U_3d, V_3d)
    
    dU = dU_3d.reshape(n*m, rank)
    dV = dV_3d.reshape(m*p, rank)
    dW = dW_3d.reshape(p*n, rank)
    
    return loss, dU, dV, dW

def _try_gradient_descent(target_rank, max_iter=2000, tol=1e-12, deadline=145.0, start_time=0):
    """Try to find a decomposition of given rank using gradient descent."""
    n, m, p = DIMENSIONS
    
    # Try a few random initializations
    for seed in range(5):
        if time.time() > deadline:
            return None
        
        np.random.seed(seed + target_rank * 1000)
        
        # Initialize with small random values
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Adam optimizer
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        t = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr = 0.05
        
        for step in range(max_iter):
            if time.time() > deadline:
                return None
            
            t += 1
            loss, dU, dV, dW = _compute_loss_grad(U, V, W)
            
            if loss < tol:
                if _verify_decomp(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return None
            
            # Adam update
            lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            mU = beta1 * mU + (1 - beta1) * dU
            vU = beta2 * vU + (1 - beta2) * dU**2
            U -= lr_t * mU / (np.sqrt(vU) + eps)
            
            mV = beta1 * mV + (1 - beta1) * dV
            vV = beta2 * vV + (1 - beta2) * dV**2
            V -= lr_t * mV / (np.sqrt(vV) + eps)
            
            mW = beta1 * mW + (1 - beta1) * dW
            vW = beta2 * vW + (1 - beta2) * dW**2
            W -= lr_t * mW / (np.sqrt(vW) + eps)
            
            # Decay learning rate
            if step % 500 == 499:
                lr *= 0.5
        
        # Check final
        if loss < tol:
            if _verify_decomp(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 75
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try to find a decomposition with rank below schoolbook
    # Start from 70 and go down. If we find one, we can try lower.
    # Given the difficulty, we might not find much below 75, but vectorized GD is faster.
    
    # We can also try to start from schoolbook and prune.
    # But for now, let's just run GD for various ranks.
    
    for target_rank in range(70, 49, -1):
        if time.time() > deadline - 10:
            break
        
        result = _try_gradient_descent(target_rank, max_iter=1000, tol=1e-12, deadline=deadline - 5, start_time=start_time)
        
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Found a valid decomposition, try to go lower
            continue
    
    if best_result is not None:
        return best_result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
