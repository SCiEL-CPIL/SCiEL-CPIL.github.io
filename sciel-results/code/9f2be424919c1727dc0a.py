"""Program for 4x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]
DEADLINE = 130  # seconds, safety margin before 150s kill

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5x6 matrix multiplication.
    Returns dict with 'U', 'V', 'W' factor matrices.
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    schoolbook_rank = n * m * p  # 120
    
    start_time = time.time()
    
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook_decomposition()
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify that the decomposition exactly reproduces matrix multiplication tensor."""
        # U_mat: (n*m, rank), V_mat: (m*p, rank), W_mat: (p*n, rank)
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        U_r = U_mat.reshape(n, m, -1)
        V_r = V_mat.reshape(m, p, -1)
        W_r = W_mat.reshape(p, n, -1)
        # Compute T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # Using einsum: 'ijr,jkr,kir->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    # Verify schoolbook
    assert verify(U, V, W), "Schoolbook decomposition failed verification"
    
    # Optimization-based approach to find lower rank decomposition
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    def optimize_rank(target_rank, max_iter=2000, lr=0.05, seed=42):
        """
        Try to find a rank-`target_rank` decomposition using gradient descent.
        Uses analytical gradients for efficiency.
        """
        if target_rank >= best_rank:
            return None
        
        np.random.seed(seed)
        
        # Initialize
        U_opt = np.random.randn(n * m, target_rank) * 0.5
        V_opt = np.random.randn(m * p, target_rank) * 0.5
        W_opt = np.random.randn(p * n, target_rank) * 0.5
        
        # Reshape for easier gradient computation
        # U_r: (n, m, r), V_r: (m, p, r), W_r: (p, n, r)
        
        def compute_loss_and_grads():
            U_r = U_opt.reshape(n, m, target_rank)
            V_r = V_opt.reshape(m, p, target_rank)
            W_r = W_opt.reshape(p, n, target_rank)
            
            # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
            T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            loss = np.sum((T - 1.0) ** 2)
            
            # Gradients:
            # dL/dU_r[i,j,r] = 2 * sum_k (T[i,j,k] - 1) * V_r[j,k,r] * W_r[k,i,r]
            # dL/dV_r[j,k,r] = 2 * sum_i (T[i,j,k] - 1) * U_r[i,j,r] * W_r[k,i,r]
            # dL/dW_r[k,i,r] = 2 * sum_j (T[i,j,k] - 1) * U_r[i,j,r] * V_r[j,k,r]
            
            diff = T - 1.0
            
            # dL/dU_r: (n, m, r)
            dU_r = 2 * np.einsum('ijk,jkr,kir->ijr', diff, V_r, W_r)
            # dL/dV_r: (m, p, r)
            dV_r = 2 * np.einsum('ijk,ijr,kir->jkr', diff, U_r, W_r)
            # dL/dW_r: (p, n, r)
            dW_r = 2 * np.einsum('ijk,ijr,jkr->kir', diff, U_r, V_r)
            
            dU = dU_r.reshape(n * m, target_rank)
            dV = dV_r.reshape(m * p, target_rank)
            dW = dW_r.reshape(p * n, target_rank)
            
            return loss, dU, dV, dW
        
        # Adam optimizer state
        mU = np.zeros_like(U_opt)
        vU = np.zeros_like(U_opt)
        mV = np.zeros_like(V_opt)
        vV = np.zeros_like(V_opt)
        mW = np.zeros_like(W_opt)
        vW = np.zeros_like(W_opt)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t = 0
        
        start_time_opt = time.time()
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            t += 1
            loss, dU, dV, dW = compute_loss_and_grads()
            
            if loss < 1e-12:
                return U_opt, V_opt, W_opt
            
            # Adam update
            mU = beta1 * mU + (1 - beta1) * dU
            vU = beta2 * vU + (1 - beta2) * dU ** 2
            mU_hat = mU / (1 - beta1 ** t)
            vU_hat = vU / (1 - beta2 ** t)
            U_opt -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            mV = beta1 * mV + (1 - beta1) * dV
            vV = beta2 * vV + (1 - beta2) * dV ** 2
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            V_opt -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            mW = beta1 * mW + (1 - beta1) * dW
            vW = beta2 * vW + (1 - beta2) * dW ** 2
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W_opt -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            if it % 100 == 0:
                pass  # Could log loss
        
        # Check if we got close enough
        U_r = U_opt.reshape(n, m, target_rank)
        V_r = V_opt.reshape(m, p, target_rank)
        W_r = W_opt.reshape(p, n, target_rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        max_err = np.max(np.abs(T - 1.0))
        
        if max_err < 1e-9:
            return U_opt, V_opt, W_opt
        
        return None
    
    # Try to find lower rank decompositions
    # Start from just below schoolbook and work down
    for target_rank in range(119, 80, -1):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        found = False
        for seed in range(5):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            result = optimize_rank(target_rank, max_iter=1500, lr=0.05, seed=seed + target_rank)
            if result is not None:
                U_opt, V_opt, W_opt = result
                if verify(U_opt, V_opt, W_opt):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    found = True
                    break
        
        if found:
            # Try to go even lower
            continue
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
