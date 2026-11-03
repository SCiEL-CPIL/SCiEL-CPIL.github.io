"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 130  # safety margin before 150s kill
    
    # Tensor T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # U shape: (n*m, r) = (9, r)
    # V shape: (m*p, r) = (12, r)  
    # W shape: (p*n, r) = (12, r)
    
    # Precompute index mappings for vectorized evaluation
    # For each (i,j,k), we need U[i*m+j], V[j*p+k], W[k*n+i]
    idx_U = []
    idx_V = []
    idx_W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U.append(i * m + j)
                idx_V.append(j * p + k)
                idx_W.append(k * n + i)
    idx_U = np.array(idx_U)
    idx_V = np.array(idx_V)
    idx_W = np.array(idx_W)
    num_eqs = n * m * p  # 36 equations
    
    def compute_residual(U, V, W, r):
        """Compute residual vector: for each (i,j,k), sum_c U*V*W - 1"""
        # U[idx_U, :] has shape (num_eqs, r)
        # V[idx_V, :] has shape (num_eqs, r)
        # W[idx_W, :] has shape (num_eqs, r)
        # Element-wise multiply then sum over c
        prod = U[idx_U, :] * V[idx_V, :] * W[idx_W, :]
        sums = np.sum(prod, axis=1)
        return sums - 1.0
    
    def compute_error(U, V, W, r):
        """Compute sum of squared residuals"""
        res = compute_residual(U, V, W, r)
        return np.sum(res ** 2)
    
    def compute_gradients(U, V, W, r):
        """Compute gradients of error w.r.t. U, V, W"""
        res = compute_residual(U, V, W, r)  # shape (num_eqs,)
        # grad_U[i*m+j, c] = sum over (i,j,k) where idx_U = i*m+j of res * V[idx_V] * W[idx_W]
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # For each equation e and each component c:
        # d(err)/dU[idx_U[e], c] += 2 * res[e] * V[idx_V[e], c] * W[idx_W[e], c]
        # Vectorized: for each c, grad_U[idx_U, c] += 2 * res * V[idx_V, c] * W[idx_W, c]
        for c in range(r):
            grad_U[idx_U, c] += 2.0 * res * V[idx_V, c] * W[idx_W, c]
            grad_V[idx_V, c] += 2.0 * res * U[idx_U, c] * W[idx_W, c]
            grad_W[idx_W, c] += 2.0 * res * U[idx_U, c] * V[idx_V, c]
        
        return grad_U, grad_V, grad_W
    
    def optimize(U, V, W, r, max_iters=500, tol=1e-10):
        """L-BFGS-like or simple gradient descent with momentum"""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        curr_err = compute_error(U, V, W, r)
        
        # Use Adam optimizer for better convergence
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lr = 0.01
        
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        t = 0
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            grad_U, grad_V, grad_W = compute_gradients(U, V, W, r)
            
            t += 1
            # Adam update
            m_U = beta1 * m_U + (1 - beta1) * grad_U
            v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
            m_U_hat = m_U / (1 - beta1 ** t)
            v_U_hat = v_U / (1 - beta2 ** t)
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
            m_W_hat = m_W / (1 - beta1 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            new_err = compute_error(U, V, W, r)
            if new_err < curr_err - 1e-12:
                curr_err = new_err
            else:
                # Revert
                U += lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                V += lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                W += lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            if curr_err < tol:
                break
        
        return U, V, W, curr_err
    
    def snap_to_rationals(U, V, W, r, tol=1e-4):
        """Try to snap values to nearest half-integers or integers"""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        for mat_list in [U_list, V_list, W_list]:
            for row in mat_list:
                for idx in range(len(row)):
                    val = row[idx]
                    # Try half-integer
                    snapped_half = round(val * 2) / 2.0
                    if abs(val - snapped_half) < tol:
                        row[idx] = snapped_half
                    else:
                        # Try integer
                        snapped_int = round(val)
                        if abs(val - snapped_int) < tol:
                            row[idx] = snapped_int
        
        return U_list, V_list, W_list
    
    def validate_exact(U, V, W, r, tol=1e-8):
        """Check if decomposition is exact"""
        res = compute_residual(np.array(U), np.array(V), np.array(W), r)
        return np.max(np.abs(res)) < tol
    
    best_rank = n * m * p
    best_decomp = None
    
    # Try ranks from a reasonable lower bound up
    # Lower bound for 3x3x4: at least max(n*m, m*p, p*n) = max(9,12,12) = 12
    # But realistically, try from around 20-36
    for r in range(20, best_rank + 1):
        if time.time() > deadline:
            break
        
        found = False
        num_trials = 30 if r <= 25 else 15
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + trial * 7919 + r * 13)
            
            # Initialize with small random values
            U = np.random.uniform(-0.5, 0.5, (n * m, r))
            V = np.random.uniform(-0.5, 0.5, (m * p, r))
            W = np.random.uniform(-0.5, 0.5, (p * n, r))
            
            # Optimize
            U, V, W, err = optimize(U, V, W, r, max_iters=300)
            
            if err < 1e-6:
                # Try to snap to rationals
                U_list, V_list, W_list = snap_to_rationals(U, V, W, r)
                
                if validate_exact(U_list, V_list, W_list, r):
                    best_rank = r
                    best_decomp = (U_list, V_list, W_list)
                    found = True
                    break
                elif validate_exact(U.tolist(), V.tolist(), W.tolist(), r):
                    best_rank = r
                    best_decomp = (U.tolist(), V.tolist(), W.tolist())
                    found = True
                    break
        
        if found:
            break
    
    if best_decomp is not None:
        U, V, W = best_decomp
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
