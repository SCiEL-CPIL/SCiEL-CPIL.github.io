"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 140.0  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
    
    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        rank = U_arr.shape[1]
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            max_err = np.max(np.abs(T - 1.0))
            return max_err < tol
        except Exception:
            return False

    def tensor_error(U, V, W):
        """Compute sum of squared errors for tensor decomposition."""
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        err = np.sum((T - 1.0) ** 2)
        return err

    def gradient_descent(U, V, W, lr=0.01, steps=2000, tol=1e-10):
        """Perform gradient descent to minimize tensor error using vectorization."""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        best_err = float('inf')
        best_U = U_3d.copy()
        best_V = V_3d.copy()
        best_W = W_3d.copy()
        
        for step in range(steps):
            if time.time() > _deadline - 10:
                break
            
            # Compute T[i,j,k]
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            err = np.sum((T - 1.0) ** 2)
            
            if err < best_err:
                best_err = err
                best_U = U_3d.copy()
                best_V = V_3d.copy()
                best_W = W_3d.copy()
                
            if err < tol:
                break
                
            # Gradients
            dT = T - 1.0
            
            # grad_U[i,j,r] = 2 * sum_k (T[i,j,k] - 1) * V[j,k,r] * W[k,i,r]
            grad_U = 2 * np.einsum('ijk,jkr,kir->ijr', dT, V_3d, W_3d)
            
            # grad_V[j,k,r] = 2 * sum_i (T[i,j,k] - 1) * U[i,j,r] * W[k,i,r]
            grad_V = 2 * np.einsum('ijk,ijr,kir->jkr', dT, U_3d, W_3d)
            
            # grad_W[k,i,r] = 2 * sum_j (T[i,j,k] - 1) * U[i,j,r] * V[j,k,r]
            grad_W = 2 * np.einsum('ijk,ijr,jkr->kir', dT, U_3d, V_3d)
            
            # Update
            U_3d -= lr * grad_U
            V_3d -= lr * grad_V
            W_3d -= lr * grad_W
            
            # Adaptive learning rate step (simple decay)
            if step % 100 == 0 and step > 0:
                lr *= 0.95
        
        return best_U.reshape(n * m, rank), best_V.reshape(m * p, rank), best_W.reshape(p * n, rank), best_err

    def random_search(max_rank, num_trials=20):
        """Try to find a decomposition of rank max_rank using gradient descent."""
        for trial in range(num_trials):
            if time.time() > _deadline - 15:
                break
            
            seed = 42 + trial * 1000
            np.random.seed(seed)
            
            # Random initialization
            scale = 0.5
            U = np.random.randn(n * m, max_rank) * scale
            V = np.random.randn(m * p, max_rank) * scale
            W = np.random.randn(p * n, max_rank) * scale
            
            # Run gradient descent
            U, V, W, err = gradient_descent(U, V, W, lr=0.05, steps=3000)
            
            # Check if we found a solution
            if err < 1e-9:
                return U.tolist(), V.tolist(), W.tolist()
            
        return None

    # Try ranks from 39 down to 20
    for rank in range(39, 19, -1):
        if time.time() > _deadline - 30:
            break
        
        # Adjust trials based on rank?
        num_trials = 50 if rank >= 35 else 20
        
        result = random_search(rank, num_trials=num_trials)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
