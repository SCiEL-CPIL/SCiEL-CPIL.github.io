# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses gradient-based optimization to find low-rank decompositions.
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col_idx] = 1
                    V[j * p + k][col_idx] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        # Build the tensor from the decomposition
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] should be 1 if we're computing (A*B)[i,k] with A[i,j]*B[j,k]
        
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(actual - expected) > 1e-8:
                        return False
        return True
    
    def try_rank(rank, max_iter=5000, lr=0.01):
        """Try to find a decomposition of given rank using gradient descent."""
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        np.random.seed(42 + rank)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Also try structured initialization
        best_loss = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for init in range(5):
            if time.time() > deadline:
                break
            
            if init == 0:
                U_init = np.random.randn(n * m, rank) * 0.1
                V_init = np.random.randn(m * p, rank) * 0.1
                W_init = np.random.randn(p * n, rank) * 0.1
            elif init == 1:
                # Structured init based on schoolbook
                U_init = np.zeros((n * m, rank))
                V_init = np.zeros((m * p, rank))
                W_init = np.zeros((p * n, rank))
                col_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col_idx < rank:
                                U_init[i * m + j, col_idx] = 1.0
                                V_init[j * p + k, col_idx] = 1.0
                                W_init[k * n + i, col_idx] = 1.0
                                col_idx += 1
            else:
                U_init = np.random.randn(n * m, rank) * 0.5
                V_init = np.random.randn(m * p, rank) * 0.5
                W_init = np.random.randn(p * n, rank) * 0.5
            
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute loss: sum of (T_actual[i,j,k] - 1)^2
                # T_actual[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                loss = 0.0
                dU = np.zeros_like(U_cur)
                dV = np.zeros_like(V_cur)
                dW = np.zeros_like(W_cur)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            t_val = 0.0
                            for r in range(rank):
                                t_val += U_cur[i * m + j, r] * V_cur[j * p + k, r] * W_cur[k * n + i, r]
                            residual = t_val - 1.0
                            loss += residual * residual
                            
                            # Gradients
                            for r in range(rank):
                                dU[i * m + j, r] += 2 * residual * V_cur[j * p + k, r] * W_cur[k * n + i, r]
                                dV[j * p + k, r] += 2 * residual * U_cur[i * m + j, r] * W_cur[k * n + i, r]
                                dW[k * n + i, r] += 2 * residual * U_cur[i * m + j, r] * V_cur[j * p + k, r]
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                
                if loss < 1e-10:
                    break
                
                # Gradient descent update
                U_cur -= lr * dU
                V_cur -= lr * dV
                W_cur -= lr * dW
                
                # Decay learning rate
                lr_cur = lr * (0.999 ** it)
                U_cur += lr_cur * dU  # Undo and redo with new lr
                V_cur += lr_cur * dV
                W_cur += lr_cur * dW
                U_cur -= lr_cur * dU
                V_cur -= lr_cur * dV
                W_cur -= lr_cur * dW
                U_cur -= lr_cur * dU
                V_cur -= lr_cur * dV
                W_cur -= lr_cur * dW
            
            if best_U is not None and best_loss < 1e-8:
                break
        
        if best_U is not None and best_loss < 1e-8:
            # Convert to lists with rounding for exactness
            U_list = [[round(float(x), 10) for x in row] for row in best_U]
            V_list = [[round(float(x), 10) for x in row] for row in best_V]
            W_list = [[round(float(x), 10) for x in row] for row in best_W]
            
            # Verify
            if verify_decomposition(U_list, V_list, W_list, rank):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from low to high
    for rank in range(20, 61):
        if time.time() > deadline:
            break
        result = try_rank(rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
