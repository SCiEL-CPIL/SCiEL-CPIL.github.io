import numpy as np
import time

DIMENSIONS = [2, 2, 8]
_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def _check_deadline():
    return (time.time() - _START_TIME) >= _DEADLINE

def construct():
    n, m, p = DIMENSIONS
    
    # First, build the rank-28 Strassen solution as fallback
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    W_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    num_blocks = p // 2
    rank_28 = num_blocks * 7
    
    U28 = [[0] * rank_28 for _ in range(n * m)]
    V28 = [[0] * rank_28 for _ in range(m * p)]
    W28 = [[0] * rank_28 for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(n * m):
                U28[i][col] = U_strassen[r][i]
            
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V28[v_indices[l]][col] = V_strassen[r][l]
                
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W28[w_indices[l]][col] = W_strassen[r][l]
                
            col += 1
            
    # Verify the rank-28 solution
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify(U28, V28, W28, n, m, p):
        # Should not happen, but return anyway
        return {"U": U28, "V": V28, "W": W28}
    
    # Try to find a lower-rank decomposition
    # The tensor has 32 entries (i,j,k triples)
    # We need: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 for all i,j,k
    
    # Try rank 24, 20, etc.
    for target_rank in [24, 20, 16]:
        if _check_deadline():
            break
            
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        # Use gradient descent to minimize the error
        # Error = sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] - 1)^2
        
        lr = 0.01
        for step in range(500):
            if _check_deadline():
                break
                
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Current value
                        val = 0
                        for r in range(target_rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err = val - 1
                        
                        for r in range(target_rank):
                            grad_U[i*m+j, r] += err * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += err * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += err * U[i*m+j, r] * V[j*p+k, r]
                        
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check if we've converged
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(target_rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-8:
                # Found a valid decomposition!
                U_list = [[float(U[i][r]) for r in range(target_rank)] for i in range(n*m)]
                V_list = [[float(V[i][r]) for r in range(target_rank)] for i in range(m*p)]
                W_list = [[float(W[i][r]) for r in range(target_rank)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
        
    # Return the rank-28 solution
    return {"U": U28, "V": V28, "W": W28}
