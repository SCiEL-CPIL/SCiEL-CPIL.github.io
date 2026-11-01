"""Matrix multiplication 2x3x7: attempt to find a rank-deficient exact decomposition."""
DIMENSIONS = [2, 3, 7]
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (120s limit, 10s buffer)
    
    n, m, p = DIMENSIONS
    rank_sb = n * m * p  # 42
    
    # Schoolbook decomposition
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if decomposition is valid."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        
        rank = U.shape[1]
        
        # Check: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r] = δ_{i,i'} δ_{j,j'} δ_{k,k'}
        for i in range(n):
            for ip in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                actual = 0.0
                                for r in range(rank):
                                    actual += U[i * m + j, r] * V[jp * p + k, r] * W[ip * n + kp, r]
                                if abs(actual - expected) > tol:
                                    return False
        return True
    
    def try_random_search(target_rank, max_attempts=50, time_budget=30):
        """Try random initialization with local optimization."""
        attempt_start = time.time()
        
        for attempt in range(max_attempts):
            if time.time() > start_time + deadline or time.time() > attempt_start + time_budget:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Gradient descent
            lr = 0.01
            for it in range(200):
                if time.time() > start_time + deadline:
                    break
                
                # Compute gradient
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for ip in range(n):
                        for j in range(m):
                            for jp in range(m):
                                for k in range(p):
                                    for kp in range(p):
                                        expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                        actual = 0.0
                                        for r in range(target_rank):
                                            actual += U[i*m+j, r] * V[jp*p+k, r] * W[ip*n+kp, r]
                                        err = actual - expected
                                        for r in range(target_rank):
                                            dU[i*m+j, r] += err * V[jp*p+k, r] * W[ip*n+kp, r]
                                            dV[jp*p+k, r] += err * U[i*m+j, r] * W[ip*n+kp, r]
                                            dW[ip*n+kp, r] += err * U[i*m+j, r] * V[jp*p+k, r]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if it % 50 == 0:
                    if check_decomposition(U.tolist(), V.tolist(), W.tolist(), tol=1e-6):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            if check_decomposition(U.tolist(), V.tolist(), W.tolist(), tol=1e-8):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try to find lower rank decompositions
    best_decomp = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try ranks from 41 down to 35 (limited by time)
    for target_rank in range(rank_sb - 1, max(30, rank_sb - 12), -1):
        if time.time() > start_time + deadline - 20:
            break
        
        result = try_random_search(target_rank, max_attempts=10, time_budget=15)
        if result is not None:
            best_decomp = result
            best_rank = target_rank
            break
    
    return best_decomp
# EVOLVE-BLOCK-END
