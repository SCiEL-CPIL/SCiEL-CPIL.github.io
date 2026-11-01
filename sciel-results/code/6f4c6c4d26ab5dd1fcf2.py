"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s to stay safely under 150s
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Strategy: Find rank-11 decomposition for 2x3x2, then use it 3 times for 2x3x6
    # This gives rank 33 vs schoolbook 36
    
    def find_2x3x2_decomposition(rank=11, max_iter=50, time_budget=30):
        """Find a rank-r decomposition for 2x3x2 using gradient descent."""
        n2, m2, p2 = 2, 3, 2
        start = time.time()
        
        # Target: T[i,j,k] = 1 for all i,j,k
        # U: (n2*m2, rank), V: (m2*p2, rank), W: (p2*n2, rank)
        
        for attempt in range(max_iter):
            if time.time() - start > time_budget:
                break
            
            # Initialize with small random values
            U = np.random.randn(n2 * m2, rank) * 0.1
            V = np.random.randn(m2 * p2, rank) * 0.1
            W = np.random.randn(p2 * n2, rank) * 0.1
            
            # Gradient descent
            lr = 0.01
            for step in range(2000):
                if time.time() - start > time_budget:
                    break
                
                # Compute current tensor values
                residuals = np.zeros((n2, m2, p2))
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            residuals[i,j,k] = sum(
                                U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r] 
                                for r in range(rank)
                            ) - 1.0
                
                # Check convergence
                if np.max(np.abs(residuals)) < 1e-8:
                    return U, V, W
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            res = residuals[i,j,k]
                            for r in range(rank):
                                dU[i*m2+j, r] += res * V[j*p2+k, r] * W[k*n2+i, r]
                                dV[j*p2+k, r] += res * U[i*m2+j, r] * W[k*n2+i, r]
                                dW[k*n2+i, r] += res * U[i*m2+j, r] * V[j*p2+k, r]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate
                if step % 100 == 0:
                    lr *= 0.95
        
        return None
    
    # Try to find rank-11 decomposition for 2x3x2
    if time.time() < deadline - 20:
        result = find_2x3x2_decomposition(rank=11, max_iter=30, time_budget=120)
        if result is not None:
            U232, V232, W232 = result
            # Use this decomposition 3 times for 2x3x6
            rank = 33
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            for block in range(3):
                base = block * 11
                for i in range(n):
                    for j in range(m):
                        for r in range(11):
                            U[i*m+j][base+r] = U232[i*2+j, r]
                for j in range(m):
                    for k in range(2):
                        for r in range(11):
                            V[j*p + block*2 + k][base+r] = V232[j*2+k, r]
                for k in range(2):
                    for i in range(n):
                        for r in range(11):
                            W[(block*2+k)*n + i][base+r] = W232[k*2+i, r]
            
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
