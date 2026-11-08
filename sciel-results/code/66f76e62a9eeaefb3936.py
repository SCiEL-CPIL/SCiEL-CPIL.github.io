import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Ranks to try (based on known bounds)
    # For 4x5x5, rank is known to be at most 96 (schoolbook), but we can do better
    # Let's try ranks from 80 down to 50
    ranks_to_try = [80, 75, 70, 65, 60, 55, 50]
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
            
        # Try multiple restarts with different initialization strategies
        for restart in range(3):
            if time.time() > deadline - 10:
                break
                
            # Strategy 1: Block-structured initialization
            # Partition into 2x2 blocks where possible
            # For 4x5 x 5x5, we can think of it as:
            # A (4x5) * B (5x5) = C (4x5)
            # Split B into columns: B = [B1 | B2] where B1 is 5x2, B2 is 5x3
            # Then C = [A*B1 | A*B2]
            # A*B1 is 4x2, A*B2 is 4x3
            
            # Use Strassen-like decomposition for the 4x2 and 4x3 parts
            # For 4x2: can be done with rank 8 (schoolbook) or less
            # For 4x3: can be done with rank 12 (schoolbook) or less
            
            # Let's try a structured initialization
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Initialize with small random values
            np.random.seed(restart * 1000 + R)
            U = np.random.randn(n * m, R) * 0.05
            V = np.random.randn(m * p, R) * 0.05
            W = np.random.randn(p * n, R) * 0.05
            
            # Gradient-based optimization
            lr = 0.01
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                    
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Tensor entry: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            # Should equal 1
                            prod = U[i * m + j] * V[j * p + k] * W[k * n + i]
                            val = np.sum(prod)
                            err = val - 1
                            
                            # Gradients
                            grad_U[i * m + j] += err * (V[j * p + k] * W[k * n + i])
                            grad_V[j * p + k] += err * (U[i * m + j] * W[k * n + i])
                            grad_W[k * n + i] += err * (U[i * m + j] * V[j * p + k])
                
                # Update with learning rate
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                            err += (val - 1) ** 2
                if err < 1e-8:
                    break
            
            # Check if solution is good
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                        err += (val - 1) ** 2
                        
            if err < 1e-6:
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify exactness
                exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_snap[i * m + j], V_snap[j * p + k] * W_snap[k * n + i])
                            if abs(val - 1) > 1e-9:
                                exact = False
                                break
                        if not exact:
                            break
                    if not exact:
                        break
                
                if exact:
                    # Convert to list of lists
                    U_list = U_snap.tolist()
                    V_list = V_snap.tolist()
                    W_list = W_snap.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result
