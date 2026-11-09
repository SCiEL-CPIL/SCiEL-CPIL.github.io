"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Build schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Try to find a lower rank decomposition using structured approaches
    # For (2,3,7), the theoretical lower bound is max(2*7, 3*7, 2*3) = 21
    # but practical bounds suggest higher. Let's try ranks from 35 down to 25
    
    # Strategy: Use a structured initialization based on Kruskal-Cannon analysis
    # and try multiple random restarts with better optimization
    
    best_result = result
    best_rank = rank
    
    # Try ranks from 35 down to 28
    for r in range(35, 27, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with structured random values
            # Use a mix of small integers and half-integers for better convergence
            np.random.seed(42 + r * 100 + restart)
            U_rand = np.random.randint(-2, 3, size=(n*m, r)).astype(float) * 0.5
            V_rand = np.random.randint(-2, 3, size=(m*p, r)).astype(float) * 0.5
            W_rand = np.random.randint(-2, 3, size=(p*n, r)).astype(float) * 0.5
            
            # Ensure some non-zero entries
            if np.all(U_rand == 0):
                U_rand[0, 0] = 1
            if np.all(V_rand == 0):
                V_rand[0, 0] = 1
            if np.all(W_rand == 0):
                W_rand[0, 0] = 1
            
            # Gradient descent with adaptive learning rate
            lr = 0.05
            lr_decay = 0.999
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Compute gradient
                U_grad = np.zeros_like(U_rand)
                V_grad = np.zeros_like(V_rand)
                W_grad = np.zeros_like(W_rand)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(r):
                                U_grad[i*m+j, c] += V_rand[j*p+k, c] * W_rand[k*n+i, c]
                                V_grad[j*p+k, c] += U_rand[i*m+j, c] * W_rand[k*n+i, c]
                                W_grad[k*n+i, c] += U_rand[i*m+j, c] * V_rand[j*p+k, c]
                
                # Normalize gradients to avoid divergence
                U_norm = np.linalg.norm(U_grad)
                V_norm = np.linalg.norm(V_grad)
                W_norm = np.linalg.norm(W_grad)
                
                if U_norm > 0:
                    U_rand -= lr * U_grad / U_norm
                if V_norm > 0:
                    V_rand -= lr * V_grad / V_norm
                if W_norm > 0:
                    W_rand -= lr * W_grad / W_norm
                
                lr *= lr_decay
            
            # Round to half-integers
            U_rounded = np.round(U_rand * 2) / 2
            V_rounded = np.round(V_rand * 2) / 2
            W_rounded = np.round(W_rand * 2) / 2
            
            # Check if exact
            exact = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                        if abs(s - 1) > 1e-9:
                            exact = False
                            break
                    if not exact:
                        break
                if not exact:
                    break
            
            if exact:
                result = {'U': U_rounded.tolist(), 'V': V_rounded.tolist(), 'W': W_rounded.tolist()}
                best_result = result
                best_rank = r
                break
        
        if best_rank == r:
            break
    
    return best_result
# EVOLVE-BLOCK-END
