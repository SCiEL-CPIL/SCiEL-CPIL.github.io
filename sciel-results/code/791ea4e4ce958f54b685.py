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
    
    # Try to find a lower rank decomposition
    # For (2,3,7), try ranks from 38 down to 30
    for r in range(38, 29, -1):
        if time.time() > deadline:
            break
        
        # Try to find a rank-r decomposition using a structured approach
        # Initialize with schoolbook and try to zero out some columns
        U_try = np.zeros((n*m, r))
        V_try = np.zeros((m*p, r))
        W_try = np.zeros((p*n, r))
        
        # Fill with first r schoolbook terms
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= r:
                        break
                    U_try[i*m+j, col] = 1
                    V_try[j*p+k, col] = 1
                    W_try[k*n+i, col] = 1
                    col += 1
                    if col >= r:
                        break
                if col >= r:
                    break
            if col >= r:
                break
        
        # This won't work because we need all (i,j,k) triples to be covered
        
        # Better approach: use numerical optimization
        # Set up the problem as minimizing the sum of squared errors
        # subject to the constraint that all (i,j,k) entries equal 1
        
        # For each (i,j,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is a system of n*m*p = 42 equations in 3*r unknowns
        
        # Use gradient descent on the real-valued problem, then round
        
        U_rand = np.random.randn(n*m, r)
        V_rand = np.random.randn(m*p, r)
        W_rand = np.random.randn(p*n, r)
        
        # Scale to reasonable magnitude
        U_rand *= 1.0
        V_rand *= 1.0
        W_rand *= 1.0
        
        # Simple gradient descent
        lr = 0.01
        for it in range(500):
            if time.time() > deadline:
                break
            
            # Compute gradient
            U_grad = np.zeros_like(U_rand)
            V_grad = np.zeros_like(V_rand)
            W_grad = np.zeros_like(W_rand)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # f = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] - 1
                        # df/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c]
                        # df/dV[j*p+k,c] = U[i*m+j,c] * W[k*n+i,c]
                        # df/dW[k*n+i,c] = U[i*m+j,c] * V[j*p+k,c]
                        
                        for c in range(r):
                            val = U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_rand[k*n+i, c]
                            U_grad[i*m+j, c] += V_rand[j*p+k, c] * W_rand[k*n+i, c]
                            V_grad[j*p+k, c] += U_rand[i*m+j, c] * W_rand[k*n+i, c]
                            W_grad[k*n+i, c] += U_rand[i*m+j, c] * V_rand[j*p+k, c]
            
            U_rand -= lr * U_grad
            V_rand -= lr * V_grad
            W_rand -= lr * W_grad
        
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
            break
    
    return result
# EVOLVE-BLOCK-END
