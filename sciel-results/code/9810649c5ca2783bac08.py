import numpy as np
from itertools import product
import time

def construct():
    n, m, p = 2, 5, 5
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    total = n * m * p  # 50
    
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin for safety
    
    # Schoolbook solution (rank 50)
    U_school = np.zeros((nm, total), dtype=float)
    V_school = np.zeros((mp, total), dtype=float)
    W_school = np.zeros((pn, total), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    def verify(U, V, W):
        """Check if the decomposition is exact (within tolerance)"""
        # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Should equal 1.0
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    error += abs(s - 1.0)
        return error < 1e-8
    
    # Try to find lower-rank solutions using gradient descent
    # We'll try ranks from 40 down to 20
    best_rank = 50
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try various ranks
    for rank in range(40, 14, -1):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            np.random.seed(rank * 100 + attempt)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Gradient descent to minimize sum of squared errors
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline - 2:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # s = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err = s - 1.0
                            error += err * err
                            
                            # Gradients
                            for c in range(rank):
                                grad_U[i*m+j, c] += 2 * err * V[j*p+k, c] * W[k*n+i, c]
                                grad_V[j*p+k, c] += 2 * err * U[i*m+j, c] * W[k*n+i, c]
                                grad_W[k*n+i, c] += 2 * err * U[i*m+j, c] * V[j*p+k, c]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if error < 1e-12:
                    break
            
            # Check if we found a solution
            if error < 1e-8:
                # Try to snap to integer/half-integer
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if verify(U_snap, V_snap, W_snap):
                    best_rank = rank
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    found = True
                    break
                else:
                    # Use the floating point solution if it's exact enough
                    if verify(U, V, W):
                        best_rank = rank
                        best_U = U
                        best_V = V
                        best_W = W
                        found = True
                        break
        
        if found:
            # Try to go even lower
            continue
    
    return {
        'U': best_U.tolist(),
        'V': best_V.tolist(),
        'W': best_W.tolist()
    }
