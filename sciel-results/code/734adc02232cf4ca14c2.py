"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    total_terms = n * m * p  # 24
    
    deadline = time.time() + 110
    
    def verify(U, V, W, r):
        """Check if the decomposition is valid."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    def schoolbook():
        r = total_terms
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, r
    
    # Try to find a lower rank decomposition
    # Start with schoolbook and try to reduce
    
    # Strategy: Try specific known lower-rank constructions
    # For 2x3x4, try rank 12, 14, 16, etc.
    
    best = schoolbook()
    
    # Try a structured approach: group by j
    # For each j, the sub-problem is 2x1x4 multiplication (rank 8 in schoolbook)
    # But we can share across j if we're clever.
    
    # Let me try rank 16 as a starting point for optimization
    # Use random initialization and gradient-like descent
    
    for target_r in [12, 14, 16, 18, 20, 22, 23]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 1000 + target_r)
            U = np.random.randn(nm, target_r) * 0.5
            V = np.random.randn(mp, target_r) * 0.5
            W = np.random.randn(pn, target_r) * 0.5
            
            # Simple iterative refinement
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Compute residuals
                # For each (i,j,k): residual = 1 - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # Update U: for each (i,j,k), dU[i*m+j,c] += lr * residual * V[j*p+k,c] * W[k*n+i,c]
                lr = 0.01
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_r):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            resid = 1.0 - s
                            for c in range(target_r):
                                dU[i*m+j, c] += lr * resid * V[j*p+k, c] * W[k*n+i, c]
                                dV[j*p+k, c] += lr * resid * U[i*m+j, c] * W[k*n+i, c]
                                dW[k*n+i, c] += lr * resid * U[i*m+j, c] * V[j*p+k, c]
                
                U += dU
                V += dV
                W += dW
                
                # Check convergence
                max_resid = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_r):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            max_resid = max(max_resid, abs(1.0 - s))
                
                if max_resid < 1e-6:
                    found = True
                    break
            
            if found:
                if verify(U, V, W, target_r):
                    best = (U, V, W, target_r)
                    break
        
        if found:
            break
    
    U, V, W, r = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
