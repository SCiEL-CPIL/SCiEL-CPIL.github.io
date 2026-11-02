"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Build the target tensor T
    # T[i,j,j',k,i',k'] = 1 if i==i', j==j', k==k' else 0
    # But we only need to check the "diagonal" conditions
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid decomposition."""
        rank = U.shape[1]
        # Check: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        # Check: for all other combinations, should be 0
        # This is expensive, so check a subset
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i==i2 and j==j2 and k==k2:
                                    continue
                                s = 0.0
                                for r in range(rank):
                                    s += U[i*m+j, r] * V[j2*p+k2, r] * W[k2*n+i2, r]
                                if abs(s) > tol:
                                    return False
        return True
    
    def try_rank(R, max_iter=500):
        """Try to find a decomposition of rank R."""
        if time.time() > deadline - 30:
            return None
        
        np.random.seed(42)
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        best_err = np.inf
        best_U, best_V, best_W = None, None, None
        
        for it in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            # Compute error: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
            # For other combos, should be 0
            
            # Use gradient descent
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:])
                        err += (s - 1.0)**2
                        # Gradient
                        g = (s - 1.0) * 2 * V[j*p+k,:] * W[k*n+i,:]
                        dU[i*m+j,:] += g
                        g2 = (s - 1.0) * 2 * U[i*m+j,:] * W[k*n+i,:]
                        dV[j*p+k,:] += g2
                        g3 = (s - 1.0) * 2 * U[i*m+j,:] * V[j*p+k,:]
                        dW[k*n+i,:] += g3
            
            # Also add penalty for off-diagonal terms (sample some)
            for _ in range(100):
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                i2 = np.random.randint(n)
                j2 = np.random.randint(m)
                k2 = np.random.randint(p)
                if i==i2 and j==j2 and k==k2:
                    continue
                s = np.sum(U[i*m+j,:] * V[j2*p+k2,:] * W[k2*n+i2,:])
                err += s**2
                g = s * 2 * V[j2*p+k2,:] * W[k2*n+i2,:]
                dU[i*m+j,:] += g
                g2 = s * 2 * U[i*m+j,:] * W[k2*n+i2,:]
                dV[j2*p+k2,:] += g2
                g3 = s * 2 * U[i*m+j,:] * V[j2*p+k2,:]
                dW[k2*n+i2,:] += g3
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Update
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_err < 1e-6:
            return best_U, best_V, best_W
        return None
    
    # Try decreasing ranks
    for R in [96, 88, 80, 72, 64, 56, 48, 40]:
        if time.time() > deadline - 40:
            break
        result = try_rank(R)
        if result is not None:
            U, V, W = result
            # Convert to list format
            U_list = [[float(U[i,j]) for j in range(R)] for i in range(n*m)]
            V_list = [[float(V[i,j]) for j in range(R)] for i in range(m*p)]
            W_list = [[float(W[i,j]) for j in range(R)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
