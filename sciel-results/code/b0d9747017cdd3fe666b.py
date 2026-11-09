import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Build the set of valid (a,b,c) indices
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    
    # Trivial decomposition
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    def full_residual(U, V, W):
        """Compute the full residual including zero constraints."""
        r = U.shape[1]
        res = 0.0
        # Valid triples: should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    res += (val - 1.0) ** 2
        # Invalid triples: should equal 0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) not in valid:
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        res += val ** 2
        return res
    
    # This is too slow. Let me use a different approach.
    # Instead of checking all invalid triples, note that the tensor is sparse.
    # The number of invalid triples is n*m*p*(m*p*n - 1) which is huge.
    
    # Alternative: use only the valid triple constraints and hope that
    # a solution satisfying those also satisfies the zero constraints.
    # This is not guaranteed, but might work in practice.
    
    def residual_valid_only(U, V, W):
        """Residual using only valid triple constraints."""
        r = U.shape[1]
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    res += (val - 1.0) ** 2
        return res
    
    # Try gradient descent with only valid constraints
    for r in range(rank0 - 1, max(40, rank0 - 25), -1):
        if time.time() > deadline:
            break
        
        for trial in range(3):
            if time.time() > deadline:
                break
            
            indices = np.random.choice(rank0, r, replace=False)
            U = U0[:, indices].copy()
            V = V0[:, indices].copy()
            W = W0[:, indices].copy()
            
            U += np.random.randn(n*m, r) * 0.01
            V += np.random.randn(m*p, r) * 0.01
            W += np.random.randn(p*n, r) * 0.01
            
            # Gradient descent
            lr = 0.002
            for it in range(300):
                if time.time() > deadline:
                    break
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err = val - 1.0
                            dU[i*m+j, :] += 2 * err * (V[j*p+k, :] * W[k*n+i, :])
                            dV[j*p+k, :] += 2 * err * (U[i*m+j, :] * W[k*n+i, :])
                            dW[k*n+i, :] += 2 * err * (U[i*m+j, :] * V[j*p+k, :])
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            res = residual_valid_only(U, V, W)
            if res < 1e-8:
                # Verify full residual
                full_res = full_residual(U, V, W)
                if full_res < 1e-6:
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
