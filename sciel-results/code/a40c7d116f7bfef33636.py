import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition (rank 30)
    def schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Search for lower rank decompositions
    # We try ranks from best_rank - 1 down to 20
    for target_rank in range(best_rank - 1, 19, -1):
        if time.time() > deadline:
            break
        
        # Number of variables
        nU = n * m * target_rank
        nV = m * p * target_rank
        nW = p * n * target_rank
        total_vars = nU + nV + nW
        
        def objective_and_gradient(x):
            U = x[:nU].reshape(n*m, target_rank)
            V = x[nU:nU+nV].reshape(m*p, target_rank)
            W = x[nU+nV:].reshape(p*n, target_rank)
            
            # Compute the reconstructed tensor S
            # S[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            S = np.einsum('ar,br,cr->abc', U, V, W)
            
            error = S - T
            val = np.sum(error**2)
            
            # Gradients
            # d(val)/dU[a,r] = 2 * sum_b,c error[a,b,c] V[b,r] W[c,r]
            # d(val)/dV[b,r] = 2 * sum_a,c error[a,b,c] U[a,r] W[c,r]
            # d(val)/dW[c,r] = 2 * sum_a,b error[a,b,c] U[a,r] V[b,r]
            
            gU = 2.0 * np.einsum('abc,br,cr->ar', error, V, W)
            gV = 2.0 * np.einsum('abc,ar,cr->br', error, U, W)
            gW = 2.0 * np.einsum('abc,ar,br->cr', error, U, V)
            
            grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            return val, grad

        # Number of restarts depends on the rank
        # Higher ranks are easier, lower ranks are harder
        restarts = 30 if target_rank > 27 else 10
        
        for r in range(restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            # Use small values to avoid overflow/underflow initially
            x0 = np.random.randn(total_vars) * 0.1
            
            try:
                # Use L-BFGS-B for efficiency
                res = opt.minimize(
                    objective_and_gradient,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue
            
            if res.success and res.fun < 1e-8:
                # Extract solution
                U = res.x[:nU].reshape(n*m, target_rank)
                V = res.x[nU:nU+nV].reshape(m*p, target_rank)
                W = res.x[nU+nV:].reshape(p*n, target_rank)
                
                # Verify
                S = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(S, T, atol=1e-6):
                    # Round to nearest half-integer if close?
                    # The problem statement mentions integer/half-integer.
                    # If the solution is numerical, it might be close to these.
                    # We return the float values as is, as they satisfy the decomposition.
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If no solution found for this rank, continue to next lower rank
        # (We don't update best_result here because we only return if we find a valid one)
        
    return best_result
