"""Matrix multiplication tensor decomposition for 3x4x8."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # 110s safety margin within 120s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 96
    
    # Known theoretical lower bound for M_{3,4,8} is around 41-48
    # We'll try to find a decomposition with rank < 96
    
    # Strategy: Try to find a low-rank decomposition using numerical optimization
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    # We need U, V, W such that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # First, let's verify what ranks are feasible by trying smaller ranks
    # We know rank 96 works (schoolbook). Let's try to find lower.
    
    # For 3x4x8, known results suggest rank <= 72 might be achievable
    # Let's try a structured approach using block decomposition
    
    # Try rank values from a reasonable lower bound up to schoolbook
    # Known: M_{3,4,8} rank is at most 72 (from literature)
    # Let's try to construct explicitly
    
    # Approach: Use the fact that 3x4x8 can be decomposed using
    # a combination of smaller matrix multiplications
    
    # Actually, let me try a direct numerical approach for specific ranks
    # Starting from a random initialization and optimizing
    
    def try_rank(r):
        """Try to find a decomposition of rank r."""
        if time.time() > deadline - 5:
            return None
        
        # Build the tensor
        T = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1.0
        
        # Parameterize: U (nm x r), V (mp x r), W (pn x r)
        # Total params: r*(nm + mp + pn) = r*(12+32+24) = 68r
        # For r=60: 4080 params
        
        # Initialize randomly
        np.random.seed(42 + r)
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        # Loss function: ||sum_r U[:,r] V[:,r] W[:,r] - T||^2
        def loss(params):
            U_flat, V_flat, W_flat = params[:nm*r], params[nm*r:nm*r+mp*r], params[nm*r+mp*r:68*r]
            U = U_flat.reshape(nm, r)
            V = V_flat.reshape(mp, r)
            W = W_flat.reshape(pn, r)
            # Compute tensor decomposition
            # T_pred[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # This is expensive to compute fully, so use a subset for gradient
            residual = 0.0
            # Compute full residual
            T_pred = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_pred - T
            return np.sum(diff**2)
        
        def loss_grad(params):
            U_flat, V_flat, W_flat = params[:nm*r], params[nm*r:nm*r+mp*r], params[nm*r+mp*r:68*r]
            U = U_flat.reshape(nm, r)
            V = V_flat.reshape(mp, r)
            W = W_flat.reshape(pn, r)
            
            T_pred = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_pred - T
            
            # Gradients
            dU = np.einsum('abc,br,cr->ar', diff, V, W)
            dV = np.einsum('abc,ar,cr->br', diff, U, W)
            dW = np.einsum('abc,ar,br->cr', diff, U, V)
            
            grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            return 2 * (loss(params) + grad @ (U_flat + V_flat + W_flat))  # placeholder
        
        # Use L-BFGS-B for optimization
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(loss, x0, method='L-BFGS-B', 
                            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-8})
            
            if result.fun < 1e-6:
                U_opt, V_opt, W_opt = result.x[:nm*r].reshape(nm, r), \
                                      result.x[nm*r:nm*r+mp*r].reshape(mp, r), \
                                      result.x[nm*r+mp*r:68*r].reshape(pn, r)
                return U_opt, V_opt, W_opt
        except Exception:
            pass
        
        return None
    
    # Try decreasing ranks from a reasonable upper bound
    # Based on known results, try: 72, 64, 56, 48
    best_result = None
    for r in [72, 64, 56, 48]:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r)
        if result is not None:
            U_opt, V_opt, W_opt = result
            best_result = (U_opt, V_opt, W_opt, r)
            break
    
    if best_result is not None:
        U_opt, V_opt, W_opt, r = best_result
        # Convert to lists
        return {
            "U": U_opt.tolist(),
            "V": V_opt.tolist(),
            "W": W_opt.tolist()
        }
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
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
