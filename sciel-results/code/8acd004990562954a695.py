import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 4, 16, 16
    
    # Build the target tensor T as a list of (a,b,c) indices where T=1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    
    # All possible (a,b,c) triples
    all_triples = [(a, b, c) for a in range(nm) for b in range(mp) for c in range(pn)]
    target_set = set(target_indices)
    
    start_time = time.time()
    deadline = start_time + 100  # Leave 50s margin
    
    # Try ranks from low to high
    for rank in [16, 18, 20, 22, 24, 26, 28, 30, 32]:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        best_result = None
        for trial in range(10):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + rank * 100 + trial)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Define the residual: we want T[a,b,c] = sum_i U[a,i]*V[b,i]*W[c,i]
            # For target indices, this should be 1; for others, 0.
            
            def loss(params):
                U = params[:nm*rank].reshape(nm, rank)
                V = params[nm*rank:2*nm*rank].reshape(mp, rank)
                W = params[2*nm*rank:].reshape(pn, rank)
                
                # Compute all T entries
                # T[a,b,c] = sum_i U[a,i] * V[b,i] * W[c,i]
                # This is a tensor, compute via einsum
                T = np.einsum('ai,bi,ci->abc', U, V, W)
                
                # Target: 1 for target_indices, 0 otherwise
                target = np.zeros((nm, mp, pn))
                for a, b, c in target_indices:
                    target[a, b, c] = 1
                
                residual = T - target
                return np.sum(residual**2)
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use gradient descent or L-BFGS
            try:
                result = minimize(loss, params, method='L-BFGS-B', maxiter=50, options={'maxfun': 200})
                final_loss = result.fun
            except:
                continue
            
            if final_loss < 1e-6:
                best_result = (rank, result.x)
                break
        
        if best_result:
            r, params = best_result
            U = params[:nm*r].reshape(nm, r)
            V = params[nm*r:2*nm*r].reshape(mp, r)
            W = params[2*nm*r:].reshape(pn, r)
            
            # Round to nearest integer or half-integer if close
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            T_check = np.einsum('ai,bi,ci->abc', U_rounded, V_rounded, W_rounded)
            target = np.zeros((nm, mp, pn))
            for a, b, c in target_indices:
                target[a, b, c] = 1
            if np.max(np.abs(T_check - target)) < 1e-6:
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
            # If rounding didn't work, try with original
            T_check = np.einsum('ai,bi,ci->abc', U, V, W)
            if np.max(np.abs(T_check - target)) < 1e-6:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = nm * mp * pn // (nm * mp * pn)  # Just use schoolbook
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
