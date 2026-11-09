import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    total_entries = nm * mp * pn
    
    # Precompute indices for non-zero entries of T
    # T_{a,b,c} = 1 if a=(i,j), b=(j,k), c=(k,i)
    # a = i*m + j, b = j*p + k, c = k*n + i
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_indices.append((a, b, c))
    
    # All indices
    all_indices = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                all_indices.append((a, b, c))
                
    # Map non-zero indices to target value 1, others 0
    # We only need to check the non-zero ones for the "1" constraint?
    # No, if we don't enforce the 0 constraints, the solution might have non-zero values there.
    # But if we minimize the sum of squares over ALL entries, it will try to make them 0 too.
    
    # Residual function
    def residuals(params):
        R = len(params) // (nm + mp + pn)
        # params is flattened U, V, W
        U = params[:R*nm].reshape(R, nm).T # shape (nm, R)
        V = params[R*nm:R*nm+R*mp].reshape(R, mp).T # shape (mp, R)
        W = params[R*nm+R*mp:].reshape(R, pn).T # shape (pn, R)
        
        # Compute tensor product sum
        # T = sum_r U[:,r] \otimes V[:,r] \otimes W[:,r]
        # We want T_{a,b,c} to match target.
        
        # Efficient computation:
        # For each r, U[:,r] V[:,r] is a matrix of size nm x mp.
        # We need to contract with W[:,r] to get the full tensor?
        # Actually, we just need the residuals.
        
        # Residual for each (a,b,c) is sum_r U[a,r] V[b,r] W[c,r] - target[a,b,c]
        
        # This is expensive to compute for all 2304 entries if done naively in Python.
        # Use vectorization.
        
        # Let's compute the tensor T_flat of size total_entries.
        # T_flat[idx] = sum_r U[a,r] V[b,r] W[c,r]
        
        # We can compute this as:
        # For each r:
        #   UV_r = U[:,r] @ V[:,r].T  (nm x mp)
        #   This doesn't directly give the full tensor.
        
        # Alternative:
        # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
        # This is a 3-way tensor.
        
        # We can compute the residuals for the non-zero indices only?
        # If we ignore the zero indices, we might find a solution that is non-zero there.
        # However, the "score" is the rank. The validity check likely checks ALL entries.
        # The task says "exactly decompose the matrix multiplication tensor".
        # So we MUST satisfy the 0 constraints.
        
        # Let's compute the full tensor.
        # T = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates a tensor of size (nm, mp, pn).
        # Then flatten and compare to target.
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        T_flat = T.flatten()
        
        # Target tensor
        target_flat = np.zeros(total_entries)
        for a, b, c in nonzero_indices:
            idx = a * mp * pn + b * pn + c
            target_flat[idx] = 1.0
            
        return T_flat - target_flat

    # Try ranks from 47 down to 40
    for R in range(47, 39, -1):
        start_time = time.time()
        deadline = start_time + 10 # 10s per rank attempt
        
        # Initial guess: Schoolbook scaled down?
        # We have 48 terms. We need R terms.
        # We can try to randomly combine terms.
        
        best_error = float('inf')
        best_params = None
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize U, V, W
            # Strategy: Start with schoolbook, then randomly zero out some columns?
            # No, we need to fit R columns.
            
            # Random initialization
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # Or, use a subset of schoolbook?
            # If R < 48, we can't just pick R columns.
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = least_squares(residuals, params, method='lm', max_nfev=1000)
                error = np.linalg.norm(res.fun)
                if error < best_error:
                    best_error = error
                    best_params = res.x
            except Exception:
                pass
        
        if best_error < 1e-6:
            # Extract U, V, W
            U = best_params[:R*nm].reshape(nm, R)
            V = best_params[R*nm:R*nm+R*mp].reshape(mp, R)
            W = best_params[R*nm+R*mp:].reshape(pn, R)
            
            # Convert to lists
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    R = 48
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
