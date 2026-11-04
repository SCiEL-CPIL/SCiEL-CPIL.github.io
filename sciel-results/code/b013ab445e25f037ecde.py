import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 5
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = nm * p  # 45
    
    # 1. Construct Schoolbook Decomposition (Rank 45)
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, idx] = 1
                V_school[j*p+k, idx] = 1
                W_school[k*n+i, idx] = 1
                idx += 1
                
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    best_rank = schoolbook_rank
    
    # 2. Attempt to find lower rank (44, 43, ...)
    # We use a time-boxed search.
    start_time = time.time()
    time_limit = 100  # Leave margin for 120s limit
    
    def verify(U, V, W, R):
        # Check if U, V, W of rank R satisfy the tensor equation
        # And if entries are integer/half-integer
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(R):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        # Check integer/half-integer constraint
        for matrix in [U, V, W]:
            for val in matrix:
                if abs(val * 2 - round(val * 2)) > 1e-6:
                    return False
        return True

    def objective(x, R):
        # x is flattened U, V, W
        U = x[:nm*R].reshape(nm, R)
        V = x[nm*R:nm*R+mp*R].reshape(mp, R)
        W = x[nm*R+mp*R:].reshape(pn, R)
        
        # Residual sum of squares
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    res += (val - 1.0)**2
        return res

    # Try ranks 44 down to 40
    for target_rank in range(schoolbook_rank - 1, 39, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with schoolbook truncated? No, random is better for escape.
        # But schoolbook truncated might be closer if the lower rank is a subset.
        # Let's try 3 restarts.
        for restart in range(3):
            if time.time() - start_time > time_limit:
                break
                
            rng = np.random.default_rng(restart + target_rank)
            # Initialize with small noise around 0
            U0 = rng.normal(0, 0.5, (nm, target_rank))
            V0 = rng.normal(0, 0.5, (mp, target_rank))
            W0 = rng.normal(0, 0.5, (pn, target_rank))
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                # Use Powell's method, which doesn't require gradients and handles non-convexity better than LM for discrete-like targets
                res = minimize(objective, x0, args=(target_rank,), method='Powell', 
                               options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
                
                if res.fun < 1e-6:
                    U_sol = res.x[:nm*target_rank].reshape(nm, target_rank)
                    V_sol = res.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                    W_sol = res.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                    
                    # Round to half-integers
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    if verify(U_rounded, V_rounded, W_rounded, target_rank):
                        best_U = U_rounded.tolist()
                        best_V = V_rounded.tolist()
                        best_W = W_rounded.tolist()
                        best_rank = target_rank
                        break # Found lower rank, try lower still?
                        
            except Exception:
                pass
        
        if best_rank < schoolbook_rank:
            # We found a valid lower rank. Continue to see if we can go lower.
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
