<<<<<<< SEARCH
    rank = len(U)
    return {"U": U, "V": V, "W": W}
=======
    import numpy as np
    
    # U, V, W are lists of columns (each column is a list of length n*m, m*p, p*n)
    # We need to return matrices of shape (n*m, rank), (m*p, rank), (p*n, rank)
    
    if U:
        U_np = np.array(U).T
        V_np = np.array(V).T
        W_np = np.array(W).T
    else:
        # Fallback if no terms were added (should not happen for n,m,p > 0)
        U_np = np.zeros((n*m, 1))
        V_np = np.zeros((m*p, 1))
        W_np = np.zeros((p*n, 1))
        
    return {"U": U_np, "V": V_np, "W": W_np}
>>>>>>> REPLACE
