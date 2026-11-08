import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Try to find rank 17
    R = 17
    # Use a deterministic or fast randomized search
    # Given the small size, we can try a grid search on a subset of parameters
    # Or use a pre-computed solution if available
    
    # Since we don't have a pre-computed solution, we use a heuristic search
    # We'll try to minimize the error using a few restarts
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    time_limit = 100  # Leave margin
    
    for restart in range(200):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with small random half-integers
        U = np.random.randint(-4, 5, (n*m, R)) * 0.5
        V = np.random.randint(-4, 5, (m*p, R)) * 0.5
        W = np.random.randint(-4, 5, (p*n, R)) * 0.5
        
        # Local search
        for step in range(50):
            # Compute error
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.sum((T_calc - T)**2)
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
                if error < 1e-10:
                    # Snap and return
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    T_check = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                    if np.allclose(T_check, T, atol=1e-9):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
            
            # Update: perturb one random entry
            mat = np.random.choice([U, V, W])
            r = np.random.randint(0, mat.shape[0])
            c = np.random.randint(0, mat.shape[1])
            old_val = mat[r, c]
            mat[r, c] += np.random.choice([-0.5, 0.5])
            
            # If error increases too much, revert (simple hill climbing)
            T_calc_new = np.einsum('ar,br,cr->abc', U, V, W)
            error_new = np.sum((T_calc_new - T)**2)
            if error_new > error + 1e-5:
                mat[r, c] = old_val

    # Fallback to schoolbook
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
