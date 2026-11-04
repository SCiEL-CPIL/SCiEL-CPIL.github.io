import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin

    # Schoolbook solution
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Try to find a lower rank solution
    # We try ranks from 71 down to 60
    best_solution = None
    
    for r in range(rank_school - 1, max(rank_school - 15, 0), -1):
        if time.time() > deadline:
            break
            
        # Initialize with random or schoolbook subset
        # Using schoolbook subset might be a good start
        # We take the first r columns of the schoolbook solution
        U_init = U_school[:, :r].copy()
        V_init = V_school[:, :r].copy()
        W_init = W_school[:, :r].copy()
        
        # Flatten
        def flatten(x):
            return np.concatenate([x.ravel() for x in x])
        
        def unflatten(x, shapes):
            mats = []
            idx = 0
            for shape in shapes:
                size = np.prod(shape)
                mats.append(x[idx:idx+size].reshape(shape))
                idx += size
            return mats

        shapes = [(n*m, r), (m*p, r), (p*n, r)]
        x0 = flatten([U_init, V_init, W_init])
        
        def residuals(x):
            U, V, W = unflatten(x, shapes)
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        res.append(val - 1.0)
            return np.array(res)
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if time.time() > deadline:
                continue
                
            if sol.cost < 1e-10:
                U, V, W = unflatten(sol.x, shapes)
                # Try to round to integers or half-integers
                # Check if entries are close to integers
                if np.allclose(U, np.round(U), atol=1e-5) and \
                   np.allclose(V, np.round(V), atol=1e-5) and \
                   np.allclose(W, np.round(W), atol=1e-5):
                    U_int = np.round(U).astype(int)
                    V_int = np.round(V).astype(int)
                    W_int = np.round(W).astype(int)
                    
                    # Verify
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_int[i*m+j], V_int[j*p+k] * W_int[k*n+i])
                                if abs(val - 1.0) > 1e-5:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if valid:
                        best_solution = (U_int, V_int, W_int)
                        break # Found a valid lower rank
        except Exception:
            pass

    if best_solution:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
