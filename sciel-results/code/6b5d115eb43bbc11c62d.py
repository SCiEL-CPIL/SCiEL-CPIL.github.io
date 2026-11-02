import numpy as np
import random
import time
import scipy.optimize

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def verify_decomposition(U, V, W, rank):
        # U: (n, m, R), V: (m, p, R), W: (p, n, R)
        # Check sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1
        S = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        if not np.allclose(S, 1.0, atol=1e-9):
            return False
        return True

    def to_interface(U, V, W, R):
        # U is (n, m, R) -> (n*m, R)
        # V is (m, p, R) -> (m*p, R)
        # W is (p, n, R) -> (p*n, R)
        # Interface W index is k*n+i.
        # Our W is (p, n, R) with indices [k, i, c].
        # Flatten W to (p*n, R) with row index k*n+i.
        # This is exactly W.reshape(p*n, R) if W is C-contiguous and indexed [k, i].
        
        U_flat = U.reshape(n*m, R)
        V_flat = V.reshape(m*p, R)
        W_flat = W.reshape(p*n, R)
        
        # Convert to list of lists of ints/floats
        def mat_to_list(M):
            res = []
            for row in M:
                r = []
                for val in row:
                    nearest = round(val * 2) / 2
                    if nearest == int(nearest):
                        r.append(int(nearest))
                    else:
                        r.append(float(nearest))
                res.append(r)
            return res
            
        return mat_to_list(U_flat), mat_to_list(V_flat), mat_to_list(W_flat)

    start_time = time.time()
    time_limit = 110
    
    # Try ranks from lower to higher
    # Lower bound is at least n*m*p / something? 
    # Let's try 25 to 59.
    for R in range(25, 60):
        if time.time() - start_time > time_limit:
            break
            
        # Number of restarts
        num_restarts = 100
        
        for r_idx in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U, V, W
            # Shape: U (n, m, R), V (m, p, R), W (p, n, R)
            U_init = np.random.randn(n, m, R) * 0.5
            V_init = np.random.randn(m, p, R) * 0.5
            W_init = np.random.randn(p, n, R) * 0.5
            
            # Flatten
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U = x[:n*m*R].reshape(n, m, R)
                V = x[n*m*R:n*m*R + m*p*R].reshape(m, p, R)
                W = x[n*m*R + m*p*R:].reshape(p, n, R)
                
                S = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                err = S - 1.0
                return np.sum(err**2)
            
            try:
                res = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12})
            except:
                continue
                
            if res.fun < 1e-6:
                # Extract solution
                x_sol = res.x
                U_sol = x_sol[:n*m*R].reshape(n, m, R)
                V_sol = x_sol[n*m*R:n*m*R + m*p*R].reshape(m, p, R)
                W_sol = x_sol[n*m*R + m*p*R:].reshape(p, n, R)
                
                # Round to half-integers
                U_round = np.round(U_sol * 2) / 2
                V_round = np.round(V_sol * 2) / 2
                W_round = np.round(W_sol * 2) / 2
                
                # Verify
                if verify_decomposition(U_round, V_round, W_round, R):
                    U_out, V_out, W_out = to_interface(U_round, V_round, W_round, R)
                    return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback to schoolbook
    U_out, V_out, W_out = to_interface(U_school, V_school, W_school, rank_school)
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
