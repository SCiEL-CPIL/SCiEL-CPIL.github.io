import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    # Target tensor T is all 1s for indices i in [0,n), j in [0,m), k in [0,p)
    # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
    
    # Precompute indices for the 40 non-zero entries
    # i, j, k ranges
    # U shape: (n*m, R) -> indices i*m + j
    # V shape: (m*p, R) -> indices j*p + k
    # W shape: (p*n, R) -> indices k*n + i
    
    # We will search for R from a low bound to 39
    # Lower bound heuristic: 
    # For 2x2x2, rank 7. 
    # Scaling suggests R ~ (n*m*p) * (7/8) = 40 * 0.875 = 35.
    # Let's start from 25 to be safe and efficient, or maybe 20.
    # If we find a solution at 20, great. If not, go up.
    
    best_solution = None
    best_rank = 40 # Schoolbook
    
    # Generate schoolbook solution as fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    # Try to find a better rank
    for R in range(20, 40):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this R
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale matters. Let's try values around 1.
            np.random.seed(42 + R + restart)
            U_init = np.random.randn(n * m, R)
            V_init = np.random.randn(m * p, R)
            W_init = np.random.randn(p * n, R)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U = x[:n*m*R].reshape(n*m, R)
                V = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W = x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Compute tensor contraction
                # Reshape to (n, m, R), (m, p, R), (p, n, R)
                U_r = U.reshape(n, m, R)
                V_r = V.reshape(m, p, R)
                W_r = W.reshape(p, n, R)
                
                # C[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # Note: W is indexed by k*n + i, so W_r[k, i, r]
                C = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                
                # Target is all 1s
                error = np.sum((C - 1.0)**2)
                return error

            # Use L-BFGS-B
            try:
                res = opt.minimize(objective, x0, method='L-BFGS-B', 
                                   options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6})
                
                if res.fun < 1e-6:
                    # Extract solution
                    U_sol = res.x[:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    # Round to half-integers
                    # Multiply by 2, round to int, divide by 2
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    # Verify exactness
                    U_r = U_rounded.reshape(n, m, R)
                    V_r = V_rounded.reshape(m, p, R)
                    W_r = W_rounded.reshape(p, n, R)
                    C_check = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                    
                    if np.allclose(C_check, 1.0, atol=1e-9):
                        # Convert to lists of ints/floats as required
                        # The interface expects integer/half-integer entries.
                        # We can return floats if they are .0 or .5
                        U_list = U_rounded.tolist()
                        V_list = V_rounded.tolist()
                        W_list = W_rounded.tolist()
                        
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception as e:
                pass

    # If no better solution found, return schoolbook
    U, V, W, rank = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
