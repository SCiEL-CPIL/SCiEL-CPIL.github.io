import numpy as np
from scipy.optimize import fsolve
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Target rank to try
    # We want to find a rank < 36.
    # Let's try ranks 30 down to 25.
    
    # Define the tensor equations
    # Sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] = 1
    # for all i in [0, n), j in [0, m), k in [0, p)
    
    def residual(params, r):
        # Unpack params into U, V, W
        # U: n*m*r
        # V: m*p*r
        # W: p*n*r
        # We fix W[0, c] = 1 for all c to remove scaling invariance?
        # Actually, let's just solve for all variables and check scaling later?
        # No, fixing is better for convergence.
        # Let's fix W[0, c] = 1.
        # Indices for W: k*n + i.
        # W[0, c] corresponds to k=0, i=0.
        
        u_size = n * m * r
        v_size = m * p * r
        w_size = p * n * r
        
        U = params[:u_size].reshape(n, m, r)
        V = params[u_size:u_size + v_size].reshape(m, p, r)
        W = params[u_size + v_size:].reshape(p, n, r)
        
        # Apply constraints: W[0, c] = 1
        # This means we should not include W[0, c] in params?
        # Or just set them to 1 in the residual calculation?
        # If we set them to 1, the gradient wrt those params is 0, which is bad.
        # Better to remove them from the variable vector.
        
        # Let's re-structure:
        # Variables: U (all), V (all), W (all except W[0, c])
        # This is complex to implement dynamically in a simple loop.
        # Alternative: Solve with all variables, but add penalty (x_W0 - 1)^2?
        # Or just use all variables and hope fsolve finds a solution where W[0,c] != 0,
        # then we can scale.
        
        # Let's try solving with all variables first.
        
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over c
                    val = 0
                    for c in range(r):
                        val += U[i, j, c] * V[j, k, c] * W[k, i, c]
                    res[idx] = val - 1
                    idx += 1
        return res

    best_solution = None
    best_rank = rank_schoolbook
    
    # Time guard
    start_time = time.time()
    deadline = 140.0
    
    for r in range(30, 19, -1): # Try 30 down to 20
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        found = False
        for trial in range(5):
            if time.time() > deadline:
                break
                
            # Initialize
            # Random integers or small floats
            U_init = np.random.randint(-2, 3, (n, m, r)).astype(float)
            V_init = np.random.randint(-2, 3, (m, p, r)).astype(float)
            W_init = np.random.randint(-2, 3, (p, n, r)).astype(float)
            
            # Ensure W[0, c] is not 0 to allow scaling later if needed
            # Or just let fsolve do its thing.
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Solve
            try:
                sol, info, ier, msg = fsolve(residual, params_init, args=(r,), full_output=True)
                if ier == 1:
                    # Check residual
                    res_val = residual(sol, r)
                    if np.max(np.abs(res_val)) < 1e-5:
                        # Solution found!
                        # Round to nearest 0.5
                        U_sol = sol[:n*m*r].reshape(n, m, r)
                        V_sol = sol[n*m*r:n*m*r + m*p*r].reshape(m, p, r)
                        W_sol = sol[n*m*r + m*p*r:].reshape(p, n, r)
                        
                        U_round = np.round(U_sol * 2) / 2
                        V_round = np.round(V_sol * 2) / 2
                        W_round = np.round(W_sol * 2) / 2
                        
                        # Verify exactness
                        res_round = residual(np.concatenate([U_round.flatten(), V_round.flatten(), W_round.flatten()]), r)
                        if np.max(np.abs(res_round)) < 1e-9:
                            # Convert to lists
                            U_list = U_round.tolist()
                            V_list = V_round.tolist()
                            W_list = W_round.tolist()
                            
                            # Flatten as required by interface
                            # U: (n*m, r)
                            U_flat = [U_list[i*m + j][c] for i in range(n) for j in range(m) for c in range(r)]
                            # Wait, the interface expects list of lists.
                            # U[i*m + j][c]
                            U_out = [[0]*r for _ in range(n*m)]
                            for i in range(n):
                                for j in range(m):
                                    for c in range(r):
                                        U_out[i*m + j][c] = U_round[i, j, c]
                                        
                            V_out = [[0]*r for _ in range(m*p)]
                            for j in range(m):
                                for k in range(p):
                                    for c in range(r):
                                        V_out[j*p + k][c] = V_round[j, k, c]
                                        
                            W_out = [[0]*r for _ in range(p*n)]
                            for k in range(p):
                                for i in range(n):
                                    for c in range(r):
                                        W_out[k*n + i][c] = W_round[k, i, c]
                                        
                            return {"U": U_out, "V": V_out, "W": W_out}
            except Exception as e:
                pass

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
