"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # 1. Construct Schoolbook Solution (Rank 50)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=float)
    V_school = np.zeros((mp, rank_school), dtype=float)
    W_school = np.zeros((pn, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    best_result = {
        "U": U_school.tolist(), 
        "V": V_school.tolist(), 
        "W": W_school.tolist()
    }
    best_rank = rank_school

    # 2. Heuristic search for lower rank
    start_time = time.time()
    deadline = start_time + 100

    def is_integer_or_half(x):
        return np.allclose(x, np.round(2*x)/2)

    # Try to find a lower rank decomposition
    # We search for R from 49 down to 1
    # For each R, we try random sparse U, V and solve for W
    for R in range(49, 10, -1):
        if time.time() > deadline:
            break
            
        # Try a few random instances
        for trial in range(50):
            if time.time() > deadline:
                break
                
            # Generate random sparse U, V
            # Density: ~10% non-zeros
            U_rand = np.zeros((nm, R))
            V_rand = np.zeros((mp, R))
            
            # Randomly set entries to 1 or -1
            # We ensure each column has at least a few non-zeros
            for r in range(R):
                # Pick random indices for U
                num_u = np.random.randint(2, 6)
                idx_u = np.random.choice(nm, size=num_u, replace=False)
                U_rand[idx_u, r] = np.random.choice([-1, 1], size=num_u)
                
                # Pick random indices for V
                num_v = np.random.randint(2, 6)
                idx_v = np.random.choice(mp, size=num_v, replace=False)
                V_rand[idx_v, r] = np.random.choice([-1, 1], size=num_v)
            
            # Check if this U, V allows a solution for W
            valid = True
            W_sol = np.zeros((pn, R))
            
            for i in range(n):
                for k in range(p):
                    # Construct matrix Z (m x R)
                    # Z[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    Z = np.zeros((m, R))
                    for j in range(m):
                        Z[j, :] = U_rand[i*m+j, :] * V_rand[j*p+k, :]
                    
                    # Solve Z @ w = 1 (vector of ones)
                    try:
                        w, residuals, rank_z, s = np.linalg.lstsq(Z, np.ones(m), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if w is close to integers/half-integers
                    if not is_integer_or_half(w):
                        valid = False
                        break
                    else:
                        W_sol[k*n+i, :] = w
            
            if valid:
                # Verify
                U_check = np.round(U_rand * 2) / 2
                V_check = np.round(V_rand * 2) / 2
                W_check = np.round(W_sol * 2) / 2
                
                # Verify tensor identity
                success = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += U_check[i*m+j, r] * V_check[j*p+k, r] * W_check[k*n+i, r]
                            if not np.isclose(s, 1.0):
                                success = False
                                break
                        if not success:
                            break
                    if not success:
                        break
                
                if success:
                    best_result = {
                        "U": U_check.tolist(), 
                        "V": V_check.tolist(), 
                        "W": W_check.tolist()
                    }
                    best_rank = R
                    if R < 20:
                        break

    return best_result
# EVOLVE-BLOCK-END
