"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.
    
    We search for the minimum rank R such that there exist matrices
    U (15 x R), V (25 x R), W (15 x R) with the property that:
    sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*3+i, r] = 1 for all i,j,k.
    """
    n, m, p = DIMENSIONS
    
    # Try ranks from 19 upward (known lower bound is 25, but let's check if we can beat it)
    # Actually, the trivial lower bound is max(n*m, m*p, n*p) = 25.
    # But there might be a better lower bound. Let's just try to find the minimum.
    
    # For each candidate rank R, try to find a solution.
    # We'll use an iterative approach: fix V, solve for U and W, then update V.
    
    best_solution = None
    
    for R in range(19, 76):  # Try from 19 to 75
        # Try multiple random initializations
        found = False
        for trial in range(50):
            np.random.seed(trial * 1000 + R)
            
            # Initialize V randomly
            V = np.random.randn(m * p, R)
            
            # Iterate: given V, solve for U and W
            # The equation: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # For fixed V, this is bilinear in U and W.
            
            # Let's fix W and solve for U, then fix U and solve for W.
            
            W = np.random.randn(p * n, R)
            
            for it in range(100):
                # Solve for U given V and W
                # For each (i,j): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # This is: for each (i,j), sum_k [sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]] = ... 
                # Actually, for each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # Fixing V and W, for each (i,j), we have p equations in R unknowns (U[i*m+j, :])
                # But these equations are coupled through the sum over r.
                
                # Let me set up the linear system properly.
                # Unknowns: U (15*R unknowns)
                # For each (i,j,k): sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
                # Let C_r[i,j,k] = V[j*p+k,r] * W[k*n+i,r]
                # Then: sum_r U[i*m+j,r] * C_r[i,j,k] = 1 for all i,j,k
                
                # For fixed (i,j): sum_r U[i*m+j,r] * C_r[i,j,k] = 1 for all k
                # This is a system of p equations in R unknowns.
                
                # Build the system:
                # A_U * U_vec = b_U
                # where A_U has 75 rows and 15*R columns
                
                A_U = np.zeros((n * m * p, n * m * R))
                b_U = np.ones(n * m * p)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a_idx = i * m + j
                            for r in range(R):
                                A_U[eq_idx, a_idx * R + r] = V[j * p + k, r] * W[k * n + i, r]
                            eq_idx += 1
                
                # Solve least squares
                U, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                U = U.reshape(n * m, R)
                
                # Now solve for W given U and V
                A_W = np.zeros((n * m * p, p * n * R))
                b_W = np.ones(n * m * p)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            c_idx = k * n + i
                            for r in range(R):
                                A_W[eq_idx, c_idx * R + r] = U[i * m + j, r] * V[j * p + k, r]
                            eq_idx += 1
                
                W, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                W = W.reshape(p * n, R)
                
                # Check if solution is valid
                valid = True
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a_idx = i * m + j
                            b_idx = j * p + k
                            c_idx = k * n + i
                            s = 0.0
                            for r in range(R):
                                s += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                            err = abs(s - 1.0)
                            max_err = max(max_err, err)
                            if err > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    found = True
                    break
            
            if found:
                best_solution = (R, U, V, W)
                break
        
        if found:
            break
    
    if best_solution:
        R, U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 25 construction
    R = 25
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    for idx in range(m*p):
        V[idx, idx] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                U[a_idx, b_idx] = 1
                W[c_idx, b_idx] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

