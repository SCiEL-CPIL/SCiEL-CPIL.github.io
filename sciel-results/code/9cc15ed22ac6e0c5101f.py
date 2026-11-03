import numpy as np
import random
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Lower bound is ceil((nm + mp + pn)/2) = ceil((9+12+12)/2) = 17
    min_rank = 17
    max_rank = n * m * p
    
    # Helper to verify a solution
    def verify(U, V, W, r):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Helper to check if values are close to half-integers
    def is_half_integer(arr, tol=1e-6):
        doubled = 2 * arr
        rounded = np.round(doubled)
        if np.allclose(doubled, rounded, atol=tol):
            return rounded / 2.0
        return None

    # Try different ranks
    for r in range(min_rank, max_rank + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Try several random initializations
        for attempt in range(200):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with sparse random integers
            # Use {-1, 0, 1}
            V = np.random.choice([-1, 0, 1], size=(m * p, r)).astype(float)
            W = np.random.choice([-1, 0, 1], size=(p * n, r)).astype(float)
            
            # Ensure non-zero
            if np.all(V == 0): V[0,0] = 1
            if np.all(W == 0): W[0,0] = 1
            
            U = np.zeros((n * m, r))
            valid = True
            
            # Iterate ALS steps
            for step in range(50):
                if time.time() - start_time > time_limit:
                    valid = False
                    break
                    
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        A_ij = np.zeros((p, r))
                        for k in range(p):
                            A_ij[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        
                        b = np.ones(p)
                        try:
                            # Check if A_ij has full column rank? No, r can be > p.
                            # We want a solution u such that A_ij u = b.
                            # This is an underdetermined system if r > p.
                            # We want a simple solution.
                            # Use lstsq to get minimum norm solution.
                            u, residuals, rank, s = np.linalg.lstsq(A_ij, b, rcond=None)
                            if np.allclose(A_ij @ u, b, atol=1e-9):
                                U[i * m + j, :] = u
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
                
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        B_jk = np.zeros((n, r))
                        for i in range(n):
                            B_jk[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        
                        b = np.ones(n)
                        try:
                            v, residuals, rank, s = np.linalg.lstsq(B_jk, b, rcond=None)
                            if np.allclose(B_jk @ v, b, atol=1e-9):
                                V[j * p + k, :] = v
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
                
                # Solve for W
                for i in range(n):
                    for k in range(p):
                        C_ik = np.zeros((m, r))
                        for j in range(m):
                            C_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        
                        b = np.ones(m)
                        try:
                            w, residuals, rank, s = np.linalg.lstsq(C_ik, b, rcond=None)
                            if np.allclose(C_ik @ w, b, atol=1e-9):
                                W[k * n + i, :] = w
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                # Check if we can round to half-integers
                U_rounded = is_half_integer(U)
                V_rounded = is_half_integer(V)
                W_rounded = is_half_integer(W)
                
                if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                    # Verify the rounded solution
                    if verify(U_rounded, V_rounded, W_rounded, r):
                        # Convert to lists of lists for the interface
                        U_list = U_rounded.tolist()
                        V_list = V_rounded.tolist()
                        W_list = W_rounded.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
        
    # Fallback: Schoolbook algorithm
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

# EVOLVE-BLOCK-END
