import time
import numpy as np
from scipy.optimize import fsolve

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 5 matrix.
    
    The schoolbook decomposition has rank n*m*p = 2*3*5 = 30.
    We attempt to find a rank 29 decomposition by solving the system of
    trilinear equations using scipy.optimize.fsolve with a warm start
    from the schoolbook solution.
    """
    start_time = time.time()
    deadline = 120.0
    time_limit = 100.0  # Safety margin
    
    n, m, p = DIMENSIONS
    target_rank = n * m * p - 1  # Try rank 29
    
    # Build the list of all (a,b,c,d,i,k) tuples
    all_indices = []
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            all_indices.append((a, b, c, d, i, k))
    
    num_constraints = len(all_indices)  # 900
    
    def residuals(x):
        """
        x is a flattened vector of length rank*(n*m + m*p + p*n)
        First rank*(n*m) elements are U (flattened row-major)
        Next rank*(m*p) elements are V
        Last rank*(p*n) elements are W
        """
        U_flat = x[:target_rank * n * m].reshape(n * m, target_rank)
        V_flat = x[target_rank * n * m : target_rank * n * m + target_rank * m * p].reshape(m * p, target_rank)
        W_flat = x[target_rank * n * m + target_rank * m * p :].reshape(p * n, target_rank)
        
        res = np.zeros(num_constraints)
        idx = 0
        for (a, b, c, d, i, k) in all_indices:
            s = 0.0
            for mu in range(target_rank):
                s += U_flat[a * m + b, mu] * V_flat[c * p + d, mu] * W_flat[k * n + i, mu]
            if a == i and b == c and d == k:
                res[idx] = s - 1.0
            else:
                res[idx] = s
            idx += 1
        return res
    
    # Create initial guess: take schoolbook rank 30, remove one column,
    # and slightly perturb to help convergence
    def make_initial_guess():
        # Schoolbook rank 30
        U30 = np.zeros((n * m, n * m * p))
        V30 = np.zeros((m * p, n * m * p))
        W30 = np.zeros((p * n, n * m * p))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U30[i * m + j, col] = 1.0
                    V30[j * p + k, col] = 1.0
                    W30[k * n + i, col] = 1.0
                    col += 1
        
        # Try removing column 0 and see if we can solve for rank 29
        # Use columns 1..29 as initial guess, set column 0 to small random values
        U_init = np.zeros((n * m, target_rank))
        V_init = np.zeros((m * p, target_rank))
        W_init = np.zeros((p * n, target_rank))
        
        # Use schoolbook columns 1-29 as the first 29 columns
        for mu in range(1, n * m * p):
            U_init[:, mu - 1] = U30[:, mu]
            V_init[:, mu - 1] = V30[:, mu]
            W_init[:, mu - 1] = W30[:, mu]
        
        # Add small perturbation to help escape degeneracy
        np.random.seed(42)
        U_init += 0.01 * np.random.randn(n * m, target_rank)
        V_init += 0.01 * np.random.randn(m * p, target_rank)
        W_init += 0.01 * np.random.randn(p * n, target_rank)
        
        # Flatten
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        return x0
    
    x0 = make_initial_guess()
    
    # Try to solve using fsolve
    solution = None
    try:
        # Use a tighter tolerance and more iterations
        sol, info, ier, msg = fsolve(residuals, x0, full_output=True, maxfev=5000, xtol=1e-8)
        
        if ier == 1:
            # Check if solution is good
            res = residuals(sol)
            max_res = np.max(np.abs(res))
            if max_res < 1e-6:
                solution = sol
                print(f"Found rank {target_rank} solution with max residual {max_res:.2e}")
    except Exception as e:
        print(f"fsolve failed: {e}")
    
    # If fsolve didn't work, try a few random restarts
    if solution is None and time.time() - start_time < time_limit:
        for attempt in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            
            np.random.seed(42 + attempt * 7)
            U_init = np.zeros((n * m, target_rank))
            V_init = np.zeros((m * p, target_rank))
            W_init = np.zeros((p * n, target_rank))
            
            # Random initialization with small values
            U_init = 0.1 * np.random.randn(n * m, target_rank)
            V_init = 0.1 * np.random.randn(m * p, target_rank)
            W_init = 0.1 * np.random.randn(p * n, target_rank)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                sol, info, ier, msg = fsolve(residuals, x0, full_output=True, maxfev=3000, xtol=1e-8)
                if ier == 1:
                    res = residuals(sol)
                    max_res = np.max(np.abs(res))
                    if max_res < 1e-6:
                        solution = sol
                        print(f"Found rank {target_rank} solution (attempt {attempt}) with max residual {max_res:.2e}")
                        break
            except Exception as e:
                continue
    
    # If we found a solution, extract it
    if solution is not None:
        U = solution[:target_rank * n * m].reshape(n * m, target_rank)
        V = solution[target_rank * n * m : target_rank * n * m + target_rank * m * p].reshape(m * p, target_rank)
        W = solution[target_rank * n * m + target_rank * m * p :].reshape(p * n, target_rank)
        
        # Round to nearest half-integer if close
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Check if rounded version is still valid
        def check_rounded(Ur, Vr, Wr):
            for (a, b, c, d, i, k) in all_indices:
                s = 0.0
                for mu in range(target_rank):
                    s += Ur[a * m + b, mu] * Vr[c * p + d, mu] * Wr[k * n + i, mu]
                if a == i and b == c and d == k:
                    if abs(s - 1.0) > 1e-4:
                        return False
                else:
                    if abs(s) > 1e-4:
                        return False
            return True
        
        if check_rounded(U_rounded, V_rounded, W_rounded):
            U, V, W = U_rounded, V_rounded, W_rounded
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return schoolbook rank 30
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
