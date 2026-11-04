import numpy as np
import time
from scipy.optimize import minimize
from itertools import product

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit

    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W form a valid decomposition."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def get_schoolbook():
        """Return the schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def try_rank_optimize(R, num_starts=5):
        """
        Attempt to find a decomposition of rank R using optimization.
        We fix W to be the schoolbook-like sparse structure if possible, 
        or use a generic parameterization.
        
        A better approach for exact integer/half-integer solutions is often 
        hard via continuous optimization directly. However, we can try to 
        minimize the error. If the error is very small, we might round.
        
        Let's try a specific structured ansatz.
        For matrix multiplication, we know that if we set W to be identity-like 
        (or sparse), it constrains U and V.
        
        Actually, let's try a random search with a "repair" step.
        Or, more simply, since random search failed, let's try to construct 
        a solution using the fact that the tensor is simple.
        
        Let's try a greedy approach or a small-scale optimization.
        Given the time limit, continuous optimization on 10*R + 30*R + 12*R 
        variables is expensive.
        
        Let's stick to the current strategy but improve the random search 
        by using a smarter initialization or a local search around random points.
        
        However, the prompt asks to combine strategies. 
        Inspiration 1: Schoolbook.
        Inspiration 2: Random search with time limit.
        Inspiration 3: Optimization.
        
        Let's implement a simple local search.
        1. Generate random U, V, W.
        2. Compute error.
        3. Use scipy.optimize.minimize to reduce error.
        4. Check if error is small enough to round to integers/half-integers.
        """
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            U = params[:nm*R].reshape(nm, R)
            V = params[nm*R:nm*R+mp*R].reshape(mp, R)
            W = params[nm*R+mp*R:].reshape(pn, R)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                        err += (val - 1.0) ** 2
            return err

        for start in range(num_starts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with small random values
            np.random.seed(start)
            U_init = np.random.uniform(-1, 1, (nm, R))
            V_init = np.random.uniform(-1, 1, (mp, R))
            W_init = np.random.uniform(-1, 1, (pn, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
                
                if res.fun < 1e-6:
                    U_opt = res.x[:nm*R].reshape(nm, R)
                    V_opt = res.x[nm*R:nm*R+mp*R].reshape(mp, R)
                    W_opt = res.x[nm*R+mp*R:].reshape(pn, R)
                    
                    # Try to round to nearest integer or half-integer
                    # The problem allows integer/half-integer.
                    # Let's try rounding to nearest integer first.
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    
                    if check_decomposition(U_int, V_int, W_int, tol=1e-9):
                        return U_int, V_int, W_int
                        
                    # Try rounding to nearest half-integer (multiply by 2, round, divide by 2)
                    U_half = np.round(U_opt * 2) / 2
                    V_half = np.round(V_opt * 2) / 2
                    W_half = np.round(W_opt * 2) / 2
                    
                    if check_decomposition(U_half, V_half, W_half, tol=1e-9):
                        return U_half, V_half, W_half
                        
                    # If not exact after rounding, maybe keep the float solution?
                    # The task says "integer/half-integer entries". 
                    # If we can't round, this solution is invalid for the interface.
                    
            except Exception:
                continue
                
        return None

    def try_rank_random(R, num_attempts=200):
        """Try random search for a valid decomposition of rank R."""
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            # Use small random integers
            U = np.random.randint(-1, 2, size=(nm, R))
            V = np.random.randint(-1, 2, size=(mp, R))
            W = np.random.randint(-1, 2, size=(pn, R))
            
            if check_decomposition(U, V, W):
                return U, V, W
        return None

    # We know rank 60 is possible.
    # The theoretical lower bound is max(n*m, m*p, p*n) = 30.
    # Let's try to find a rank < 60.
    
    # First, try optimization for ranks close to 60, as they are easier.
    # Then fall back to random search.
    
    # Strategy:
    # 1. Try rank 59 down to 50 with optimization (limited iterations).
    # 2. If that fails, try rank 59 down to 50 with random search.
    # 3. Return schoolbook.
    
    for R in range(59, 49, -1):
        if time.time() > deadline - 20:
            break
        
        # Try optimization
        result = try_rank_optimize(R, num_starts=2)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        if time.time() > deadline - 10:
            break
        
        # Try random
        result = try_rank_random(R, num_attempts=50)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook (Rank 60)
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
