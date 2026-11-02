"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
import random

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.
    Uses discrete local search on half-integers to find a rank < 60.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    # Allowed values for discrete search
    # We can expand this if needed, but {-1, -0.5, 0, 0.5, 1} is a good start
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    val_idx = {v: i for i, v in enumerate(values)}

    def compute_error(U, V, W):
        """
        Compute the sum of squared errors.
        U: (n, m, R)
        V: (m, p, R)
        W: (p, n, R)
        """
        # S_ijk = sum_r U_ijr V_jkr W_kir
        S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        err = np.sum((S - 1.0)**2)
        return err

    def local_search(U, V, W, max_iters, time_limit):
        """
        Perform discrete local search on U, V, W.
        Returns (U, V, W, error)
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        current_err = compute_error(U, V, W)
        
        if current_err < 1e-8:
            return U, V, W, current_err

        stuck_count = 0
        best_err = current_err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        for it in range(max_iters):
            if time.time() > time_limit:
                break

            # Randomly pick a matrix and an index
            mat_type = np.random.choice(3)
            if mat_type == 0:
                mat = U
                shape = U.shape
            elif mat_type == 1:
                mat = V
                shape = V.shape
            else:
                mat = W
                shape = W.shape

            idx = (np.random.randint(shape[0]), np.random.randint(shape[1]), np.random.randint(shape[2]))
            old_val = mat[idx]
            
            # Pick a new value from neighbors
            # Neighbors are values +/- 0.5
            new_val = values[np.random.randint(len(values))]
            
            if new_val == old_val:
                continue

            # Apply change
            mat[idx] = new_val
            
            # Compute new error
            new_err = compute_error(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
                stuck_count = 0
                if new_err < best_err:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                # Revert
                mat[idx] = old_val
                stuck_count += 1
                
                # If stuck too long, try a larger perturbation or restart
                if stuck_count > 500:
                    # Perturb a random few elements
                    for _ in range(10):
                        mt = np.random.choice(3)
                        if mt == 0:
                            mat = U; shp = U.shape
                        elif mt == 1:
                            mat = V; shp = V.shape
                        else:
                            mat = W; shp = W.shape
                        idx = (np.random.randint(shp[0]), np.random.randint(shp[1]), np.random.randint(shp[2]))
                        mat[idx] = values[np.random.randint(len(values))]
                    
                    current_err = compute_error(U, V, W)
                    if current_err < best_err:
                        best_err = current_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    stuck_count = 0

            if best_err < 1e-8:
                break

        return best_U, best_V, best_W, best_err

    # Search for lower ranks
    # We start from 59 and go down.
    # We use a time budget per rank.
    for rank in range(59, 20, -1):
        if time.time() > deadline - 10:
            break
        
        rank_time_budget = deadline - time.time()
        if rank_time_budget < 5:
            break
            
        # Allocate time based on rank
        # Lower ranks are harder, so give them more time if available
        # But we need to finish.
        # Let's try 2 trials per rank, with a max of 100k iterations each
        num_trials = 2
        max_iters = 50000
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(rank * 1000 + trial)
            
            # Initialize with small random values
            # Scale: 0.5 is a good magnitude
            U = np.random.choice(values, size=(n, m, rank))
            V = np.random.choice(values, size=(m, p, rank))
            W = np.random.choice(values, size=(p, n, rank))
            
            # Run local search
            time_limit = deadline - 5
            U_sol, V_sol, W_sol, err = local_search(U, V, W, max_iters, time_limit)
            
            if err < 1e-8:
                # Found a solution!
                # Convert to list of lists of floats (which are half-integers)
                U_list = [[float(U_sol[i, j, r]) for r in range(rank)] for i in range(n) for j in range(m)]
                V_list = [[float(V_sol[i, j, r]) for r in range(rank)] for i in range(m) for j in range(p)]
                W_list = [[float(W_sol[i, j, r]) for r in range(rank)] for i in range(p) for j in range(n)]
                
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we found a solution with error < 1e-4 but not < 1e-8, we might try to polish it
        # But for now, we just continue to next rank

    return best_result
# EVOLVE-BLOCK-END
