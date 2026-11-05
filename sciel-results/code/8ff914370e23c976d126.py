import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    def get_tensor_error(U, V, W, n, m, p, rank):
        """Compute tensor error using vectorized operations."""
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        # S[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        # Vectorize: for each t, compute outer product contribution
        S = np.zeros((n, m, p))
        for t in range(rank):
            Ut = U3[:, :, t]  # (n, m)
            Vt = V3[:, :, t]  # (m, p)
            Wt = W3[:, :, t]  # (p, n)
            # Contribution: U[i,j] * V[j,k] * W[k,i] for all i,j,k
            # = np.einsum('ij,jk,ki->ijk', Ut, Vt, Wt)
            S += np.einsum('ij,jk,ki->ijk', Ut, Vt, Wt)
        err = np.sum((S - 1.0)**2)
        return err, S

    def verify(U, V, W, n, m, p, rank, tol=1e-8):
        err, S = get_tensor_error(U, V, W, n, m, p, rank)
        return err < tol

    def try_rank_als(rank, seed, max_iter=300):
        """Improved ALS with vectorized solves and better convergence."""
        if time.time() > deadline - 15:
            return None
            
        rng = np.random.RandomState(seed)
        
        # Better initialization: use structured random values
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        best_err = np.inf
        best_state = None
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            # Solve for U: For each (i,j), solve A @ U[i,j,:] = ones(p)
            # where A[k,t] = V[j,k,t] * W[k,i,t]
            for i in range(n):
                for j in range(m):
                    # A has shape (p, rank), A[k,t] = V3[j,k,t] * W3[k,i,t]
                    A = V3[j, :, :] * W3[:, i, :]  # (p, rank)
                    b = np.ones(p)
                    try:
                        U3[i, j, :] = np.linalg.lstsq(A, b, rcond=1e-12)[0]
                    except:
                        pass
            
            U = U3.reshape(n*m, rank)
            
            # Solve for V: For each (j,k), solve A @ V[j,k,:] = ones(n)
            # where A[i,t] = U[i,j,t] * W[k,i,t]
            V3 = V.reshape(m, p, rank)
            for j in range(m):
                for k in range(p):
                    A = U3[:, j, :] * W3[k, :, :]  # (n, rank)
                    b = np.ones(n)
                    try:
                        V3[j, k, :] = np.linalg.lstsq(A, b, rcond=1e-12)[0]
                    except:
                        pass
            
            V = V3.reshape(m*p, rank)
            
            # Solve for W: For each (k,i), solve A @ W[k,i,:] = ones(m)
            # where A[j,t] = U[i,j,t] * V[j,k,t]
            W3 = W.reshape(p, n, rank)
            for k in range(p):
                for i in range(n):
                    A = U3[i, :, :] * V3[:, k, :]  # (m, rank)
                    b = np.ones(m)
                    try:
                        W3[k, i, :] = np.linalg.lstsq(A, b, rcond=1e-12)[0]
                    except:
                        pass
            
            W = W3.reshape(p*n, rank)
            
            # Check error
            err, S = get_tensor_error(U, V, W, n, m, p, rank)
            
            if err < best_err:
                best_err = err
                best_state = (U.copy(), V.copy(), W.copy())
            
            # If close enough, try to polish
            if err < 1e-8:
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                err_r, _ = get_tensor_error(U_r, V_r, W_r, n, m, p, rank)
                if err_r < 1e-6:
                    return U_r, V_r, W_r
            
            # Early stopping if converged
            if err < 1e-12:
                break
        
        return None

    def try_rank_random_search(rank, seed):
        """Improved random search with structured initialization and local refinement."""
        if time.time() > deadline - 15:
            return None
            
        rng = np.random.RandomState(seed)
        
        # Try multiple random initializations with different strategies
        for attempt in range(30):
            if time.time() > deadline - 10:
                return None
            
            # Strategy 1: Random half-integers
            if attempt % 3 == 0:
                U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
                V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
                W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            # Strategy 2: Mostly zeros with some non-zero
            elif attempt % 3 == 1:
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for _ in range(rank * 3):
                    mat = rng.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        U[rng.randint(n*m), rng.randint(rank)] = rng.choice([-1, -0.5, 0.5, 1])
                    elif mat == 'V':
                        V[rng.randint(m*p), rng.randint(rank)] = rng.choice([-1, -0.5, 0.5, 1])
                    else:
                        W[rng.randint(p*n), rng.randint(rank)] = rng.choice([-1, -0.5, 0.5, 1])
            # Strategy 3: Sparse with structured patterns
            else:
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                for t in range(rank):
                    # Assign random non-zero values to random positions
                    U[rng.randint(n*m), t] = rng.choice([-1, -0.5, 0.5, 1])
                    V[rng.randint(m*p), t] = rng.choice([-1, -0.5, 0.5, 1])
                    W[rng.randint(p*n), t] = rng.choice([-1, -0.5, 0.5, 1])
            
            err, S = get_tensor_error(U, V, W, n, m, p, rank)
            if err < 1e-8:
                return U, V, W
            
            # Local refinement: try changing individual entries
            for flip_round in range(30):
                if time.time() > deadline - 10:
                    return None
                
                improved = False
                # Try changing random entries to different half-integer values
                for _ in range(150):
                    mat = rng.choice(['U', 'V', 'W'])
                    possible_vals = [-1, -0.5, 0, 0.5, 1]
                    
                    if mat == 'U':
                        row, col = rng.randint(n*m), rng.randint(rank)
                        old_val = U[row, col]
                        new_val = rng.choice(possible_vals)
                        U[row, col] = new_val
                        err_new, _ = get_tensor_error(U, V, W, n, m, p, rank)
                        if err_new < err - 1e-14:
                            err = err_new
                            improved = True
                        else:
                            U[row, col] = old_val
                    
                    elif mat == 'V':
                        row, col = rng.randint(m*p), rng.randint(rank)
                        old_val = V[row, col]
                        new_val = rng.choice(possible_vals)
                        V[row, col] = new_val
                        err_new, _ = get_tensor_error(U, V, W, n, m, p, rank)
                        if err_new < err - 1e-14:
                            err = err_new
                            improved = True
                        else:
                            V[row, col] = old_val
                    
                    else:
                        row, col = rng.randint(p*n), rng.randint(rank)
                        old_val = W[row, col]
                        new_val = rng.choice(possible_vals)
                        W[row, col] = new_val
                        err_new, _ = get_tensor_error(U, V, W, n, m, p, rank)
                        if err_new < err - 1e-14:
                            err = err_new
                            improved = True
                        else:
                            W[row, col] = old_val
                
                if not improved:
                    break
                
                if err < 1e-8:
                    return U, V, W
        
        return None

    best_result = None
    
    # Try ranks from low to high, with more aggressive search for lower ranks
    for rank in [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
        if time.time() > deadline - 20:
            break
        
        # More seeds for lower ranks
        num_als_seeds = max(5, 20 - rank)
        num_random_seeds = max(3, 15 - rank)
        
        # Try ALS
        for seed in range(num_als_seeds):
            if time.time() > deadline - 15:
                break
            res = try_rank_als(rank, seed)
            if res is not None:
                U, V, W = res
                best_result = {
                    "U": [[float(x) for x in row] for row in U],
                    "V": [[float(x) for x in row] for row in V],
                    "W": [[float(x) for x in row] for row in W]
                }
                break
        
        if best_result:
            break
        
        # Try random search
        for seed in range(num_random_seeds):
            if time.time() > deadline - 15:
                break
            res = try_rank_random_search(rank, seed)
            if res is not None:
                U, V, W = res
                best_result = {
                    "U": [[float(x) for x in row] for row in U],
                    "V": [[float(x) for x in row] for row in V],
                    "W": [[float(x) for x in row] for row in W]
                }
                break
        
        if best_result:
            break

    if best_result:
        return best_result

    # Fallback schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
