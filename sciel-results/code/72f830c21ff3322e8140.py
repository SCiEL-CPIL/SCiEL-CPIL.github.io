import numpy as np
import time
import random
import itertools

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook solution
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1.0
                    V[v, col] = 1.0
                    W[w, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Precompute indices
    nonzero_indices = []
    all_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                nonzero_indices.append((u, v, w))
                
    # We only need to check a subset of zero indices for speed, or all if fast enough
    # Total indices: n*m*p*n*m*p? No, the tensor is defined on U_idx x V_idx x W_idx.
    # U_idx size: n*m = 9
    # V_idx size: m*p = 12
    # W_idx size: p*n = 12
    # Total tuples: 9 * 12 * 12 = 1296.
    # Nonzero: 36.
    # Zero: 1260.
    # Checking all 1296 is feasible.
    
    zero_indices = []
    nonzero_set = set(nonzero_indices)
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                if (u, v, w) not in nonzero_set:
                    zero_indices.append((u, v, w))

    def calculate_error(U, V, W):
        # U, V, W are numpy arrays
        rank = U.shape[1]
        err = 0.0
        # Non-zero check
        for u, v, w in nonzero_indices:
            s = np.sum(U[u, :] * V[v, :] * W[w, :])
            err += (s - 1.0)**2
        # Zero check
        for u, v, w in zero_indices:
            s = np.sum(U[u, :] * V[v, :] * W[w, :])
            err += s**2
        return err

    def discrete_local_search(rank, max_restarts=5, max_iters=2000):
        if time.time() > deadline:
            return None
            
        vals = [0.0, 0.5, 1.0, -0.5, -1.0, 1.5, -1.5]
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with random discrete values
            U = np.random.choice(vals, size=(n*m, rank))
            V = np.random.choice(vals, size=(m*p, rank))
            W = np.random.choice(vals, size=(p*n, rank))
            
            curr_err = calculate_error(U, V, W)
            best_err = curr_err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            stuck = 0
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Pick a random matrix and position
                mat_type = random.choice(['U', 'V', 'W'])
                if mat_type == 'U':
                    mat = U
                    rows = n*m
                elif mat_type == 'V':
                    mat = V
                    rows = m*p
                else:
                    mat = W
                    rows = p*n
                    
                row = random.randint(0, rows - 1)
                col = random.randint(0, rank - 1)
                old_val = mat[row, col]
                
                # Try changing to a different value
                new_val = random.choice([v for v in vals if v != old_val])
                
                # Apply change
                mat[row, col] = new_val
                new_err = calculate_error(U, V, W)
                
                if new_err < curr_err:
                    curr_err = new_err
                    stuck = 0
                    if new_err < best_err:
                        best_err = new_err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        
                        # Check for exact solution
                        if best_err < 1e-9:
                            # Verify integer/half-integer (already true by construction)
                            # Verify exactness
                            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                else:
                    # Revert
                    mat[row, col] = old_val
                    stuck += 1
                    
                if stuck > 100:
                    break
                    
        return None

    # Search from low rank to high rank
    # Lower bound is max(nm, mp, pn) = 12.
    # Let's start from 15 to be safe but efficient.
    for rank in range(15, n*m*p + 1):
        if time.time() > deadline:
            break
        
        result = discrete_local_search(rank)
        if result is not None:
            return result
        
    # Fallback to schoolbook
    return schoolbook()
