import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 100
    
    deadline = time.time() + 140
    
    # Precompute index mappings
    ijks = list(product(range(n), range(m), range(p)))
    U_idx = np.array([i*m + j for i,j,k in ijks])
    V_idx = np.array([j*p + k for i,j,k in ijks])
    W_idx = np.array([k*n + i for i,j,k in ijks])
    
    def compute_errors(U, V, W):
        R = U.shape[1]
        terms = U[U_idx] * V[V_idx] * W[W_idx]
        sums = terms.sum(axis=1)
        return np.abs(sums - 1.0)
    
    def verify(U, V, W):
        errors = compute_errors(U, V, W)
        return np.all(errors < 1e-9)
    
    # Start with schoolbook
    best_rank = schoolbook_rank
    best_U = np.zeros((n*m, schoolbook_rank))
    best_V = np.zeros((m*p, schoolbook_rank))
    best_W = np.zeros((p*n, schoolbook_rank))
    for idx, (i,j,k) in enumerate(ijks):
        best_U[i*m+j, idx] = 1.0
        best_V[j*p+k, idx] = 1.0
        best_W[k*n+i, idx] = 1.0
        
    # Try to find lower rank decompositions
    # Target ranks: 25, 30, 35, ...
    target_ranks = [25, 28, 30, 35, 40]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
            
        success = False
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Sparse random initialization
            # Each column has a few non-zeros
            for r in range(target_rank):
                # Pick random rows for U, V, W
                # Ensure some overlap to allow non-zero products
                u_rows = np.random.choice(n*m, 3, replace=False)
                v_rows = np.random.choice(m*p, 3, replace=False)
                w_rows = np.random.choice(p*n, 3, replace=False)
                
                U[u_rows, r] = np.random.choice([-1.0, 1.0], 3)
                V[v_rows, r] = np.random.choice([-1.0, 1.0], 3)
                W[w_rows, r] = np.random.choice([-1.0, 1.0], 3)
            
            # Optimize
            # Use a simple gradient descent or random perturbation
            # We'll use a "hill climbing" approach with random perturbations
            current_err = compute_errors(U, V, W).max()
            
            for step in range(500):
                if time.time() > deadline:
                    break
                
                # Pick a random column and a random factor
                r = np.random.randint(target_rank)
                factor = np.random.choice(['U', 'V', 'W'])
                
                if factor == 'U':
                    row = np.random.randint(n*m)
                    old_val = U[row, r]
                    U[row, r] += np.random.randn() * 0.1
                    new_err = compute_errors(U, V, W).max()
                    if new_err < current_err:
                        current_err = new_err
                    else:
                        U[row, r] = old_val
                        
                elif factor == 'V':
                    row = np.random.randint(m*p)
                    old_val = V[row, r]
                    V[row, r] += np.random.randn() * 0.1
                    new_err = compute_errors(U, V, W).max()
                    if new_err < current_err:
                        current_err = new_err
                    else:
                        V[row, r] = old_val
                        
                else:
                    row = np.random.randint(p*n)
                    old_val = W[row, r]
                    W[row, r] += np.random.randn() * 0.1
                    new_err = compute_errors(U, V, W).max()
                    if new_err < current_err:
                        current_err = new_err
                    else:
                        W[row, r] = old_val
                
                if current_err < 1e-9:
                    break
            
            if current_err < 1e-9:
                if verify(U, V, W):
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    success = True
                    break
        
        if success:
            continue
        else:
            # If we fail at this rank, try next
            pass
            
    # Final verification
    if best_rank < schoolbook_rank:
        errors = compute_errors(best_U, best_V, best_W)
        if errors.max() < 1e-9:
            best_U[np.abs(best_U) < 1e-8] = 0
            best_V[np.abs(best_V) < 1e-8] = 0
            best_W[np.abs(best_W) < 1e-8] = 0
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
