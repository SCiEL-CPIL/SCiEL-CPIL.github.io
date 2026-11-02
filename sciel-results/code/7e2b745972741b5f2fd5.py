import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

def _construct_tensor(n, m, p):
    """Construct the full tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i*m+j, j*p+k, k*n+i] = 1
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _tensor_error(U, V, W, T):
    """Calculate the squared Frobenius error between the reconstructed tensor and T."""
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Reconstructed: sum_r U[:,r] outer V[:,r] outer W[:,r]
    # Efficient calculation:
    # Rec[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # This is equivalent to the tensor product.
    # We can compute this by summing rank-1 tensors.
    R = U.shape[1]
    Rec = np.zeros_like(T)
    for r in range(R):
        Rec += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return np.sum((Rec - T)**2)

def _local_search_optimize(U, V, W, T, max_iter=1000, step=0.1, seed=42):
    """Local search to minimize tensor error."""
    rng = np.random.RandomState(seed)
    U = U.copy()
    V = V.copy()
    W = W.copy()
    current_err = _tensor_error(U, V, W, T)
    
    best_err = current_err
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Randomly pick a matrix and an index
        mat = rng.randint(3)
        if mat == 0:
            idx = (rng.randint(U.shape[0]), rng.randint(U.shape[1]))
            delta = rng.uniform(-step, step)
            old_val = U[idx]
            U[idx] += delta
            new_err = _tensor_error(U, V, W, T)
            if new_err < current_err:
                current_err = new_err
                if current_err < best_err:
                    best_err = current_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                U[idx] = old_val
        elif mat == 1:
            idx = (rng.randint(V.shape[0]), rng.randint(V.shape[1]))
            delta = rng.uniform(-step, step)
            old_val = V[idx]
            V[idx] += delta
            new_err = _tensor_error(U, V, W, T)
            if new_err < current_err:
                current_err = new_err
                if current_err < best_err:
                    best_err = current_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                V[idx] = old_val
        else:
            idx = (rng.randint(W.shape[0]), rng.randint(W.shape[1]))
            delta = rng.uniform(-step, step)
            old_val = W[idx]
            W[idx] += delta
            new_err = _tensor_error(U, V, W, T)
            if new_err < current_err:
                current_err = new_err
                if current_err < best_err:
                    best_err = current_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                W[idx] = old_val
                
    return best_U, best_V, best_W, best_err

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    n, m, p = 3, 3, 6
    T = _construct_tensor(n, m, p)
    
    # Schoolbook initialization
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    # Try to find lower rank solutions
    # Start from rank 53 down to 45
    for target_rank in range(rank_sb - 1, 44, -1):
        if time.time() > deadline - 10:
            break
            
        # Initialize with random noise or perturbed schoolbook
        # Using perturbed schoolbook might be safer?
        # Let's try random initialization with small values
        # Or better: Take schoolbook, remove one column, and optimize the rest?
        
        # Strategy: Remove one column from schoolbook, then optimize the remaining target_rank columns
        # This requires target_rank = rank_sb - 1 for the first step.
        if target_rank == rank_sb - 1:
            # Remove last column
            U_init = U_sb[:, :-1].copy()
            V_init = V_sb[:, :-1].copy()
            W_init = W_sb[:, :-1].copy()
        else:
            # For lower ranks, random restart
            U_init = np.random.randn(n*m, target_rank) * 0.1
            V_init = np.random.randn(m*p, target_rank) * 0.1
            W_init = np.random.randn(p*n, target_rank) * 0.1
            
        # Local search
        U_opt, V_opt, W_opt, err = _local_search_optimize(
            U_init, V_init, W_opt if 'W_opt' in locals() else W_init, T, 
            max_iter=500, step=0.05, seed=target_rank
        )
        
        if err < 1e-6:
            best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
            best_rank = target_rank
            break # Found a solution at this rank, try lower?
            # Actually, we want the minimal rank, so we should continue if time permits
            # But breaking is safer to ensure valid output
            
    return best_result
# EVOLVE-BLOCK-END
