import time
import numpy as np
import random

DIMENSIONS = [2, 5, 5]

def _verify_decomposition_np(U, V, W, n, m, p):
    """Verify decomposition using NumPy for speed."""
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # Target T[i, j, k] = 1
    
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # Reshape U to (n, m, rank)
    U_t = U.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V_t = V.reshape(m, p, rank)
    # Reshape W to (p, n, rank)
    W_t = W.reshape(p, n, rank)
    
    # We need to compute sum_c U[i,m,c] * V[j,p,c] * W[k,n,c] for all i,j,k
    # This is a batched contraction.
    # We can iterate over c or use einsum.
    # sum_c U[i,m,c] V[j,p,c] W[k,n,c]
    # This is equivalent to the trace of U_i^T V_j W_k? No.
    # It is the element-wise product summed over c.
    
    # Let's construct the tensor T of shape (n, m, p)
    # T[i, j, k] = sum_c U_t[i, j, c] * V_t[j, k, c] * W_t[k, i, c]
    
    # This is not a simple einsum because the indices are permuted in W.
    # W_t is (p, n, rank). We need W[k, i, c].
    # So W_t[k, i, c] is just W_t with axes (p, n, rank).
    # We want T[i, j, k].
    # Term c: U_t[i, j, c] * V_t[j, k, c] * W_t[k, i, c]
    
    # We can loop over c if rank is small, but rank is ~50.
    # Better: use einsum.
    # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    # Let's rename W indices to match: W' [i, k, c] = W[k, i, c]
    # So W_trans = W_t.transpose(1, 0, 2) # shape (n, p, rank)
    # Then T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W_trans[i, k, c]
    
    W_trans = W_t.transpose(1, 0, 2) # (n, p, rank)
    
    # T = np.einsum('ijc,jkc,ikc->ijk', U_t, V_t, W_trans)
    
    T = np.einsum('ijc,jkc,ikc->ijk', U_t, V_t, W_trans)
    
    # Check if T is all 1s
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_local_search(n, m, p, rank, time_limit=10.0, seed=42):
    """Try to find a valid decomposition with given rank using local search."""
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Allowed values: 0, 0.5, -0.5, 1, -1
    values = np.array([0, 0.5, -0.5, 1, -1])
    
    start_time = time.time()
    
    for attempt in range(1000):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with random values
        U = rng.choice(values, size=(nm, rank))
        V = rng.choice(values, size=(mp, rank))
        W = rng.choice(values, size=(pn, rank))
        
        # Local search
        # We will use a simple gradient-like or random hill climbing.
        # Since values are discrete, we do random flips.
        
        current_error = None
        for step in range(5000):
            if time.time() - start_time > time_limit:
                break
            
            # Verify
            if _verify_decomposition_np(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
            
            # Calculate error to guide search?
            # For speed, we just pick a random variable and flip it.
            # If error decreases, keep. If not, revert (or keep with prob p for SA).
            
            # To be efficient, we don't calculate full error every step,
            # we just try a move and verify? No, verification is expensive.
            # We need a fast error metric.
            
            # Let's compute the error tensor T
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            W_trans = W_t.transpose(1, 0, 2)
            T = np.einsum('ijc,jkc,ikc->ijk', U_t, V_t, W_trans)
            
            if current_error is None:
                current_error = np.sum((T - 1.0)**2)
            
            # Pick a random variable
            var_idx = rng.randrange(3)
            if var_idx == 0:
                i, j, c = rng.randrange(n), rng.randrange(m), rng.randrange(rank)
                old_val = U[i*m+j, c]
                new_val = rng.choice(values)
                if new_val == old_val: continue
                U[i*m+j, c] = new_val
            elif var_idx == 1:
                j, k, c = rng.randrange(m), rng.randrange(p), rng.randrange(rank)
                old_val = V[j*p+k, c]
                new_val = rng.choice(values)
                if new_val == old_val: continue
                V[j*p+k, c] = new_val
            else:
                k, i, c = rng.randrange(p), rng.randrange(n), rng.randrange(rank)
                old_val = W[k*n+i, c]
                new_val = rng.choice(values)
                if new_val == old_val: continue
                W[k*n+i, c] = new_val
            
            # Recalculate error
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            W_trans = W_t.transpose(1, 0, 2)
            T_new = np.einsum('ijc,jkc,ikc->ijk', U_t, V_t, W_trans)
            new_error = np.sum((T_new - 1.0)**2)
            
            if new_error < current_error:
                current_error = new_error
            else:
                # Revert
                if var_idx == 0:
                    U[i*m+j, c] = old_val
                elif var_idx == 1:
                    V[j*p+k, c] = old_val
                else:
                    W[k*n+i, c] = old_val

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # First, get the schoolbook baseline
    baseline = _schoolbook(n, m, p)
    best_result = baseline
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Lower bound: 25
    for rank in range(49, 24, -1):
        if time.time() > deadline:
            break
        
        # Try local search
        result = _try_rank_local_search(n, m, p, rank, time_limit=10.0, seed=rank)
        if result is not None:
            best_result = result
            best_rank = rank
            break
            
    return best_result
