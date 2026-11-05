"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the schoolbook (rank n*m*p) decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta_{i,k} for all i,j,k."""
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    rank = U.shape[1]
    # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We can do this efficiently:
    # For each c, U[:,c] is vector of length n*m, V[:,c] length m*p, W[:,c] length p*n
    # T[i,j,k] += U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    #
    # Let's reshape:
    # U_c reshaped to (n, m)
    # V_c reshaped to (m, p)
    # W_c reshaped to (p, n)
    # T += np.einsum('ij,jk,ki->ik', U_c, V_c, W_c) but we need full (i,j,k)
    # Actually we need T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # The target is: T[i,j,k] should equal 1 if i==k else 0 (since matrix mult: (AB)[i,k] = sum_j A[i,j]B[j,k],
    # and the tensor T[i,j,k] represents the coefficient for A[i,j]*B[j,k] in output C[i,k],
    # so T[i,j,k] = delta_{i,k})
    
    # Efficient check:
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For fixed i,k: sum_j T[i,j,k] should be... no, each (i,j,k) independently must equal delta_{i,k}
    
    # Let's just compute it directly with einsum for each c and accumulate
    T = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        # T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
        # = np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc)
        T += np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc)
    
    # Target: T[i,j,k] = 1 if i==k else 0
    target = np.zeros((n, m, p))
    for i in range(n):
        target[i, :, i] = 1.0
    
    diff = np.abs(T - target)
    return np.max(diff) < tol


def _try_random_decomposition(n, m, p, rank, num_trials=50, seed=42):
    """Try to find a rank-`rank` decomposition via random initialization and local search."""
    rng = np.random.RandomState(seed)
    best_err = float('inf')
    best = None
    
    for trial in range(num_trials):
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Compute error
        T = np.zeros((n, m, p))
        for c in range(rank):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', Uc, Vc, Vc * 0 + Wc)  # placeholder
        
        # Recompute properly
        T = np.zeros((n, m, p))
        for c in range(rank):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc)
        
        target = np.zeros((n, m, p))
        for i in range(n):
            target[i, :, i] = 1.0
        
        err = np.linalg.norm(T - target)
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
    
    return best, best_err


def _round_to_half_int(arr):
    """Round array to nearest half-integer (multiple of 0.5)."""
    return np.round(arr * 2) / 2


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook()
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    # Try decreasing ranks with random search
    # Known lower bounds: for 4x5x5, the rank is at least n*m = 20 (trivial) 
    # but better bounds exist. Let's try from rank 50 down.
    # Actually, let's be smarter. Try a range of ranks.
    
    # For each candidate rank, do a few random trials
    ranks_to_try = list(range(50, 100))  # Try ranks 50-99
    ranks_to_try.sort()  # Start from lower ranks
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Multiple random trials for this rank
        for trial_seed in range(20):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Compute error
            T = np.zeros((n, m, p))
            for c in range(rank):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                T += np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc)
            
            target = np.zeros((n, m, p))
            for i in range(n):
                target[i, :, i] = 1.0
            
            err = np.linalg.norm(T - target)
            
            if err < 1e-3:
                # Close! Try to refine with local search
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                    # Gradient step
                    lr = 0.01
                    dU = np.zeros_like(U)
                    dV = np.zeros_like(V)
                    dW = np.zeros_like(W)
                    
                    for c in range(rank):
                        Uc = U[:, c].reshape(n, m)
                        Vc = V[:, c].reshape(m, p)
                        Wc = W[:, c].reshape(p, n)
                        
                        # Gradient w.r.t. U[:,c]: dU[i*m+j,c] = sum_{k'} (T[i,j,k'] - target[i,j,k']) * Vc[j,k'] * Wc[k',i]
                        # This is getting complex. Let's just do a simple random perturbation instead.
                        pass
                    
                    # Simple random perturbation
                    noise_scale = 0.1
                    U += noise_scale * np.random.randn(n * m, rank)
                    V += noise_scale * np.random.randn(m * p, rank)
                    W += noise_scale * np.random.randn(p * n, rank)
                    
                    T_new = np.zeros((n, m, p))
                    for c in range(rank):
                        Uc = U[:, c].reshape(n, m)
                        Vc = V[:, c].reshape(m, p)
                        Wc = W[:, c].reshape(p * n, rank).reshape(p, n)
                        T_new += np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc)
                    err_new = np.linalg.norm(T_new - target)
                    
                    if err_new > err:
                        U -= noise_scale * np.random.randn(n * m, rank)  # revert approximately
                        V -= noise_scale * np.random.randn(m * p, rank)
                        W -= noise_scale * np.random.randn(p * n, rank)
                        break
                    else:
                        err = err_new
                        if err < 1e-6:
                            break
                
                if err < 1e-6:
                    # Found a valid decomposition!
                    best_result = (U, V, W)
                    best_rank = rank
                    break
            if best_rank < n * m * p:
                break
        if best_rank < n * m * p:
            break
    
    # If we found something better, use it; otherwise schoolbook
    U_best, V_best, W_best = best_result
    
    # Convert to lists of floats (or half-integers if close)
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
