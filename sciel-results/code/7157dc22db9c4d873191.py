"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 84
    
    # Build the target tensor T[a, b, c]
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Trivial decomposition
    U_trivial = np.zeros((n*m, trivial_rank), dtype=np.float64)
    V_trivial = np.zeros((m*p, trivial_rank), dtype=np.float64)
    W_trivial = np.zeros((p*n, trivial_rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, col] = 1.0
                V_trivial[b, col] = 1.0
                W_trivial[c, col] = 1.0
                col += 1
                
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = trivial_rank

    def verify_decomposition(U, V, W, rank):
        """Verify that sum_r U[:,r] outer V[:,r] outer W[:,r] == T"""
        recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(recon, T, atol=1e-8)

    def try_rank_decomposition(target_rank, time_limit=10.0):
        """
        Attempt to find a decomposition of the given rank using gradient descent.
        Returns (U, V, W) if successful, else None.
        """
        if time.time() > deadline - 5:
            return None
        
        start_t = time.time()
        
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > start_t + time_limit:
                return None
            
            np.random.seed(42 + attempt)
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Adaptive learning rate
            lr = 0.01
            best_err = float('inf')
            best_state = (U.copy(), V.copy(), W.copy())
            
            for step in range(2000):
                if time.time() > start_t + time_limit:
                    break
                
                # Compute reconstruction efficiently
                recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                for r in range(target_rank):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
                err = T - recon
                err_norm = np.linalg.norm(err)
                
                if err_norm < 1e-6:
                    return U, V, W
                
                if err_norm < best_err:
                    best_err = err_norm
                    best_state = (U.copy(), V.copy(), W.copy())
                
                # Compute gradients
                grad_U = np.einsum('abc,br,cr->ar', err, V, W)
                grad_V = np.einsum('abc,ar,cr->br', err, U, W)
                grad_W = np.einsum('abc,ar,br->cr', err, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate
                new_err = np.linalg.norm(T - (lambda: np.zeros((n*m, m*p, p*n), dtype=np.float64))())
                # Actually, let's just keep it simple for now
                
        # Final check with best state
        U_best, V_best, W_best = best_state
        recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(target_rank):
            recon += np.einsum('a,b,c->abc', U_best[:, r], V_best[:, r], W_best[:, r])
        if np.allclose(recon, T, atol=1e-6):
            return U_best, V_best, W_best
        return None

    # Try to find a lower rank
    # We try a few specific ranks below 84
    candidate_ranks = [83, 82, 81, 80, 79, 78, 77, 76, 75]
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
        result = try_rank_decomposition(r, time_limit=8.0)
        if result is not None:
            U_new, V_new, W_new = result
            if verify_decomposition(U_new, V_new, W_new, r):
                best_U = U_new
                best_V = V_new
                best_W = W_new
                best_rank = r
                # If we found a lower rank, we can try to go even lower
                break  # Stop at first success to save time
    
    # If we didn't find any lower rank, we return the trivial one
    # Convert to lists
    U_out = best_U[:, :best_rank].tolist()
    V_out = best_V[:, :best_rank].tolist()
    W_out = best_W[:, :best_rank].tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
