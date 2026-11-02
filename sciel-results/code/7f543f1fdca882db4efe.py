"""Combined program for the suite task; uses schoolbook with structured low-rank search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin before 150s kill
    n, m, p = 3, 5, 7
    
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    def verify(U, V, W, tol=1e-10):
        """Verify that sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1 for all i,j,k."""
        n, m, p = 3, 5, 7
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        # Build the tensor
        T = np.zeros((n, m, p), dtype=float)
        for t in range(r):
            u_t = U[:, t].reshape(n, m)
            v_t = V[:, t].reshape(m, p)
            w_t = W[:, t].reshape(p, n)
            # Outer product: T[i,j,k] += u_t[i,j] * v_t[j,k] * w_t[k,i]
            for i in range(n):
                for j in range(m):
                    T[i, j, :] += u_t[i, j] * v_t[j, :] * w_t[:, i]
        # Check if T is all ones
        if not np.allclose(T, 1.0, atol=tol):
            return False
        return True
    
    def try_random_search(target_rank, max_iters=50000):
        """
        Try to find a rank-target_rank decomposition using random initialization
        and coordinate descent.
        """
        n, m, p = 3, 5, 7
        dim_u = n * m
        dim_v = m * p
        dim_w = p * n
        
        # Target tensor: all ones
        # We want: sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t] = ones(n,m,p)
        
        # Strategy: Start with schoolbook and try to perturb
        # Use random ±1 initializations for a subset of columns
        
        best_result = None
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initialize: use schoolbook for most, random for a few
            U = np.zeros((dim_u, target_rank), dtype=float)
            V = np.zeros((dim_v, target_rank), dtype=float)
            W = np.zeros((dim_w, target_rank), dtype=float)
            
            # Fill with schoolbook-like pattern for first target_rank-2 columns
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if idx >= target_rank - 2:
                            break
                        U[i*m+j, idx] = 1
                        V[j*p+k, idx] = 1
                        W[k*n+i, idx] = 1
                        idx += 1
                    if idx >= target_rank - 2:
                        break
                if idx >= target_rank - 2:
                    break
            
            # Last 2 columns: random ±1
            for t in range(idx, target_rank):
                U[:, t] = np.random.choice([-1, 1], size=dim_u)
                V[:, t] = np.random.choice([-1, 1], size=dim_v)
                W[:, t] = np.random.choice([-1, 1], size=dim_w)
            
            # Coordinate descent
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Check current error
                T = np.zeros((n, m, p), dtype=float)
                for t in range(target_rank):
                    u_t = U[:, t].reshape(n, m)
                    v_t = V[:, t].reshape(m, p)
                    w_t = W[:, t].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            T[i, j, :] += u_t[i, j] * v_t[j, :] * w_t[:, i]
                
                err = np.max(np.abs(T - 1.0))
                if err < 1e-10:
                    # Found a valid decomposition!
                    # Convert to integer/half-integer if possible
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    if verify(U_int, V_int, W_int):
                        return {'U': U_int.tolist(), 'V': V_int.tolist(), 'W': W_int.tolist()}
                    else:
                        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                
                # Update one random column
                t = np.random.randint(0, target_rank)
                # For column t, we want to minimize ||T_target - T_without_t - u_t⊗v_t⊗w_t||
                # This is still non-linear, so just do a small random perturbation
                if np.random.random() < 0.3:
                    # Flip a random entry in U
                    i, j = np.random.randint(0, n), np.random.randint(0, m)
                    U[i*m+j, t] *= -1
                elif np.random.random() < 0.6:
                    # Flip a random entry in V
                    j, k = np.random.randint(0, m), np.random.randint(0, p)
                    V[j*p+k, t] *= -1
                else:
                    # Flip a random entry in W
                    k, i = np.random.randint(0, p), np.random.randint(0, n)
                    W[k*n+i, t] *= -1
        
        return None
    
    # Try to find a lower-rank decomposition
    # Start with rank 104 (save 1 multiplication)
    result = try_random_search(104, max_iters=30000)
    
    if result is not None:
        return result
    
    # If that fails, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
