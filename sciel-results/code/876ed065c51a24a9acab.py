import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 120  # seconds
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook
    
    # Try to find a rank-44 decomposition
    if time.time() - start_time < time_limit - 30:
        try:
            target_rank = 44
            rng = np.random.RandomState(42)
            
            # Initialize U, V randomly
            U = rng.randn(n * m, target_rank)
            V = rng.randn(m * p, target_rank)
            
            # Build the target tensor
            T = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i * m + j, j * p + k, k * n + i] = 1
            
            # For each entry (a,b), solve for W[:, :] using the equation:
            # T[a,b,:] = sum_r U[a,r] * V[b,r] * W[:,r]
            # This is a linear system: W[:,r] * (U[a,r] * V[b,r]) = T[a,b,:]
            # Let's set up the system for all (a,b) pairs.
            
            # The system is: sum_r (U[a,r] * V[b,r]) * W[c,r] = T[a,b,c] for all (a,b,c)
            # This is a linear system in W.
            
            # Build the matrix M where M[(a,b,c), r] = U[a,r] * V[b,r]
            # And the vector b where b[(a,b,c)] = T[a,b,c]
            # Then solve M * W = b for W.
            
            num_entries = n * m * m * p * p * n  # This is wrong
            # Actually, the number of (a,b,c) entries is (n*m) * (m*p) * (p*n) = 9*15*15 = 2025
            num_entries = (n*m) * (m*p) * (p*n)
            
            M = np.zeros((num_entries, target_rank))
            b_vec = np.zeros(num_entries)
            
            idx = 0
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        for r in range(target_rank):
                            M[idx, r] = U[a, r] * V[b, r]
                        b_vec[idx] = T[a, b, c]
                        idx += 1
            
            # Solve M * W = b_vec
            # This is an overdetermined system. Use least squares.
            W, residuals, rank, sv = np.linalg.lstsq(M, b_vec, rcond=None)
            
            # Check the error
            E = np.zeros((n*m, m*p, p*n))
            for r in range(target_rank):
                E += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = np.sum((T - E) ** 2)
            
            if err < 1e-10:
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
        except Exception as e:
            pass
    
    if time.time() - start_time > time_limit:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
