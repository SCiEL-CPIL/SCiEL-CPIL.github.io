import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    
    def verify_decomposition(U, V, W, rank):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        U_reshaped = U.reshape((n, m, rank))
        V_reshaped = V.reshape((m, p, rank))
        W_reshaped = W.reshape((p, n, rank))
        
        # Compute sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check if all entries are close to 1
        return np.allclose(T, 1.0, atol=1e-9)

    start_time = time.time()
    time_limit = 145  # seconds, safety margin before 150s kill
    
    # Search ranks from low to high.
    # Lower bound for 3x4x5 is known to be at least 20.
    # We try 20 to 40.
    
    for R in range(20, 41):
        if R >= best_rank:
            continue
            
        # Try to find a solution for this R
        # Use random restarts.
        
        while time.time() - start_time < time_limit:
            # Random initialization using numpy for speed
            # randint(-1, 2) gives -1, 0, 1
            U_rand = np.random.randint(-1, 2, size=(nm, R))
            V_rand = np.random.randint(-1, 2, size=(mp, R))
            
            # Build linear system A W = b
            # A is (n*m*p) x (R*n*p)
            A = np.zeros((n * m * p, R * n * p))
            b = np.ones(n * m * p)
            
            # Fill A
            row_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        coeffs = U_rand[i * m + j] * V_rand[j * p + k]
                        start_col = (k * n + i) * R
                        A[row_idx, start_col:start_col+R] = coeffs
                        row_idx += 1
            
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W_sol = W_flat.reshape((p * n, R))
            
            # Check if W_sol is close to half-integers
            W_doubled = 2 * W_sol
            W_rounded = np.round(W_doubled)
            
            # Tolerance for half-integer check
            if np.max(np.abs(W_doubled - W_rounded)) < 1e-6:
                W_nice = (W_rounded / 2.0)
                
                # Verify exact decomposition
                if verify_decomposition(U_rand, V_rand, W_nice, R):
                    best_rank = R
                    best_U = U_rand.tolist()
                    best_V = V_rand.tolist()
                    best_W = W_nice.tolist()
                    return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
