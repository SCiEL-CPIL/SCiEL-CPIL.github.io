import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    # n=3, m=3, p=7
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (21, rank)
    # W: (p*n, rank) = (21, rank)
    
    # Build the target tensor T[i,j,k] = 1 for all i,j,k
    # T has shape (3, 3, 7)
    T = np.ones((n, m, p))
    T_flat = T.flatten()
    T_size = T_flat.size  # 63
    
    # Schoolbook as fallback
    best_rank = n * m * p  # 63
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_result = {'U': U_school.tolist(), 'V': V_school.tolist(), 'W': W_school.tolist()}
    best_error = np.inf
    
    # Helper to compute error
    def compute_error(U, V, W, rank):
        # Reshape to tensor forms
        # U: (n, m, rank)
        U_t = U.reshape(n, m, rank)
        # V: (m, p, rank)
        V_t = V.reshape(m, p, rank)
        # W: (p, n, rank)
        W_t = W.reshape(p, n, rank)
        
        # T_recon[i,j,k] = sum_c U_t[i,j,c] * V_t[j,k,c] * W_t[k,i,c]
        T_recon = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        return np.linalg.norm(T_flat - T_recon.flatten())

    # Try to find lower rank decompositions
    # We search from low rank up. If we find a valid one, we return it.
    # The hint suggests rank 33 might be achievable.
    # We will try ranks from 20 to 62.
    
    for rank in range(20, best_rank):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        found_valid = False
        for trial in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + rank * 100 + trial)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Alternating Least Squares
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # 1. Update U
                # Variables: U_flat (n*m*rank,)
                # Equations: T_flat (n*m*p,)
                # A_U shape: (n*m*p, n*m*rank)
                A_U = np.zeros((n*m*p, n*m*rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = (i*m + j)*p + k
                            for c in range(rank):
                                col = (i*m + j)*rank + c
                                # Coeff is V[j,k,c] * W[k,i,c]
                                # V is (m*p, rank) -> V[j*p+k, c]
                                # W is (p*n, rank) -> W[k*n+i, c]
                                A_U[row, col] = V[j*p+k, c] * W[k*n+i, c]
                
                try:
                    U_flat_sol, _, _, _ = np.linalg.lstsq(A_U, T_flat, rcond=None)
                    U = U_flat_sol.reshape(n*m, rank)
                except np.linalg.LinAlgError:
                    pass
                
                # 2. Update V
                # Variables: V_flat (m*p*rank,)
                # A_V shape: (n*m*p, m*p*rank)
                A_V = np.zeros((n*m*p, m*p*rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = (i*m + j)*p + k
                            for c in range(rank):
                                col = (j*p + k)*rank + c
                                # Coeff is U[i,j,c] * W[k,i,c]
                                A_V[row, col] = U[i*m+j, c] * W[k*n+i, c]
                
                try:
                    V_flat_sol, _, _, _ = np.linalg.lstsq(A_V, T_flat, rcond=None)
                    V = V_flat_sol.reshape(m*p, rank)
                except np.linalg.LinAlgError:
                    pass
                
                # 3. Update W
                # Variables: W_flat (p*n*rank,)
                # A_W shape: (n*m*p, p*n*rank)
                A_W = np.zeros((n*m*p, p*n*rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = (i*m + j)*p + k
                            for c in range(rank):
                                col = (k*n + i)*rank + c
                                # Coeff is U[i,j,c] * V[j,k,c]
                                A_W[row, col] = U[i*m+j, c] * V[j*p+k, c]
                
                try:
                    W_flat_sol, _, _, _ = np.linalg.lstsq(A_W, T_flat, rcond=None)
                    W = W_flat_sol.reshape(p*n, rank)
                except np.linalg.LinAlgError:
                    pass
            
            # Check the error
            error = compute_error(U, V, W, rank)
            
            if error < 1e-6:
                # Found a decomposition!
                # Try to round to integers/half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                error_rounded = compute_error(U_rounded, V_rounded, W_rounded, rank)
                
                if error_rounded < 1e-6:
                    best_result = {'U': U_rounded.tolist(), 'V': V_rounded.tolist(), 'W': W_rounded.tolist()}
                    best_rank = rank
                    found_valid = True
                    break
                else:
                    # If rounding fails, we might still accept the float solution if the task allows it?
                    # "integer/half-integer entries" is specified. 
                    # If we can't round, this rank is not valid under the constraint.
                    # We continue searching.
                    pass
            
            if found_valid:
                break
        
        if found_valid:
            break

    return best_result
