import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
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
    U_list = U_school.tolist()
    V_list = V_school.tolist()
    W_list = W_school.tolist()
    
    # Try to find a lower rank decomposition
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit
    
    best_rank = rank_schoolbook
    best_U = U_list
    best_V = V_list
    best_W = W_list
    
    # Target ranks to try
    target_ranks = list(range(119, 99, -1))
    
    for r in target_ranks:
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for trial in range(20):
            if time.time() > deadline:
                break
                
            # Random initialize U and V
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            
            # Build the linear system for W
            # Equation: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
            # Let M[i,j,k,c] = U[i,j,c] * V[j,k,c]
            # Then sum_c M[i,j,k,c] * W[k,i,c] = 1
            # This is a linear system in W.
            # W has shape (p*n, r).
            # There are n*m*p equations.
            # Each equation involves r terms of W.
            # We can write this as A * W_flat = b
            # Where A is (n*m*p) x (p*n*r)
            
            A = np.zeros((n * m * p, p * n * r))
            b = np.ones(n * m * p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            # W index is k*n + i
                            # W_flat index is (k*n + i) * r + c
                            A[idx, (k * n + i) * r + c] = U[i * m + j, c] * V[j * p + k, c]
                        idx += 1
            
            # Solve least squares
            W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Check residual
            if residuals is not None and len(residuals) > 0 and residuals[0] < 1e-6:
                # Reshape W
                W = W_flat.reshape(p * n, r)
                
                # Check if we can round to half-integers
                # Scale by 2 to get integers
                U_2 = np.round(U * 2) / 2
                V_2 = np.round(V * 2) / 2
                W_2 = np.round(W * 2) / 2
                
                # Verify exactness
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(r):
                                val += U_2[i * m + j, c] * V_2[j * p + k, c] * W_2[k * n + i, c]
                            if abs(val - 1) > 1e-9:
                                error = 1
                                break
                        if error:
                            break
                    if error:
                        break
                
                if not error:
                    # Found a valid decomposition
                    best_rank = r
                    best_U = U_2.tolist()
                    best_V = V_2.tolist()
                    best_W = W_2.tolist()
                    break # Found for this r, try lower r? No, break and return?
                    # Actually, we can break the trial loop, but continue to lower r?
                    # The goal is to minimize rank. So if we find r, we should try r-1.
                    # But finding r-1 might be harder.
                    # Let's just break and return this one.
                    # Or continue to see if we can find a lower one.
                    # Given time, let's break.
                    break
        
        if best_rank < rank_schoolbook:
            # We found a better one, but we might find an even better one.
            # Let's continue to next r.
            pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
