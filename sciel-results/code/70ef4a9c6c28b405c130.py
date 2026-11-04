"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Schoolbook rank: n*m*p = 60
    rank_schoolbook = n * m * p
    
    # Build schoolbook decomposition as fallback
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
    
    def verify(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    best_rank = rank_schoolbook
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # We'll try ranks from 30 upward
    
    np.random.seed(42)
    
    for trial_rank in range(30, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts for each rank
        n_attempts = 20 if trial_rank <= 40 else 10
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            scale = 1.0 / np.sqrt(trial_rank)
            U_try = np.random.randn(n * m, trial_rank) * scale
            V_try = np.random.randn(m * p, trial_rank) * scale
            W_try = np.random.randn(p * n, trial_rank) * scale
            
            # Alternating Least Squares
            success = False
            for iteration in range(500):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                        # Let A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # Then A @ U[i*m+j, :] = ones(p)
                        A = np.zeros((p, trial_rank))
                        for k in range(p):
                            A[k, :] = V_try[j*p+k] * W_try[k*n+i]
                        b = np.ones(p)
                        try:
                            U_try[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), solve for V[j*p+k, :]
                for j in range(m):
                    for k in range(p):
                        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                        # Let A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        # Then A @ V[j*p+k, :] = ones(n)
                        A = np.zeros((n, trial_rank))
                        for i in range(n):
                            A[i, :] = U_try[i*m+j] * W_try[k*n+i]
                        b = np.ones(n)
                        try:
                            V_try[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i), solve for W[k*n+i, :]
                for k in range(p):
                    for i in range(n):
                        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                        # Let A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        # Then A @ W[k*n+i, :] = ones(m)
                        A = np.zeros((m, trial_rank))
                        for j in range(m):
                            A[j, :] = U_try[i*m+j] * V_try[j*p+k]
                        b = np.ones(m)
                        try:
                            W_try[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                if verify(U_try, V_try, W_try, tol=1e-10):
                    success = True
                    break
            
            if success:
                best_rank = trial_rank
                best_U = U_try.copy()
                best_V = V_try.copy()
                best_W = W_try.copy()
                break  # Found solution for this rank, try lower
        
        # If we found a solution at this rank, we can stop trying higher ranks
        # But we want the minimum, so continue to next (lower) rank if possible
        # Actually, we're going from 30 upward, so if we find one at 30, that's optimal
        # Let's break if we found a solution
        if best_rank == trial_rank:
            break
    
    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-8):
        # Fallback to schoolbook
        best_U = U_school
        best_V = V_school
        best_W = W_school
        best_rank = rank_schoolbook
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
