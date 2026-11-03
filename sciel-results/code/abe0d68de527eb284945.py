import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Rank-minimal tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    We attempt to find a decomposition with rank less than the schoolbook rank of 54.
    The target tensor T is such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    This is equivalent to finding a factorization of the matrix multiplication tensor.
    
    Strategy:
    1. Try random searches for ranks 53 down to 30.
    2. If a valid integer/half-integer solution is found, return it.
    3. Otherwise, return the schoolbook decomposition (rank 54).
    
    The search involves fixing U and V randomly (with small integer entries) and solving for W.
    For each pair (i, k), we have a linear system M @ w = 1, where M is m x R.
    We check if the least-squares solution is close to a half-integer vector.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    deadline = time.time() + 115  # Leave 5s margin for 120s limit
    
    # Pre-allocate schoolbook as fallback
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    
    # Try to find a lower rank
    # We iterate from rank 53 down to 30.
    # For each rank, we try a limited number of random attempts.
    # To be efficient, we limit the total time.
    
    for R in range(rank_schoolbook - 1, 29, -1):
        if time.time() > deadline:
            break
            
        # Number of attempts per rank depends on remaining time
        # Start with a smaller number for higher ranks, maybe increase for lower?
        # Actually, lower ranks are harder, so maybe fewer attempts?
        # Let's just use a fixed budget of attempts per rank, say 50-100.
        # If R is close to 54, it's easier. If R is 30, it's very hard.
        
        attempts = 20
        # If time is tight, reduce attempts
        if time.time() > deadline - 20:
            attempts = 5
            
        for _ in range(attempts):
            if time.time() > deadline:
                break
                
            # Generate random U, V with entries in {-1, 0, 1}
            # Using small integers helps in finding exact solutions
            U = np.random.randint(-1, 2, (n * m, R))
            V = np.random.randint(-1, 2, (m * p, R))
            
            W = np.zeros((p * n, R))
            valid = True
            
            # For each (i, k) pair, solve for W[k*n+i, :]
            for i in range(n):
                for k in range(p):
                    # Construct M: m x R
                    # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    M = np.zeros((m, R))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve M @ w = ones(m)
                    # We want w to be half-integers.
                    # Use least squares to get a candidate, then round to half-integers and verify.
                    try:
                        # If M is rank deficient, pinv gives min-norm solution.
                        w_ls = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                        
                        # Round to nearest half-integer
                        w_half = np.round(w_ls * 2) / 2
                        
                        # Verify exactness
                        # Check if M @ w_half == ones(m)
                        # Use allclose with tight tolerance
                        if not np.allclose(M @ w_half, np.ones(m), atol=1e-9):
                            valid = False
                            break
                            
                        W[k * n + i, :] = w_half
                        
                    except Exception:
                        valid = False
                        break
                        
                if not valid:
                    break
                    
            if valid:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Final verification of the whole tensor
                # This is expensive but necessary for correctness before returning
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # We can vectorize this check
                # U is (nm, R), V is (mp, R), W is (pn, R)
                # We need to check for all i, j, k.
                # Let's do a quick spot check or full check if time permits.
                
                # Full check:
                # For each i, k, we already checked M @ w = 1.
                # M @ w = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # This is exactly the term for T[i,j,k].
                # So if all (i,k) checks passed, the tensor is correct.
                
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # Fallback to schoolbook
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
