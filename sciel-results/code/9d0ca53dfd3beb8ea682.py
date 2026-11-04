import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification: T[i,k,a,b,c,d] = sum_r U[a*m+b,r]*V[c*p+d,r]*W[k*n+i,r] should equal delta_{i,a}*delta_{b,c}*delta_{d,k}"""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # For each (a,b,c,d), check all (i,k)
    # T[i,k,a,b,c,d] = sum_r U[a*m+b,r] * V[c*p+d,r] * W[k*n+i,r]
    # We need: for fixed (a,b,c,d), the matrix M[i,k] = sum_r U[a*m+b,r]*V[c*p+d,r]*W[k*n+i,r]
    # should equal delta_{b,c} * (column i = a, row k = d) i.e., M[i,k] = 1 if i==a and k==d and b==c, else 0
    
    # Reshape W to (p, n, rank)
    W_reshaped = W_arr.reshape(p, n, rank)  # W_reshaped[k, i, r] = W[k*n+i, r]
    
    for a in range(n):
        for b in range(m):
            u_vec = U_arr[a * m + b, :]  # (rank,)
            for c in range(m):
                for d in range(p):
                    v_val = V_arr[c * p + d, :]  # (rank,)
                    # Compute M[i,k] = sum_r u_vec[r] * v_val[r] * W_reshaped[k,i,r]
                    # M[i,k] = sum_r (u_vec * v_val)[r] * W_reshaped[k,i,r]
                    uv = u_vec * v_val  # (rank,)
                    # M = W_reshaped.T @ uv  -> (n, p) where M[i,k] = sum_r W_reshaped[k,i,r]*uv[r]
                    M = W_reshaped.transpose(1, 0, 2) @ uv  # (n, p, rank) @ (rank,) -> (n, p)
                    # M[i,k] should be 1 if a==i and d==k and b==c, else 0
                    expected = np.zeros((n, p))
                    if b == c:
                        expected[a, d] = 1.0
                    if np.max(np.abs(M - expected)) > tol:
                        return False
    return True

def _make_schoolbook(n, m, p):
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Start with schoolbook
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions
    # Strategy: use structured patterns based on the specific dimensions
    
    # For (2,5,5), try various structured approaches
    
    # Approach 1: Try rank values from 25 upward
    # Use a smarter search: for each candidate rank, try to build a valid decomposition
    
    # Key insight: we can use "grouping" - for each (j,k) pair, we need to handle
    # the contributions to C[0,k] and C[1,k] from A[0,j]*B[j,k] and A[1,j]*B[j,k]
    
    # Let's try a parametric family of decompositions
    
    # For rank R, try random assignments with structure
    rng = random.Random(42)
    
    # Try ranks from 25 to best_rank-1
    for R in range(25, best_rank):
        if not still_going():
            break
        
        # Try multiple random structured attempts
        max_attempts = 2000 if R <= 35 else 500
        for attempt in range(max_attempts):
            if not still_going():
                break
            
            # Strategy: assign each of the R rank-1 terms to cover certain (j,k) pairs
            # with some probability of covering both i=0 and i=1
            
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Each rank-1 term r is assigned a (j,k) pair and a pattern for i
            for r in range(R):
                j = rng.randint(0, m - 1)
                k = rng.randint(0, p - 1)
                
                # V: activate (j,k)
                V[j * p + k, r] = 1.0
                
                # U and W: choose pattern
                # Option 1: only i=0
                # Option 2: only i=1  
                # Option 3: both with coefficients
                choice = rng.random()
                if choice < 0.3:
                    # Only i=0
                    U[0 * m + j, r] = 1.0
                    W[k * n + 0, r] = 1.0
                elif choice < 0.6:
                    # Only i=1
                    U[1 * m + j, r] = 1.0
                    W[k * n + 1, r] = 1.0
                else:
                    # Both, with random coefficients
                    c0 = rng.choice([-1.0, 1.0])
                    c1 = rng.choice([-1.0, 1.0])
                    U[0 * m + j, r] = c0
                    U[1 * m + j, r] = c1
                    W[k * n + 0, r] = c0
                    W[k * n + 1, r] = c1
            
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_rank = R
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    return best
