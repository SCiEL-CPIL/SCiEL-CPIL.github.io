"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Schoolbook decomposition: rank = n * m * p = 100
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    # Build the target tensor as a dense array for fast verification
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] = 1 iff j==j' and i==i' and k==k'
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify_fast(U_arr, V_arr, W_arr, R):
        """Fast verification using numpy tensor operations."""
        # Compute sum_r U[a,r]*V[b,r]*W[c,r] for all a,b,c
        # This is a 3-way tensor contraction
        # result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # = sum_r (U[a,r] * V[b,r]) * W[c,r]
        # Can compute as: for each r, outer product of U[:,r], V[:,r], W[:,r]
        result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(R):
            result += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
        return np.allclose(result, T, atol=1e-9)
    
    def verify_sparse(U_arr, V_arr, W_arr, R):
        """Check only the non-zero and a sample of zero entries."""
        # Check all non-zero entries (n*m*p = 100)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_arr[a] * V_arr[b], W_arr[c])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check a random sample of zero entries
        for _ in range(200):
            a = random.randint(0, n*m - 1)
            b = random.randint(0, m*p - 1)
            c = random.randint(0, p*n - 1)
            i, j = divmod(a, m)
            j2, k = divmod(b, p)
            k2, i2 = divmod(c, n)
            if j == j2 and i == i2 and k == k2:
                continue  # Skip non-zero entries
            val = np.dot(U_arr[a] * V_arr[b], W_arr[c])
            if abs(val) > 1e-9:
                return False
        return True
    
    # Strategy: For each candidate rank R, try random sparse U and V,
    # solve for W via least squares per (i,k), then verify.
    
    # Try ranks from 95 down to 25 (theoretical lower bound)
    for R in range(95, 24, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for trial in range(30):
            if time.time() > deadline - 2:
                break
            
            # Initialize U and V with sparse random integer patterns
            U_arr = np.zeros((n * m, R), dtype=np.float64)
            V_arr = np.zeros((m * p, R), dtype=np.float64)
            
            # Each row of U has 2-4 non-zeros with values in {-1, 0, 1}
            for a in range(n * m):
                num_nz = random.randint(2, 4)
                cols = random.sample(range(R), num_nz)
                for c in cols:
                    U_arr[a, c] = random.choice([-1, 1])
            
            for b in range(m * p):
                num_nz = random.randint(2, 4)
                cols = random.sample(range(R), num_nz)
                for c in cols:
                    V_arr[b, c] = random.choice([-1, 1])
            
            # Solve for W: for each (i,k), solve the system
            # For j in 0..m-1: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            W_arr = np.zeros((p * n, R), dtype=np.float64)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    c = k * n + i
                    # Build system matrix: m x R
                    A_sys = np.zeros((m, R), dtype=np.float64)
                    for j in range(m):
                        A_sys[j, :] = U_arr[i * m + j, :] * V_arr[j * p + k, :]
                    
                    # Solve A_sys @ w = ones(m)
                    try:
                        w_sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check residual
                    if len(residuals) > 0 and residuals[0] > 1e-8:
                        # Solution doesn't satisfy the equations well
                        valid = False
                        break
                    
                    W_arr[c, :] = w_sol
            
            if not valid:
                continue
            
            # Snap to nearest half-integer for cleaner entries
            W_arr = np.round(W_arr * 2) / 2
            
            # Verify
            if verify_sparse(U_arr, V_arr, W_arr, R):
                if verify_fast(U_arr, V_arr, W_arr, R):
                    # Convert to list of lists
                    best_U = U_arr.tolist()
                    best_V = V_arr.tolist()
                    best_W = W_arr.tolist()
                    best_rank = R
                    print(f"Found rank {R} decomposition!")
                    return {"U": best_U, "V": best_V, "W": best_W}
            
            # Check time
            if time.time() > deadline - 10:
                break
    
    # Try a different strategy: structured decomposition
    # Use the fact that for 5x5x5, we can potentially use FFT-based methods
    # For now, return best found
    
    if best_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
