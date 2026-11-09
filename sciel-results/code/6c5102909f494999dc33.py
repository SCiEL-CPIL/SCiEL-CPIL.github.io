import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # We want to decompose M_{3,3,6}
    # This is two copies of M_{3,3,3}
    # So we find a rank-R decomposition of M_{3,3,3}
    # and repeat it twice.
    
    # Target rank for 3x3x3
    # Try R from 23 to 27
    best_result = None
    best_rank = 54 # Schoolbook
    
    deadline = time.time() + 140 # 140s to be safe
    
    # Function to verify a decomposition
    def verify_decomp(U, V, W, n, m, p):
        # U: (n*m, R)
        # V: (m*p, R)
        # W: (p*n, R)
        # Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        # Note: T_{i,j,k} = 1 for all i,j,k in 3x3x3?
        # No, T_{i,j,k} is 1 if we multiply A_{i,j} and B_{j,k}.
        # So T_{i,j,k} = 1 for all i,j,k.
        # Wait, the tensor is the multiplication tensor.
        # T_{i,j,k} = 1 for all i,j,k in the support.
        # For 3x3x3, all 27 entries are 1.
        # So we need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
        
        # Flatten indices
        # i in 0..2, j in 0..2, k in 0..2
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1) > 1e-6:
                        return False
        return True

    # Function to solve for W given U, V
    def solve_W(U, V, n, m, p):
        # U: (n*m, R), V: (m*p, R)
        # We want W: (p*n, R) such that
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
        # This is a linear system in W.
        # Variables: W[k*n+i, r] for k in 0..p-1, i in 0..n-1, r in 0..R-1
        # Total variables: p*n*R
        # Equations: n*m*p
        # We can set up the matrix A such that A @ W_flat = 1
        
        R = U.shape[1]
        num_vars = p * n * R
        num_eqs = n * m * p
        
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Equation for T_{i,j,k} = 1
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    for r in range(R):
                        var_idx = (k * n + i) * R + r
                        A[eq_idx, var_idx] = U[i*m+j, r] * V[j*p+k, r]
                    eq_idx += 1
        
        # Solve A @ W_flat = b
        # Since it's underdetermined, we can use least squares or find a particular solution.
        # We want a solution with small entries.
        # Use np.linalg.lstsq
        W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        
        # Reshape W_flat to (p*n, R)
        W = W_flat.reshape((p*n, R))
        return W

    # Try to find a decomposition
    for R in range(23, 28):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        while time.time() < deadline and not found and attempts < 5000:
            attempts += 1
            # Random U, V with small integers
            U = np.random.randint(-2, 3, size=(n*m, R))
            V = np.random.randint(-2, 3, size=(m*p, R))
            
            # Solve for W
            W = solve_W(U, V, n, m, p)
            
            # Check if W is close to integers or half-integers
            # We can check if 2*W is close to integers
            W_2 = 2 * W
            if np.allclose(W_2, np.round(W_2), atol=1e-4):
                # Round W to half-integers
                W_rounded = np.round(W_2) / 2
                # Verify
                if verify_decomp(U, V, W_rounded, n, m, p):
                    # Found a valid decomposition
                    found = True
                    # Construct the 3x3x6 decomposition
                    # U is the same for both blocks?
                    # No, U is (n*m, R). For 3x3x6, U is (n*m, 2R)?
                    # Wait, the rank is 2R.
                    # The first R terms correspond to block 1, next R to block 2.
                    # U for block 1 is U, U for block 2 is U.
                    # So U_total = [U, U] (horizontally)
                    # V for block 1 is V, V for block 2 is V (but shifted in k)
                    # V_total: (m*p, 2R)
                    # For k in 0..2, V[j*p+k, r] = V[j*3+k, r]
                    # For k in 3..5, V[j*p+k, r+R] = V[j*3+(k-3), r]
                    # W for block 1 is W, W for block 2 is W (shifted in k)
                    # W_total: (p*n, 2R)
                    # For k in 0..2, W[k*n+i, r] = W[k*3+i, r]
                    # For k in 3..5, W[k*n+i, r+R] = W[(k-3)*3+i, r]
                    
                    U_total = np.hstack([U, U])
                    V_total = np.zeros((m*p, 2*R))
                    W_total = np.zeros((p*n, 2*R))
                    
                    for j in range(m):
                        for k in range(3):
                            V_total[j*p+k, 0:R] = V[j*3+k, 0:R]
                            V_total[j*p+(k+3), R:2*R] = V[j*3+k, 0:R]
                            
                    for k in range(3):
                        for i in range(n):
                            W_total[k*n+i, 0:R] = W_rounded[k*3+i, 0:R]
                            W_total[(k+3)*n+i, R:2*R] = W_rounded[k*3+i, 0:R]
                            
                    # Convert to lists
                    U_list = U_total.tolist()
                    V_list = V_total.tolist()
                    W_list = W_total.tolist()
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
            
        if not found:
            continue
            
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
