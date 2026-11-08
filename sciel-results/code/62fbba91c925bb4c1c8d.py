import numpy as np
import time
from itertools import product

# Matrix multiplication 3x3 by 3x4
# A is 3x3, B is 3x4, C is 3x4
# T_{i j, j' k, i' k'} = delta_{i,i'} delta_{j,j'} delta_{k,k'}

def construct():
    n, m, p = 3, 3, 4
    deadline = time.time() + 110.0  # Leave margin for 120s limit

    # Schoolbook decomposition (rank 36)
    def schoolbook():
        R = n * m * p
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        W = [[0]*R for _ in range(p*n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    # Verify a decomposition using numpy
    def verify(U, V, W):
        n, m, p = 3, 3, 4
        R = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build tensor T from U, V, W
        # T[i*m+j, j'*p+k, k'*n+i'] = sum_r U[i*m+j,r] * V[j'*p+k,r] * W[k'*n+i',r]
        # We need this to equal delta[i,i'] * delta[j,j'] * delta[k,k']
        
        # Construct the full tensor
        tensor = np.zeros((n*m, m*p, p*n))
        for r in range(R):
            outer_uv = np.outer(U_arr[:, r], V_arr[:, r])
            # outer_uv is (n*m, m*p)
            # For each row (i*m+j) and col (j'*p+k), multiply by W[k'*n+i']
            for ip in range(p*n):  # i', k'
                w_val = W_arr[ip, r]
                if w_val == 0:
                    continue
                tensor[:, :, ip] += outer_uv * w_val
        
        # Expected tensor
        expected = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # i=m row index in first dim: i*m + j
                    # j'=j, k=k in second dim: j*p + k
                    # i'=i, k'=k in third dim: k*n + i
                    expected[i*m+j, j*p+k, k*n+i] = 1.0
        
        return np.allclose(tensor, expected, atol=1e-10)

    # Try to find a lower rank decomposition
    # Known lower bound for 3x3x3 is 23. For 3x3x4, likely around 20-25.
    # We'll try ranks from 20 up to 35.
    
    # Use a randomized approach with exact verification
    for R in range(20, 36):
        if time.time() > deadline - 20:
            break
        
        # Multiple attempts per rank
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random U and V with small integer entries
            # Use {-1, 0, 1} entries
            np.random.seed(attempt * 1000 + R)
            U_rand = np.random.randint(-1, 2, (n*m, R))
            V_rand = np.random.randint(-1, 2, (m*p, R))
            
            # Build matrix A where A[(i*m+j)*(m*p) + (j'*p+k), r] = U[i*m+j, r] * V[j'*p+k, r]
            A = np.zeros((n*m*m*p, R))
            for r in range(R):
                outer = np.outer(U_rand[:, r], V_rand[:, r])
                A[:, r] = outer.flatten()
            
            # Check rank of A
            rank_A = np.linalg.matrix_rank(A)
            if rank_A < R:
                continue
            
            # For each (i', k'), solve A x = b
            W_sol = np.zeros((p*n, R))
            valid = True
            for i_prime in range(n):
                for k_prime in range(p):
                    # b is a vector of length n*m*m*p
                    # b[(i*m+j)*(m*p) + (j'*p+k)] = 1 if i==i', j==j', k==k'
                    b = np.zeros(n*m*m*p)
                    for j in range(m):
                        idx = (i_prime*m + j) * (m*p) + (j*p + k_prime)
                        b[idx] = 1.0
                    
                    # Solve A x = b
                    try:
                        x, residuals, rank_x, s = np.linalg.lstsq(A, b, rcond=None)
                        if len(residuals) > 0 and residuals[0] > 1e-9:
                            valid = False
                            break
                        # Check if x is half-integer
                        x_scaled = x * 2
                        if not np.allclose(x_scaled, np.round(x_scaled), atol=1e-9):
                            valid = False
                            break
                        W_sol[k_prime*n + i_prime] = x
                    except Exception:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Convert to lists
                U_list = [[int(val) for val in row] for row in U_rand]
                V_list = [[int(val) for val in row] for row in V_rand]
                W_list = [[float(val * 2) / 2 for val in row] for row in W_sol]
                
                # Verify with exact check
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    return schoolbook()
