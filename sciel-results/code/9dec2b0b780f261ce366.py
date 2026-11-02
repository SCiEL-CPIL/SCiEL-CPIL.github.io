import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to find a low-rank decomposition
    # Start from a reasonable lower bound and go up
    # Theoretical lower bound for 2x5x6 is not well known, but schoolbook is 60
    # Try ranks from 30 to 60
    
    best_result = None
    
    for r in range(30, 61):
        if time.time() > deadline:
            break
            
        # Try multiple structured initializations
        num_trials = 200 if r <= 40 else 100
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Use different structured patterns for U and V
            if trial % 4 == 0:
                # Random ±1
                U = np.random.choice([-1, 1], size=(n * m, r))
                V = np.random.choice([-1, 1], size=(m * p, r))
            elif trial % 4 == 1:
                # Random with more values
                U = np.random.choice([-1, 0, 1], size=(n * m, r))
                V = np.random.choice([-1, 0, 1], size=(m * p, r))
            elif trial % 4 == 2:
                # Structured: use Kronecker-like patterns
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                for alpha in range(r):
                    # Pattern based on alpha
                    a = alpha % n
                    b = (alpha // n) % m
                    c = (alpha // (n * m)) % p
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j, alpha] = 1 if (i == a and j == b) else 0
                    for j in range(m):
                        for k in range(p):
                            V[j * p + k, alpha] = 1 if (j == b and k == c) else 0
            else:
                # Random with scaling
                U = np.random.choice([-2, -1, 1, 2], size=(n * m, r))
                V = np.random.choice([-2, -1, 1, 2], size=(m * p, r))
            
            W = np.zeros((p * n, r))
            valid = True
            
            # Solve for W for each (i, k) pair
            for i in range(n):
                for k in range(p):
                    # Construct M_ik of shape (m, r)
                    U_part = U[i * m: (i + 1) * m, :]
                    V_rows = [j * p + k for j in range(m)]
                    V_part = V[V_rows, :]
                    
                    M_ik = U_part * V_part
                    # Solve M_ik @ w = ones(m)
                    try:
                        w, residuals, rank_m, s = np.linalg.lstsq(M_ik, np.ones(m), rcond=1e-10)
                    except:
                        valid = False
                        break
                    
                    # Check if w is close to half-integers
                    two_w = 2 * w
                    rounded_two_w = np.round(two_w)
                    if not np.allclose(two_w, rounded_two_w, atol=1e-6):
                        valid = False
                        break
                    
                    W[k * n + i, :] = w
            
            if valid:
                # Verify the decomposition exactly
                U_arr = np.array(U, dtype=float)
                V_arr = np.array(V, dtype=float)
                W_arr = np.array(W, dtype=float)
                
                # Check: for all i,j,k: sum_alpha U[i*m+j, alpha] * V[j*p+k, alpha] * W[k*n+i, alpha] == delta(i,j,k)
                # Wait, that's not right. The tensor decomposition is:
                # T[i,j,k] = sum_alpha U[i,j,alpha] * V[j,k,alpha] * W[k,i,alpha]
                # where T[i,j,k] = 1 if this is the (i,j) entry of A*B... 
                
                # Actually, the matrix multiplication tensor is:
                # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
                # So the tensor T has indices (i, j, k) with T[i,j,k] = 1
                # And the decomposition is: T[i,j,k] = sum_alpha U[i*m+j, alpha] * V[j*p+k, alpha] * W[k*n+i, alpha]
                
                # Let's verify: for each (i,j,k), check if sum_alpha U[i*m+j,alpha] * V[j*p+k,alpha] * W[k*n+i,alpha] equals 1
                # But wait, that's not how matrix multiplication works.
                
                # The correct tensor decomposition for matrix multiplication C = AB is:
                # C[i,k] = sum_j A[i,j] B[j,k]
                # This is a bilinear map: (A,B) -> C
                # The tensor is T_{i,j,k} where T_{i,j,k} = 1 for all i,j,k (it's the "multiplication tensor")
                # And the decomposition is: T_{i,j,k} = sum_alpha U_{i,j,alpha} * V_{j,k,alpha} * W_{k,i,alpha}
                # Wait, no. The standard tensor decomposition is:
                # T_{i,j,k} = sum_alpha U_{i,alpha} * V_{j,alpha} * W_{k,alpha}
                # But here the indices are coupled: U depends on (i,j), V on (j,k), W on (k,i)
                
                # So the check is: for each (i,j,k):
                # sum_alpha U[i*m+j, alpha] * V[j*p+k, alpha] * W[k*n+i, alpha] should equal 1
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for alpha in range(r):
                                s += U_arr[i * m + j, alpha] * V_arr[j * p + k, alpha] * W_arr[k * n + i, alpha]
                            if abs(s - 1.0) > 1e-6:
                                error += 1
                                break
                    if error > 0:
                        break
                if error > 0:
                    continue
                
                # Found a valid decomposition!
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
