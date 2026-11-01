import numpy as np
import time

DIMENSIONS = [2, 2, 7]
START_TIME = time.time()
TIME_LIMIT = 110  # Leave 10s margin for 120s limit

def construct():
    n, m, p = DIMENSIONS
    # Generate valid indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                indices.append((I, J, K))
    
    T_flat = np.ones(len(indices))
    
    # Try ranks from 23 to 28
    for r in range(23, 29):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        # Try a few random initializations
        for attempt in range(20):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            np.random.seed(attempt)
            U = np.random.choice([-1, 1], size=(n*m, r))
            V = np.random.choice([-1, 1], size=(m*p, r))
            
            # Build A
            # A shape: (num_eq, num_vars)
            # num_vars = p*n * r
            num_vars = p * n * r
            A = np.zeros((len(indices), num_vars))
            
            for e, (I, J, K) in enumerate(indices):
                for c in range(r):
                    # W index is K * r + c ?
                    # W is (p*n, r). Flat index for W[K, c] is K*r + c.
                    A[e, K * r + c] = U[I, c] * V[J, c]
            
            # Check rank
            # If rank < 28, no solution for general T
            if np.linalg.matrix_rank(A) < len(indices):
                continue
                
            # Solve A W_flat = T_flat
            W_flat, residuals, rank, s = np.linalg.lstsq(A, T_flat, rcond=None)
            
            # Check residual
            # residuals is an array. Check if it has elements and if the first one is large.
            if residuals.size > 0 and residuals[0] > 1e-9:
                continue
                
            # Reshape W
            W = W_flat.reshape(p*n, r)
            
            # Round to nearest 0.5
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            # Compute tensor
            valid = True
            for e, (I, J, K) in enumerate(indices):
                val = np.dot(U[I, :], V[J, :]) # No, sum over c
                # val = sum_c U[I,c] V[J,c] W[K,c]
                val = 0
                for c in range(r):
                    val += U[I, c] * V[J, c] * W_rounded[K, c]
                if abs(val - 1) > 1e-9:
                    valid = False
                    break
            
            if valid:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_rounded.tolist()
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
