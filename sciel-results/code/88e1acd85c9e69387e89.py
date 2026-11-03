import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Constructs the matrix multiplication tensor decomposition for 2x3 by 3x5.
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Define the tensor support
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.append((a, b, c))
    
    support_set = set(support)
    
    # Target tensor values
    target = {}
    for (a, b, c) in support:
        target[(a, b, c)] = 1.0
    
    def verify(U, V, W, r):
        """Verify that U, V, W give a valid rank-r decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check all support entries
        for (a, b, c) in support:
            val = np.sum(U[a] * V[b] * W[c])
            if abs(val - 1.0) > 1e-6:
                return False
        
        # Check all non-support entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) not in support_set:
                        val = np.sum(U[a] * V[b] * W[c])
                        if abs(val) > 1e-6:
                            return False
        
        return True
    
    # Alternating Least Squares approach
    def als_solve(W_init, max_iter=200):
        """Given W, alternate solving for U and V."""
        W = W_init.copy()
        U = np.random.randn(n*m, W.shape[1]) * 0.1
        V = np.random.randn(m*p, W.shape[1]) * 0.1
        
        r = W.shape[1]
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Solve for U: for each a, minimize sum_b,c (sum_r U[a,r] V[b,r] W[c,r] - target(a,b,c))^2
            # This is linear in U[a]
            for a in range(n*m):
                # Build system: for each (b,c), coefficient vector is V[b] * W[c] (element-wise product)
                # We want: U[a] @ (V[b] * W[c]) = target(a,b,c)
                # Stack all such equations
                A_mat = np.zeros((m*p*p*n, r))
                b_vec = np.zeros(m*p*p*n)
                idx = 0
                for b in range(m*p):
                    for c in range(p*n):
                        A_mat[idx] = V[b] * W[c]
                        if (a, b, c) in target:
                            b_vec[idx] = target[(a, b, c)]
                        idx += 1
                # Solve least squares
                U[a], residuals, rank_a, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Solve for V: for each b, minimize sum_a,c (sum_r U[a,r] V[b,r] W[c,r] - target(a,b,c))^2
            for b in range(m*p):
                A_mat = np.zeros((n*m*p*n, r))
                b_vec = np.zeros(n*m*p*n)
                idx = 0
                for a in range(n*m):
                    for c in range(p*n):
                        A_mat[idx] = U[a] * W[c]
                        if (a, b, c) in target:
                            b_vec[idx] = target[(a, b, c)]
                        idx += 1
                V[b], residuals, rank_b, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Optional: update W similarly
            for c in range(p*n):
                A_mat = np.zeros((n*m*m*p, r))
                b_vec = np.zeros(n*m*m*p)
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        A_mat[idx] = U[a] * V[b]
                        if (a, b, c) in target:
                            b_vec[idx] = target[(a, b, c)]
                        idx += 1
                W[c], residuals, rank_c, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        return U, V, W
    
    # Try ranks from 25 to 29
    for r in range(25, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + r)
            
            # Initialize W with random values
            W_init = np.random.randn(p*n, r) * 0.5
            
            result = als_solve(W_init, max_iter=100)
            if result is None:
                continue
            
            U, V, W = result
            
            if verify(U.tolist(), V.tolist(), W.tolist(), r):
                found = True
                best_rank = r
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if found:
            break
    
    return best_result
