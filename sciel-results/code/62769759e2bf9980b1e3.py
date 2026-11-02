import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    t_start = time.time()
    deadline = t_start + 145  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 12
    pn = p * n  # 12
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def round_to_half(x):
        """Round each entry to nearest half-integer."""
        return np.round(x * 2.0) / 2.0
    
    def als_decompose(rank, max_iter=200, seed=0):
        """ALS decomposition with exact rounding verification."""
        rng = np.random.RandomState(seed)
        # Initialize with small random values
        U = rng.randn(nm, rank) * 0.1
        V = rng.randn(mp, rank) * 0.1
        W = rng.randn(pn, rank) * 0.1
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Solve for W via least squares
            A_W = np.zeros((n*m*p, pn*rank))
            b = np.ones(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            A_W[idx, k*n+i*rank + c] = U[i*m+j, c] * V[j*p+k, c]
                        idx += 1
            
            W_flat, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
            W = W_flat.reshape(pn, rank)
            
            # Solve for U
            A_U = np.zeros((n*m*p, nm*rank))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            A_U[idx, i*m+j*rank + c] = V[j*p+k, c] * W[k*n+i, c]
                        idx += 1
            U_flat, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
            U = U_flat.reshape(nm, rank)
            
            # Solve for V
            A_V = np.zeros((n*m*p, mp*rank))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            A_V[idx, j*p+k*rank + c] = U[i*m+j, c] * W[k*n+i, c]
                        idx += 1
            V_flat, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
            V = V_flat.reshape(mp, rank)
            
            # Round to half-integers
            U_r = round_to_half(U)
            V_r = round_to_half(V)
            W_r = round_to_half(W)
            
            # Check if rounded version works
            if verify(U_r, V_r, W_r, tol=1e-6):
                return U_r, V_r, W_r
        
        return None
    
    # Try ranks from low to high
    for rank in range(12, 40):
        if time.time() > deadline - 5:
            break
        
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            result = als_decompose(rank, max_iter=100, seed=seed)
            if result is not None:
                U, V, W = result
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
