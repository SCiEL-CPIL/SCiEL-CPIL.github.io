import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def _verify(U, V, W, n, m, p, rank):
    """Verify if U, V, W exactly decompose the tensor using exact arithmetic."""
    # Convert to Fractions for exact verification
    U_frac = [[Fraction(int(round(x*2)))/2 if x != 0 else 0 for x in row] for row in U]
    V_frac = [[Fraction(int(round(x*2)))/2 if x != 0 else 0 for x in row] for row in V]
    W_frac = [[Fraction(int(round(x*2)))/2 if x != 0 else 0 for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                if s != 1:
                    return False
    return True

def _als_decompose(n, m, p, rank, max_iters=200, tol=1e-8):
    """
    Attempt to find a rank-R decomposition using ALS with multiple restarts.
    Returns (U, V, W) as lists of lists if successful, else None.
    """
    # Try multiple random initializations
    for init_trial in range(5):
        # Initialize with random values in [-1, 1]
        U = np.random.uniform(-1, 1, size=(n*m, rank))
        V = np.random.uniform(-1, 1, size=(m*p, rank))
        W = np.random.uniform(-1, 1, size=(p*n, rank))
        
        # Scale to have reasonable magnitude
        U *= 2
        V *= 2
        W *= 2

        prev_err = float('inf')
        for it in range(max_iters):
            # Update U
            for i in range(n):
                for j in range(m):
                # Construct matrix A (rank x p)
                    A = np.zeros((rank, p))
                    for k in range(p):
                        A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A @ u = ones(p)
                    try:
                        u, residuals, rank_a, s = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                    except np.linalg.LinAlgError:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # Construct matrix B (rank x n)
                    B = np.zeros((rank, n))
                    for i in range(n):
                        B[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, residuals, rank_b, s = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                    except np.linalg.LinAlgError:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # Construct matrix C (rank x m)
                    C = np.zeros((rank, m))
                    for j in range(m):
                        C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w, residuals, rank_c, s = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                    except np.linalg.LinAlgError:
                        pass
            
            # Calculate error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (s - 1.0)**2
            err = np.sqrt(err / (n*m*p))
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, rank):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try ranks from 19 upwards (known optimal is 19)
    for rank in range(19, 25):
        if time.time() > deadline:
            break
            
        # Try multiple initializations
        num_trials = 30 if rank == 19 else 15
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            result = _als_decompose(n, m, p, rank, max_iters=150, tol=1e-9)
            if result:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
        
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
