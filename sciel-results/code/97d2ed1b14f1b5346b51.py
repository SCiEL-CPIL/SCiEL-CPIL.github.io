"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy import linalg
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Build target tensor
    target_tensor = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_tensor[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-8):
        """Check if decomposition is exact with half-integer entries."""
        r = U.shape[1]
        R = np.einsum('ak,bk,ck->abc', U, V, W)
        if np.max(np.abs(R - target_tensor)) > tol:
            return False
        for mat in [U, V, W]:
            if not np.allclose(2.0 * mat, np.round(2.0 * mat), atol=tol):
                return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    best_result = None
    best_rank = 42

    # Try structured initializations based on block structure
    # For each rank, try multiple strategies
    
    def try_rank(rank, max_attempts=50):
        """Try to find a valid decomposition at given rank."""
        if time.time() > deadline - 10:
            return None
        
        param_size = rank * (nm + mp + pn)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Different initialization strategies
            if attempt % 3 == 0:
                # Random small
                np.random.seed(hash((rank, attempt)) % 2**32)
                theta0 = np.random.randn(param_size) * 0.3
            elif attempt % 3 == 1:
                # Random from {-1, -0.5, 0, 0.5, 1}
                np.random.seed(hash((rank, attempt + 1000)) % 2**32)
                theta0 = np.random.choice([-1, -0.5, 0, 0.5, 1], size=param_size)
            else:
                # Sparse random
                np.random.seed(hash((rank, attempt + 2000)) % 2**32)
                theta0 = np.random.randn(param_size) * 0.5
                mask = np.random.random(param_size) < 0.3
                theta0[~mask] = 0
            
            try:
                def residuals(theta):
                    U_flat = theta[:nm * rank]
                    V_flat = theta[nm * rank : nm * rank + mp * rank]
                    W_flat = theta[nm * rank + mp * rank :]
                    U = U_flat.reshape(nm, rank)
                    V = V_flat.reshape(mp, rank)
                    W = W_flat.reshape(pn, rank)
                    R = np.einsum('ak,bk,ck->abc', U, V, W)
                    return (R - target_tensor).flatten()
                
                res = least_squares(
                    residuals,
                    theta0,
                    method='trf',
                    max_nfev=500,
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12
                )
                
                if res.cost < 1e-10:
                    U_flat = res.x[:nm * rank]
                    V_flat = res.x[nm * rank : nm * rank + mp * rank]
                    W_flat = res.x[nm * rank + mp * rank :]
                    U = U_flat.reshape(nm, rank)
                    V = V_flat.reshape(mp, rank)
                    W = W_flat.reshape(pn, rank)
                    
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2.0
                    V_r = np.round(V * 2) / 2.0
                    W_r = np.round(W * 2) / 2.0
                    
                    if verify(U_r, V_r, W_r):
                        return (U_r, V_r, W_r)
                        
            except Exception:
                pass
        
        return None

    # Try ranks from low to high
    for rank in range(25, 42):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, max_attempts=30)
        if result is not None:
            best_result = result
            best_rank = rank
            break

    if best_result is None:
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = 42

    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
