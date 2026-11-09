"""Matrix multiplication 3x4x6: search for rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS

# The tensor T has indices (i, j, k) -> (i*m+j, j*p+k, k*n+i) with value 1
# We seek rank r such that T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
# where u_c is in R^(n*m), v_c in R^(m*p), w_c in R^(p*n)
# T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]

def tensor_value(i, j, k):
    """The matrix multiplication tensor: T[i,j,k] = 1 if indices align, else 0.
    Actually for matrix mult (n x m) * (m x p) -> (n x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor has T[i,j,k] = 1 for all valid i,j,k (it's the identity in j).
    The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    where T[i,j,k] = 1 for all i,j,k (since every (i,j,k) triple contributes).
    Wait, no. The tensor for matrix multiplication is:
    T[i,j,k] = 1 (it's just the identity mapping - every triple (i,j,k) is valid)
    So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    """
    return 1

def verify(U, V, W):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 72."""
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

def try_rank_r(r, max_iters=200, tol=1e-6, seed=42):
    """Try to find a rank-r decomposition using iterative refinement.
    We use a simple approach: initialize randomly, then use alternating least squares
    on the tensor equation.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n * m, r) * 0.1
    V = rng.randn(m * p, r) * 0.1
    W = rng.randn(p * n, r) * 0.1
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Flatten the constraint: for each (i,j,k), we have one equation.
    # Total equations: n*m*p = 72
    # Variables: r columns in each of U, V, W -> 3*r*(nm+mp+pn)/r ... 
    # Actually variables are U (n*m*r), V (m*p*r), W (p*n*r)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Alternating updates: fix V and W, solve for U; fix U and W, solve for V; etc.
        # For U: for each (i,j), we need sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
        # This is: U[i*m+j,:] * diag stuff ... 
        # Actually: for fixed i,j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for k=0..p-1
        # Let A_k[c] = V[j*p+k,c] * W[k*n+i,c] for c=0..r-1
        # Then: sum_c U[i*m+j,c] * A_k[c] = 1 for k=0..p-1
        # This is: A * U[i*m+j,:]^T = ones(p) where A is p x r
        
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    for c in range(r):
                        A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # Solve A @ x = ones(p) for x (least squares)
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                except:
                    x = np.ones(r) / r
                U[i*m+j, :] = x
        
        for j in range(m):
            for k in range(p):
                A = np.zeros((n * m, r))  # wrong, should be over (i) for fixed j,k... 
                # For fixed j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                # Let B_i[c] = U[i*m+j,c] * W[k*n+i,c]
                # sum_c V[j*p+k,c] * B_i[c] = 1 for i=0..n-1
                B = np.zeros((n, r))
                for i in range(n):
                    for c in range(r):
                        B[i, c] = U[i*m+j, c] * W[k*n+i, c]
                try:
                    x, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                except:
                    x = np.ones(r) / r
                V[j*p+k, :] = x
        
        for i in range(n):
            for k in range(p):
                # For fixed i,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                # Let C_j[c] = U[i*m+j,c] * V[j*p+k,c]
                # sum_c W[k*n+i,c] * C_j[c] = 1 for j=0..m-1
                C = np.zeros((m, r))
                for j in range(m):
                    for c in range(r):
                        C[j, c] = U[i*m+j, c] * V[j*p+k, c]
                try:
                    x, residuals, rank_c, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                except:
                    x = np.ones(r) / r
                W[k*n+i, :] = x
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    if best_err < tol:
        # Convert to exact fractions if possible
        U_frac = [[Fraction(int(round(U_frac_val * 1000) / 1000)) for U_frac_val in row] for row in best_U]
        V_frac = [[Fraction(int(round(V_frac_val * 1000) / 1000)) for V_frac_val in row] for row in best_V]
        W_frac = [[Fraction(int(round(W_frac_val * 1000) / 1000)) for W_frac_val in row] for row in best_W]
        
        # Verify with fractions
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook (rank 72)
    best = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank
    # Known lower bounds: for n x m x p, rank >= max(n*m, m*p, p*n, (n+m+p-1)*something)
    # For 3x4x6: n*m=12, m*p=24, p*n=18, so rank >= 24 at minimum
    # Strassen-like methods might reduce below 72
    
    # Try ranks from a reasonable lower bound up to 71
    # Use multiple seeds for robustness
    max_rank_to_try = best_rank - 1
    
    for r in range(24, max_rank_to_try + 1):
        if time.time() > deadline:
            break
        found = False
        for seed in range(10):
            if time.time() > deadline:
                break
            result = try_rank_r(r, max_iters=100, tol=1e-7, seed=seed)
            if result is not None:
                best = result
                best_rank = r
                found = True
                break
        if found:
            # Found rank r, try to go lower
            continue
    
    return best
